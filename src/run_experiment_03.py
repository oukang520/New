"""Run Experiment 3: cMHN-to-Rel-ObsTQ transition interface.

Each selected cancer cohort is trained independently on the fixed p15 event
matrix produced by Experiments 1-2. The script selects the L1 penalty strength
by cross-validation, fits a final cMHN, and converts its event hazards into
one-event genotype and stage-state transition probabilities.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import platform
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

import figure_style

COHORT_COLORS = {
    "AACR_LUAD": "#4477AA",
    "AACR_COAD": "#CC6677",
    "AACR_IDC": "#228833",
}
PROBABILITY_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "transition_probability",
    ["#F7FBFF", "#C7E9E5", "#5AB4AC", "#238B8D", "#005F73"],
)
PROBABILITY_NORM = mcolors.PowerNorm(gamma=0.45, vmin=0, vmax=1)

BIOLOGY_CHECKS = {
    "AACR_LUAD": [
        ("EGFR", "KRAS", "canonical alternative RTK-RAS drivers"),
        ("KRAS", "STK11", "KRAS-associated tumor-suppressor context"),
        ("KRAS", "KEAP1", "KRAS-associated oxidative-stress context"),
    ],
    "AACR_COAD": [
        ("KRAS", "BRAF", "canonical alternative MAPK drivers"),
        ("APC", "KRAS", "WNT-to-MAPK progression context"),
        ("TP53", "SMAD4", "late colorectal progression context"),
    ],
    "AACR_IDC": [
        ("TP53", "PIK3CA", "breast subtype-dependent driver context"),
        ("GATA3", "PIK3CA", "luminal-lineage context"),
        ("MAP3K1", "PIK3CA", "luminal MAPK-PI3K context"),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Rel-ObsTQ-MHN Experiment 3.")
    parser.add_argument("--config", default="configs/experiment_03.yaml")
    parser.add_argument(
        "--dataset-config", default="configs/selected_experiment_datasets.yaml"
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        help="Optional dataset names. Defaults to all selected cohorts.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def configure_plotting(config: dict) -> None:
    figure_style.configure_matplotlib(config)


def setup_logging(root: Path) -> None:
    (root / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=root / "logs" / "experiment_03.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def save_figure(fig: plt.Figure, base_path: Path, dpi: int) -> None:
    figure_style.save_figure_panels(fig, base_path, {"plot": {"dpi": dpi}}, dpi=dpi)


def load_mhn_backend():
    import mhn as mhn_backend
    from mhn.optimizers import Optimizer as MhnOptimizer

    return mhn_backend, MhnOptimizer


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
    )


def genotype_signature(row: np.ndarray, events: list[str]) -> str:
    present = [events[i] for i, value in enumerate(row) if int(value) == 1]
    return "+".join(present) if present else "WT"


def signature_to_vector(signature: str, events: list[str]) -> np.ndarray:
    present = set() if signature == "WT" else set(signature.split("+"))
    return np.array([int(event in present) for event in events], dtype=np.int32)


def transition_probability_matrix(
    transitions: pd.DataFrame,
    genotypes: list[str],
    events: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    probability = (
        transitions[transitions["source_genotype"].isin(genotypes)]
        .pivot(index="source_genotype", columns="event_added", values="probability")
        .reindex(index=genotypes, columns=events)
    )
    impossible = pd.DataFrame(False, index=genotypes, columns=events)
    for genotype in genotypes:
        present = set() if genotype == "WT" else set(genotype.split("+"))
        impossible.loc[genotype, :] = [event in present for event in events]
    probability = probability.mask(impossible).fillna(0)
    return probability, impossible


def supported_edges(transitions: pd.DataFrame, count: int) -> pd.DataFrame:
    work = transitions.copy()
    work["edge_support"] = work["source_sample_fraction"] * work["probability"]
    return work.sort_values(
        ["edge_support", "source_sample_count", "probability"],
        ascending=[False, False, False],
    ).head(count)


def compact_edge_label(source: str, event_added: str) -> str:
    source_events = [] if source == "WT" else source.split("+")
    if len(source_events) > 2:
        source = "+".join(source_events[:2]) + "+..."
    return f"{source} -> {event_added}"


def build_genotype_tables(
    matrix: pd.DataFrame, log_theta: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = matrix.columns.astype(str).tolist()
    values = matrix.to_numpy(dtype=np.int32)
    signatures = pd.Series(
        [genotype_signature(row, events) for row in values], name="source_genotype"
    )
    counts = signatures.value_counts()
    rows: list[dict] = []
    genotype_rows: list[dict] = []

    for signature, count in counts.items():
        state = signature_to_vector(signature, events)
        absent = np.flatnonzero(state == 0)
        event_count = int(state.sum())
        if absent.size == 0:
            genotype_rows.append(
                {
                    "source_genotype": signature,
                    "sample_count": int(count),
                    "sample_fraction": float(count / len(matrix)),
                    "event_count": event_count,
                    "outgoing_events": 0,
                    "total_outgoing_hazard": 0.0,
                    "transition_entropy": 0.0,
                }
            )
            continue

        log_hazards = np.array(
            [
                log_theta[event_idx, event_idx]
                + log_theta[event_idx, state.astype(bool)].sum()
                for event_idx in absent
            ],
            dtype=float,
        )
        offset = float(log_hazards.max())
        scaled = np.exp(log_hazards - offset)
        probabilities = scaled / scaled.sum()
        hazards = np.exp(np.clip(log_hazards, -700, 700))
        entropy = float(-(probabilities * np.log(probabilities)).sum())

        for idx, event_idx in enumerate(absent):
            target = state.copy()
            target[event_idx] = 1
            rows.append(
                {
                    "source_genotype": signature,
                    "target_genotype": genotype_signature(target, events),
                    "event_added": events[event_idx],
                    "source_event_count": event_count,
                    "target_event_count": event_count + 1,
                    "source_sample_count": int(count),
                    "source_sample_fraction": float(count / len(matrix)),
                    "log_hazard": float(log_hazards[idx]),
                    "hazard": float(hazards[idx]),
                    "probability": float(probabilities[idx]),
                    "target_observed_in_data": False,
                }
            )

        genotype_rows.append(
            {
                "source_genotype": signature,
                "sample_count": int(count),
                "sample_fraction": float(count / len(matrix)),
                "event_count": event_count,
                "outgoing_events": int(absent.size),
                "total_outgoing_hazard": float(hazards.sum()),
                "transition_entropy": entropy,
            }
        )

    transitions = pd.DataFrame(rows)
    observed = set(counts.index.astype(str))
    if not transitions.empty:
        transitions["target_observed_in_data"] = transitions[
            "target_genotype"
        ].isin(observed)
        transitions = transitions.sort_values(
            ["source_sample_count", "source_genotype", "probability"],
            ascending=[False, True, False],
        ).reset_index(drop=True)
    genotype_summary = pd.DataFrame(genotype_rows).sort_values(
        ["sample_count", "source_genotype"], ascending=[False, True]
    )
    return genotype_summary, transitions


def expand_state_transitions(
    genotype_transitions: pd.DataFrame,
    state_table: pd.DataFrame,
    usable_stages: list[str],
) -> pd.DataFrame:
    source_states = (
        state_table.groupby(["stage_group", "genotype_signature"], dropna=False)
        .size()
        .rename("source_state_count")
        .reset_index()
        .rename(
            columns={
                "stage_group": "source_stage",
                "genotype_signature": "source_genotype",
            }
        )
    )
    source_states["source_stage"] = source_states["source_stage"].fillna("unknown")
    source_states["source_genotype"] = source_states["source_genotype"].fillna("WT")
    target_observed = set(
        zip(
            source_states["source_stage"].astype(str),
            source_states["source_genotype"].astype(str),
        )
    )
    expanded = source_states.merge(
        genotype_transitions, on="source_genotype", how="inner", validate="many_to_many"
    )
    expanded["target_stage"] = expanded["source_stage"]
    expanded["source_state"] = (
        expanded["source_stage"].astype(str)
        + "::"
        + expanded["source_genotype"].astype(str)
    )
    expanded["target_state"] = (
        expanded["target_stage"].astype(str)
        + "::"
        + expanded["target_genotype"].astype(str)
    )
    expanded["stage_transition_rule"] = "same_stage_genotype_transition"
    expanded["source_stage_usable"] = expanded["source_stage"].isin(usable_stages)
    expanded["target_observed_in_same_stage"] = [
        (stage, genotype) in target_observed
        for stage, genotype in zip(
            expanded["target_stage"], expanded["target_genotype"]
        )
    ]
    columns = [
        "source_state",
        "target_state",
        "source_genotype",
        "target_genotype",
        "event_added",
        "probability",
        "hazard",
        "log_hazard",
        "source_stage",
        "target_stage",
        "stage_transition_rule",
        "source_state_count",
        "source_sample_count",
        "source_sample_fraction",
        "source_event_count",
        "target_event_count",
        "source_stage_usable",
        "target_observed_in_data",
        "target_observed_in_same_stage",
    ]
    return expanded[columns].sort_values(
        ["source_state_count", "source_state", "probability"],
        ascending=[False, True, False],
    )


def biology_sanity_table(
    dataset: str, events: list[str], log_theta: np.ndarray
) -> pd.DataFrame:
    event_index = {event: idx for idx, event in enumerate(events)}
    rows = []
    for first, second, context in BIOLOGY_CHECKS.get(dataset, []):
        if first not in event_index or second not in event_index:
            rows.append(
                {
                    "event_a": first,
                    "event_b": second,
                    "biological_context": context,
                    "log_effect_a_on_b": np.nan,
                    "log_effect_b_on_a": np.nan,
                    "qualitative_pattern": "not_in_p15_panel",
                    "interpretation_scope": "descriptive_only",
                }
            )
            continue
        a, b = event_index[first], event_index[second]
        a_on_b = float(log_theta[b, a])
        b_on_a = float(log_theta[a, b])
        if a_on_b < 0 and b_on_a < 0:
            pattern = "bidirectional_negative"
        elif a_on_b > 0 and b_on_a > 0:
            pattern = "bidirectional_positive"
        elif a_on_b * b_on_a < 0:
            pattern = "directionally_mixed"
        else:
            pattern = "weak_or_zero"
        rows.append(
            {
                "event_a": first,
                "event_b": second,
                "biological_context": context,
                "log_effect_a_on_b": a_on_b,
                "log_effect_b_on_a": b_on_a,
                "qualitative_pattern": pattern,
                "interpretation_scope": (
                    "descriptive_only; cross-sectional MHN effects are not causal"
                ),
            }
        )
    return pd.DataFrame(rows)


def train_dataset(
    dataset: str,
    display_name: str,
    config: dict,
    result_root: Path,
    force: bool,
) -> dict:
    dataset_root = result_root / dataset
    tables = dataset_root / "tables"
    figures = dataset_root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    metadata_path = dataset_root / "model_metadata.json"
    if metadata_path.exists() and not force:
        logging.info("%s skipped because completed metadata exists", dataset)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        plot_dataset(
            dataset,
            display_name,
            pd.read_csv(tables / "cv_likelihood.tsv", sep="\t"),
            pd.read_csv(tables / "theta.tsv", sep="\t", index_col=0),
            pd.read_csv(tables / "genotype_summary.tsv", sep="\t"),
            pd.read_csv(tables / "genotype_transition.tsv", sep="\t"),
            figures / "Figure_E3_MHN_interface",
            config,
        )
        return metadata

    source_root = (
        Path(config["experiments_01_02_root"])
        / dataset
        / "experiment_01_data_preparation"
        / "tables"
    )
    panel_size = int(config["panel_size"])
    matrix = pd.read_csv(source_root / f"mhn_training_matrix_p{panel_size}.csv")
    panel = pd.read_csv(source_root / f"event_panel_p{panel_size}.csv")
    state_table = pd.read_csv(source_root / f"state_table_p{panel_size}.csv")
    events = panel["event"].astype(str).tolist()
    if matrix.columns.astype(str).tolist() != events:
        raise ValueError(f"{dataset}: p15 matrix and panel column order differ")
    if not matrix.isin([0, 1]).all().all():
        raise ValueError(f"{dataset}: MHN matrix is not binary")

    mhn_backend, MhnOptimizer = load_mhn_backend()
    seed = int(config["random_seed"])
    np.random.seed(seed)
    mhn_backend.set_seed(seed)
    settings = config["mhn"]
    optimizer = MhnOptimizer(MhnOptimizer.MHNType.cMHN)
    optimizer.set_device(optimizer.Device.CPU)
    optimizer.set_penalty(optimizer.Penalty.L1)
    optimizer.load_data_matrix(matrix.astype(np.int32))

    n = len(matrix)
    multipliers = np.array(settings["lambda_multipliers"], dtype=float)
    start = time.time()
    score_frames = []
    grid_expansions = 0
    while True:
        np.random.seed(seed)
        chosen_lambda, current_scores = optimizer.lambda_from_cv(
            lambda_vector=multipliers / n,
            nfolds=int(settings["cv_folds"]),
            return_lambda_scores=True,
            pick_1se=bool(settings["pick_1se"]),
            show_progressbar=False,
        )
        current_scores["grid_expansion"] = grid_expansions
        score_frames.append(current_scores)
        chosen_multiplier = chosen_lambda * n
        at_low = np.isclose(chosen_multiplier, multipliers.min())
        at_high = np.isclose(chosen_multiplier, multipliers.max())
        can_expand = (
            bool(settings.get("expand_boundary_grid", True))
            and grid_expansions < int(settings.get("max_grid_expansions", 3))
            and (at_low or at_high)
        )
        if not can_expand:
            break
        ratio = float(np.median(multipliers[1:] / multipliers[:-1]))
        if at_high:
            multipliers = multipliers * ratio
        else:
            multipliers = multipliers / ratio
        grid_expansions += 1
    cv_seconds = time.time() - start
    cv_scores = (
        pd.concat(score_frames, ignore_index=True)
        .assign(
            lambda_multiplier=lambda frame: frame["Lambda Value"] * n,
            lambda_multiplier_key=lambda frame: np.round(
                frame["Lambda Value"] * n, 8
            ),
        )
        .sort_values(["lambda_multiplier_key", "grid_expansion"])
        .drop_duplicates("lambda_multiplier_key", keep="last")
        .reset_index(drop=True)
    )
    cv_scores = cv_scores.rename(
        columns={
            "Lambda Value": "lambda",
            "Mean Score": "mean_test_log_likelihood",
            "Standard Error": "standard_error",
        }
    )
    cv_scores["lambda_multiplier"] = cv_scores["lambda"] * n
    cv_scores["selected"] = False
    cv_scores.loc[
        (cv_scores["lambda"] - chosen_lambda).abs().idxmin(), "selected"
    ] = True
    cv_scores = cv_scores.drop(columns=["lambda_multiplier_key"])
    cv_scores.to_csv(tables / "cv_likelihood.tsv", sep="\t", index=False)

    start = time.time()
    model = optimizer.train(
        lam=float(chosen_lambda),
        maxit=int(settings["max_iterations"]),
        reltol=float(settings["relative_tolerance"]),
        round_result=False,
    )
    fit_seconds = time.time() - start
    log_theta = np.asarray(model.log_theta, dtype=float)
    theta = pd.DataFrame(log_theta, index=events, columns=events)
    theta.index.name = "target_event"
    theta.to_csv(tables / "theta.tsv", sep="\t")
    np.exp(np.clip(theta, -700, 700)).to_csv(
        tables / "theta_hazard_ratio.tsv", sep="\t"
    )

    event_baseline = pd.DataFrame(
        {
            "event": events,
            "sample_count": matrix.sum(axis=0).astype(int).values,
            "frequency": matrix.mean(axis=0).values,
            "log_baseline_hazard": np.diag(log_theta),
            "baseline_hazard": np.exp(np.clip(np.diag(log_theta), -700, 700)),
        }
    )
    event_baseline.to_csv(tables / "event_baseline_hazard.tsv", sep="\t", index=False)

    genotype_summary, genotype_transitions = build_genotype_tables(matrix, log_theta)
    genotype_summary.to_csv(tables / "genotype_summary.tsv", sep="\t", index=False)
    genotype_transitions.to_csv(
        tables / "genotype_transition.tsv", sep="\t", index=False
    )
    genotype_transitions[
        [
            "source_genotype",
            "event_added",
            "log_hazard",
            "hazard",
            "source_sample_count",
        ]
    ].to_csv(tables / "event_hazard.tsv", sep="\t", index=False)
    genotype_transitions[
        [
            "source_genotype",
            "target_genotype",
            "event_added",
            "probability",
            "source_sample_count",
        ]
    ].to_csv(tables / "next_event_probability.tsv", sep="\t", index=False)

    if len(state_table) != len(matrix):
        raise ValueError(f"{dataset}: state table and p15 matrix row counts differ")
    interface_state_table = state_table.copy()
    interface_state_table["genotype_signature"] = [
        genotype_signature(row, events)
        for row in matrix.to_numpy(dtype=np.int32)
    ]
    state_transitions = expand_state_transitions(
        genotype_transitions,
        interface_state_table,
        [str(x) for x in config["transition"]["usable_stages"]],
    )
    state_transitions.to_csv(tables / "transition_prob.tsv", sep="\t", index=False)
    biology = biology_sanity_table(dataset, events, log_theta)
    biology.to_csv(tables / "biological_sanity_checks.tsv", sep="\t", index=False)

    probability_sums = genotype_transitions.groupby("source_genotype")[
        "probability"
    ].sum()
    off_diag = log_theta.copy()
    np.fill_diagonal(off_diag, np.nan)
    metadata = {
        "dataset_name": dataset,
        "display_name": display_name,
        "experiment": "Experiment 3: MHN-to-Rel-ObsTQ interface",
        "mhn_version": mhn_backend.__version__,
        "python_version": platform.python_version(),
        "model_type": "cMHN",
        "penalty": "L1",
        "device": "CPU",
        "random_seed": seed,
        "samples": int(n),
        "events": int(len(events)),
        "observed_genotypes": int(len(genotype_summary)),
        "genotype_transition_edges": int(len(genotype_transitions)),
        "state_transition_edges": int(len(state_transitions)),
        "chosen_lambda": float(chosen_lambda),
        "chosen_lambda_multiplier": float(chosen_lambda * n),
        "cv_folds": int(settings["cv_folds"]),
        "cv_grid_expansions": int(grid_expansions),
        "cv_seconds": float(cv_seconds),
        "fit_seconds": float(fit_seconds),
        "fit_status": int(model.meta["status"]),
        "fit_message": str(model.meta["message"]),
        "fit_iterations": int(model.meta["nit"]),
        "fit_objective": float(model.meta["score"]),
        "theta_finite": bool(np.isfinite(log_theta).all()),
        "max_probability_sum_error": float(
            np.max(np.abs(probability_sums.to_numpy() - 1.0))
            if len(probability_sums)
            else 0.0
        ),
        "median_abs_off_diagonal_log_effect": float(
            np.nanmedian(np.abs(off_diag))
        ),
        "stage_rule": config["transition"]["stage_rule"],
        "stage_transition_boundary": (
            "No explicit stage-transition event is present; Experiment 3 emits "
            "same-stage genotype transitions. Stage progression is delegated to "
            "the stage transition rule in the inflow experiment."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    plot_dataset(
        dataset,
        display_name,
        cv_scores,
        theta,
        genotype_summary,
        genotype_transitions,
        figures / "Figure_E3_MHN_interface",
        config,
    )
    write_dataset_report(dataset_root, metadata, biology)
    logging.info("%s complete: %s", dataset, metadata)
    return metadata


def plot_dataset(
    dataset: str,
    display_name: str,
    cv_scores: pd.DataFrame,
    theta: pd.DataFrame,
    genotype_summary: pd.DataFrame,
    transitions: pd.DataFrame,
    output: Path,
    config: dict,
) -> None:
    color = COHORT_COLORS.get(dataset, "#4477AA")
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.8))
    fig.subplots_adjust(
        left=0.07, right=0.985, bottom=0.09, top=0.90, wspace=0.52, hspace=0.36
    )
    ax = axes[0, 0]
    ax.errorbar(
        cv_scores["lambda_multiplier"],
        cv_scores["mean_test_log_likelihood"],
        yerr=cv_scores["standard_error"],
        marker="o",
        color=color,
        capsize=2,
    )
    selected = cv_scores[cv_scores["selected"]].iloc[0]
    ax.axvline(selected["lambda_multiplier"], color="#333333", ls="--", lw=0.9)
    ax.set_xscale("log")
    ax.set_xlabel(r"Regularization strength ($\lambda \times N$)")
    ax.set_ylabel("Mean held-out log-likelihood")
    ax.set_title("Cross-validation")
    panel_label(ax, "A")

    ax = axes[0, 1]
    interaction = theta.to_numpy().copy()
    np.fill_diagonal(interaction, np.nan)
    limit = max(0.5, float(np.nanquantile(np.abs(interaction), 0.95)))
    sns.heatmap(
        interaction,
        ax=ax,
        cmap="vlag",
        center=0,
        vmin=-limit,
        vmax=limit,
        xticklabels=theta.columns,
        yticklabels=theta.index,
        square=True,
        cbar_kws={"label": "Log hazard effect", "shrink": 0.72},
    )
    ax.set_xlabel("Existing event")
    ax.set_ylabel("Next event")
    ax.set_title("cMHN interaction matrix")
    ax.tick_params(axis="x", rotation=60)
    panel_label(ax, "B")

    ax = axes[1, 0]
    top_n = int(config["plot"]["top_genotypes"])
    top = genotype_summary.head(top_n)["source_genotype"].tolist()
    probability, impossible = transition_probability_matrix(
        transitions, top, theta.columns.tolist()
    )
    labels = [
        f"{g} (n={int(genotype_summary.set_index('source_genotype').loc[g, 'sample_count'])})"
        for g in probability.index
    ]
    sns.heatmap(
        probability,
        ax=ax,
        cmap=PROBABILITY_CMAP,
        norm=PROBABILITY_NORM,
        mask=impossible,
        xticklabels=probability.columns,
        yticklabels=labels,
        cbar_kws={"label": "Next-event probability", "shrink": 0.72},
    )
    ax.set_facecolor("#D9D9D9")
    ax.set_xlabel("Event added")
    ax.set_ylabel("Observed source genotype")
    ax.set_title("MHN-derived one-step transitions\nGray = event already present")
    ax.tick_params(axis="x", rotation=60)
    panel_label(ax, "C")

    ax = axes[1, 1]
    top_edges = supported_edges(
        transitions, min(12, int(config["plot"]["top_edges"]))
    )
    edge_labels = (
        top_edges["source_genotype"].replace({"WT": "WT"})
        + "  + "
        + top_edges["event_added"]
    )
    y = np.arange(len(top_edges))
    ax.barh(y, top_edges["edge_support"], color=color, alpha=0.9)
    ax.set_yticks(y, labels=edge_labels)
    ax.tick_params(axis="y", labelsize=6.2)
    ax.invert_yaxis()
    ax.set_xlabel(r"Cohort-weighted edge support: $f(g)\times P(e\mid g)$")
    ax.set_title("Cohort-supported interface edges")
    ax.set_xlim(0, float(top_edges["edge_support"].max()) * 1.12)
    for idx, (_, edge) in enumerate(top_edges.iterrows()):
        ax.text(
            edge["edge_support"],
            idx,
            f"  p={edge['probability']:.2f}",
            va="center",
            ha="left",
            fontsize=5.8,
            color="#333333",
        )
    panel_label(ax, "D")
    sns.despine(ax=ax)

    fig.suptitle(
        f"{display_name}: cMHN-to-Rel-ObsTQ interface",
        fontweight="bold",
        y=0.975,
    )
    save_figure(fig, output, int(config["plot"]["dpi"]))


def plot_combined(
    datasets: list[str],
    display_names: dict[str, str],
    result_root: Path,
    config: dict,
) -> None:
    fig, axes = plt.subplots(len(datasets), 3, figsize=(13.2, 13.4))
    fig.subplots_adjust(
        left=0.07, right=0.985, bottom=0.055, top=0.855, wspace=0.52, hspace=0.58
    )
    for row, dataset in enumerate(datasets):
        tables = result_root / dataset / "tables"
        cv = pd.read_csv(tables / "cv_likelihood.tsv", sep="\t")
        theta = pd.read_csv(tables / "theta.tsv", sep="\t", index_col=0)
        summary = pd.read_csv(tables / "genotype_summary.tsv", sep="\t")
        transitions = pd.read_csv(tables / "genotype_transition.tsv", sep="\t")
        color = COHORT_COLORS.get(dataset, "#4477AA")

        ax = axes[row, 0]
        ax.errorbar(
            cv["lambda_multiplier"],
            cv["mean_test_log_likelihood"],
            yerr=cv["standard_error"],
            color=color,
            marker="o",
            capsize=2,
        )
        chosen = cv.loc[cv["selected"].astype(str).str.lower().eq("true")].iloc[0]
        ax.axvline(chosen["lambda_multiplier"], color="#333333", ls="--", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel(r"$\lambda \times N$")
        ax.set_ylabel("Held-out log-likelihood")
        ax.set_title(f"{display_names[dataset]} | CV", loc="left", fontweight="bold")

        ax = axes[row, 1]
        top = summary.head(10)["source_genotype"].tolist()
        probability, impossible = transition_probability_matrix(
            transitions, top, theta.columns.tolist()
        )
        sns.heatmap(
            probability,
            ax=ax,
            cmap=PROBABILITY_CMAP,
            norm=PROBABILITY_NORM,
            mask=impossible,
            xticklabels=probability.columns,
            yticklabels=probability.index,
            cbar=False,
        )
        ax.set_facecolor("#D9D9D9")
        ax.set_xlabel("Event added")
        ax.set_ylabel("")
        ax.set_title("One-step probability\nGray = already present")
        ax.tick_params(axis="x", rotation=60, labelsize=5.5)
        ax.tick_params(axis="y", labelsize=5.5)

        ax = axes[row, 2]
        top_edges = supported_edges(
            transitions, min(8, int(config["plot"]["top_edges"]))
        )
        edge_labels = [
            compact_edge_label(source, event)
            for source, event in zip(
                top_edges["source_genotype"], top_edges["event_added"]
            )
        ]
        y = np.arange(len(top_edges))
        ax.barh(y, top_edges["edge_support"], color=color, alpha=0.9)
        ax.set_yticks(y, labels=edge_labels)
        ax.invert_yaxis()
        ax.set_xlabel(r"$f(g)\times P(e\mid g)$")
        ax.set_title("Cohort-supported edges")
        ax.set_xlim(0, float(top_edges["edge_support"].max()) * 1.10)
        ax.tick_params(axis="y", labelsize=5.0)
        sns.despine(ax=ax)

        for column in range(3):
            axes[row, column].text(
                -0.20 if column == 0 else -0.18,
                1.10,
                chr(ord("A") + row * 3 + column),
                transform=axes[row, column].transAxes,
                fontsize=10,
                fontweight="bold",
                va="top",
            )

    fig.suptitle(
        "Experiment 3: independent cMHN-to-Rel-ObsTQ interfaces across three cohorts",
        fontweight="bold",
        y=0.985,
    )
    probability_map = plt.cm.ScalarMappable(
        norm=PROBABILITY_NORM, cmap=PROBABILITY_CMAP
    )
    probability_cax = fig.add_axes([0.47, 0.905, 0.16, 0.009])
    probability_bar = fig.colorbar(
        probability_map, cax=probability_cax, orientation="horizontal"
    )
    probability_bar.ax.set_title("Next-event probability", fontsize=7, pad=3)
    probability_bar.ax.tick_params(labelsize=6, length=2)
    save_figure(
        fig,
        result_root / "combined_figures" / "Figure_E3_MHN_interface_three_cohorts",
        int(config["plot"]["dpi"]),
    )


def plot_single_figures(
    datasets: list[str],
    result_root: Path,
    config: dict,
) -> None:
    """Render atomic E3 panels directly from result tables.

    These are manuscript-reassembly sources, not crops from the multipanel
    overview. They intentionally omit panel letters, figure-level titles and
    dense explanatory text.
    """
    single_dir = result_root / "single_figures"
    single_dir.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        tables = result_root / dataset / "tables"
        cv = pd.read_csv(tables / "cv_likelihood.tsv", sep="\t")
        theta = pd.read_csv(tables / "theta.tsv", sep="\t", index_col=0)
        summary = pd.read_csv(tables / "genotype_summary.tsv", sep="\t")
        transitions = pd.read_csv(tables / "genotype_transition.tsv", sep="\t")
        color = COHORT_COLORS.get(dataset, "#4477AA")

        fig, ax = plt.subplots(figsize=(3.15, 3.05))
        fig.subplots_adjust(left=0.22, right=0.96, bottom=0.18, top=0.94)
        ax.errorbar(
            cv["lambda_multiplier"],
            cv["mean_test_log_likelihood"],
            yerr=cv["standard_error"],
            marker="o",
            color=color,
            capsize=2,
            lw=1.0,
        )
        chosen = cv.loc[cv["selected"].astype(str).str.lower().eq("true")].iloc[0]
        ax.axvline(chosen["lambda_multiplier"], color="#333333", ls="--", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel(r"$\lambda \times N$")
        ax.set_ylabel("Held-out log-likelihood")
        sns.despine(ax=ax)
        figure_style.save_figure(
            fig,
            single_dir / f"Figure_E3_MHN_interface__{dataset}__cv_likelihood",
            config,
        )

        top = summary.head(10)["source_genotype"].tolist()
        probability, impossible = transition_probability_matrix(
            transitions, top, theta.columns.tolist()
        )
        fig, ax = plt.subplots(figsize=(3.35, 3.15))
        fig.subplots_adjust(left=0.27, right=0.90, bottom=0.25, top=0.96)
        sns.heatmap(
            probability,
            ax=ax,
            cmap=PROBABILITY_CMAP,
            norm=PROBABILITY_NORM,
            mask=impossible,
            xticklabels=probability.columns,
            yticklabels=probability.index,
            cbar_kws={"label": r"$P(e\mid g)$", "shrink": 0.64},
        )
        ax.set_facecolor("#D9D9D9")
        ax.set_xlabel("Event added")
        ax.set_ylabel("Observed source genotype")
        ax.tick_params(axis="x", rotation=60, labelsize=5.7)
        ax.tick_params(axis="y", labelsize=5.7)
        figure_style.save_figure(
            fig,
            single_dir / f"Figure_E3_MHN_interface__{dataset}__one_step_probability",
            config,
        )

        top_edges = supported_edges(
            transitions, min(8, int(config["plot"]["top_edges"]))
        )
        edge_labels = [
            compact_edge_label(source, event)
            for source, event in zip(
                top_edges["source_genotype"], top_edges["event_added"]
            )
        ]
        fig, ax = plt.subplots(figsize=(3.35, 3.15))
        fig.subplots_adjust(left=0.43, right=0.96, bottom=0.18, top=0.95)
        y = np.arange(len(top_edges))
        ax.barh(
            y,
            top_edges["edge_support"],
            color=color,
            edgecolor="#263238",
            linewidth=0.35,
            alpha=0.92,
        )
        ax.set_yticks(y, labels=edge_labels)
        ax.invert_yaxis()
        ax.set_xlabel(r"$f(g)\times P(e\mid g)$")
        ax.tick_params(axis="y", labelsize=5.8)
        ax.set_xlim(0, float(top_edges["edge_support"].max()) * 1.12)
        ax.grid(axis="x", color="#E6E6E6", lw=0.35)
        sns.despine(ax=ax)
        figure_style.save_figure(
            fig,
            single_dir / f"Figure_E3_MHN_interface__{dataset}__supported_edges",
            config,
        )


def write_dataset_report(
    dataset_root: Path, metadata: dict, biology: pd.DataFrame
) -> None:
    lines = [
        f"# Experiment 3: {metadata['display_name']}",
        "",
        "## Model",
        "",
        f"- Independent cMHN fit: yes",
        f"- Samples / events: {metadata['samples']} / {metadata['events']}",
        f"- Chosen lambda: {metadata['chosen_lambda']:.8g} "
        f"(lambda x N = {metadata['chosen_lambda_multiplier']:.4g})",
        f"- CV folds: {metadata['cv_folds']}",
        f"- Fit status: {metadata['fit_status']} ({metadata['fit_message']})",
        f"- Observed genotypes: {metadata['observed_genotypes']}",
        f"- Genotype transition edges: {metadata['genotype_transition_edges']}",
        f"- State transition edges: {metadata['state_transition_edges']}",
        "",
        "## Interface Boundary",
        "",
        metadata["stage_transition_boundary"],
        "",
        "## Biological Review",
        "",
        "The checks below are descriptive plausibility reviews. They do not impose "
        "sign constraints and do not convert cross-sectional associations into causal claims.",
        "",
    ]
    for _, row in biology.iterrows():
        lines.append(
            f"- {row['event_a']} / {row['event_b']}: "
            f"{row['qualitative_pattern']} ({row['biological_context']})."
        )
    (dataset_root / "experiment_03_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_summary(result_root: Path, records: list[dict]) -> None:
    summary_columns = [
        "dataset_name",
        "display_name",
        "samples",
        "events",
        "observed_genotypes",
        "genotype_transition_edges",
        "state_transition_edges",
        "chosen_lambda",
        "chosen_lambda_multiplier",
        "fit_status",
        "fit_iterations",
        "cv_seconds",
        "fit_seconds",
        "max_probability_sum_error",
    ]
    summary = pd.DataFrame(records)[summary_columns]
    summary.to_csv(result_root / "experiment_03_summary.csv", index=False)
    lines = [
        "# Experiment 3 Summary",
        "",
        "Selected cancer cohorts were fitted independently with cMHN. No parameters "
        "or transition probabilities were shared across cancer types.",
        "",
        "| " + " | ".join(summary.columns) + " |",
        "| " + " | ".join(["---"] * len(summary.columns)) + " |",
    ]
    for _, row in summary.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in summary.columns) + " |")
    lines.extend(
        [
            "",
            "The main interface emits same-stage, one-event genotype transitions. "
            "Explicit stage progression remains the responsibility of the stage "
            "transition rule in Experiment 4.",
        ]
    )
    (result_root / "experiment_03_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    selected = yaml.safe_load(Path(args.dataset_config).read_text(encoding="utf-8"))
    experiment_12_config = yaml.safe_load(
        Path("configs/experiments_01_02.yaml").read_text(encoding="utf-8")
    )
    available = [
        entry["dataset_name"] for entry in selected["included_datasets"]
    ]
    datasets = args.datasets or available
    unknown = sorted(set(datasets) - set(available))
    if unknown:
        raise ValueError(f"Datasets are not selected cohorts: {unknown}")
    display_names = {
        name: experiment_12_config["datasets"][name]["display_name"]
        for name in available
    }

    result_root = Path(config["result_root"]).resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    setup_logging(result_root)
    configure_plotting(config)
    records = []
    for dataset in datasets:
        print(f"[Experiment 3] Training {dataset}...", flush=True)
        record = train_dataset(
            dataset, display_names[dataset], config, result_root, args.force
        )
        records.append(record)
        print(
            f"[Experiment 3] {dataset} complete: "
            f"lambda={record['chosen_lambda']:.8g}, status={record['fit_status']}",
            flush=True,
        )

    if set(datasets) == set(available):
        all_records = [
            json.loads(
                (result_root / dataset / "model_metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            for dataset in available
        ]
        write_summary(result_root, all_records)
        plot_combined(available, display_names, result_root, config)
        plot_single_figures(available, result_root, config)


if __name__ == "__main__":
    main()

"""Run Experiment 4: estimate relative state inflow F_hat.

The primary analysis uses same-stage, one-event genotype predecessors. Stage
bridges, two-event predecessors, and occupancy smoothing are sensitivity rules
and are never mixed into the primary result.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
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
from scipy.stats import spearmanr


COHORT_COLORS = {
    "AACR_LUAD": "#4477AA",
    "AACR_COAD": "#CC6677",
    "AACR_IDC": "#228833",
}
RULE_COLORS = {
    "rule_b_stage_bridge": "#CC6677",
    "rule_a_two_step": "#228833",
    "rule_a_smoothed": "#AA3377",
}
RULE_LABELS = {
    "rule_b_stage_bridge": "Stage bridge",
    "rule_a_two_step": "Two-step",
    "rule_a_smoothed": "Smoothed L",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Rel-ObsTQ-MHN Experiment 4.")
    parser.add_argument("--config", default="configs/experiment_04.yaml")
    parser.add_argument(
        "--dataset-config", default="configs/selected_experiment_datasets.yaml"
    )
    return parser.parse_args()


def configure_plotting(config: dict) -> None:
    figure_style.configure_matplotlib(config)


def setup_logging(root: Path) -> None:
    (root / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=root / "logs" / "experiment_04.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def save_figure(fig: plt.Figure, base_path: Path, dpi: int) -> None:
    figure_style.save_figure_panels(fig, base_path, {"plot": {"dpi": dpi}}, dpi=dpi)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
    )


def genotype_signature(vector: np.ndarray, events: list[str]) -> str:
    present = [events[i] for i, value in enumerate(vector) if int(value) == 1]
    return "+".join(present) if present else "WT"


def genotype_vector(signature: str, events: list[str]) -> np.ndarray:
    present = set() if signature == "WT" else set(signature.split("+"))
    return np.array([int(event in present) for event in events], dtype=np.int32)


def compact_state(state: str, max_events: int = 3) -> str:
    stage, genotype = state.split("::", 1)
    if genotype == "WT":
        return f"{stage}: WT"
    events = genotype.split("+")
    if len(events) > max_events:
        genotype = "+".join(events[:max_events]) + "+..."
    return f"{stage}: {genotype}"


def clean_stage(
    dataset: str, state_table: pd.DataFrame, dataset_config: dict
) -> tuple[pd.Series, pd.Series]:
    stage = state_table["stage_group"].fillna("unknown").astype(str).str.lower()
    status = (
        state_table["metastasis_status"].fillna("").astype(str).str.lower()
    )
    raw = state_table["stage_raw"].fillna("").astype(str).str.upper()
    reason = pd.Series("included", index=state_table.index, dtype=object)

    if dataset.startswith("AACR_"):
        clean = stage.where(stage.isin(["primary", "metastatic"]), "excluded")
        reason[clean.eq("excluded")] = "unknown_or_nonordered_stage"
        return clean, reason

    excluded_keywords = [
        str(x).lower()
        for x in dataset_config.get("exclude_specimen_keywords", [])
    ]
    model_system = pd.Series(False, index=state_table.index)
    for keyword in excluded_keywords:
        model_system |= status.str.contains(keyword, regex=False)

    metastatic = (
        status.str.contains("metastatic", regex=False)
        | raw.str.match(r"^IV[A-C]?$", na=False)
        | raw.str.contains(r"M1(?:\D|$)", regex=True)
    )
    primary = status.str.contains("primary tumour", regex=False)
    clean = pd.Series("excluded", index=state_table.index, dtype=object)
    clean[primary] = "primary"
    clean[metastatic] = "metastatic"
    clean[model_system] = "excluded"
    reason[model_system] = "non_patient_model_or_recurrent_specimen"
    reason[clean.eq("excluded") & ~model_system] = "unresolved_disease_compartment"
    return clean, reason


def prepare_states(
    dataset: str, config: dict
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    source = (
        Path(config["experiments_01_02_root"])
        / dataset
        / "experiment_01_data_preparation"
        / "tables"
    )
    panel_size = int(config["panel_size"])
    matrix = pd.read_csv(source / f"mhn_training_matrix_p{panel_size}.csv")
    panel = pd.read_csv(source / f"event_panel_p{panel_size}.csv")
    state_table = pd.read_csv(source / f"state_table_p{panel_size}.csv")
    events = panel["event"].astype(str).tolist()
    if matrix.columns.astype(str).tolist() != events:
        raise ValueError(f"{dataset}: p15 event order mismatch")
    if len(matrix) != len(state_table):
        raise ValueError(f"{dataset}: matrix/state row mismatch")

    stage, reason = clean_stage(dataset, state_table, config["datasets"][dataset])
    work = state_table[
        ["analysis_id", "patient_id", "sample_id", "stage_raw", "metastasis_status"]
    ].copy()
    work["stage"] = stage
    work["inclusion_reason"] = reason
    work["genotype"] = [
        genotype_signature(row, events)
        for row in matrix.to_numpy(dtype=np.int32)
    ]
    work["event_count"] = matrix.sum(axis=1).astype(int).values
    work["included"] = work["stage"].ne("excluded")
    included = work[work["included"]].copy()
    included["state"] = included["stage"] + "::" + included["genotype"]

    occupancy = (
        included.groupby(["state", "stage", "genotype", "event_count"], dropna=False)
        .size()
        .rename("N_v")
        .reset_index()
    )
    n = len(included)
    occupancy["L_v"] = occupancy["N_v"] / max(n, 1)
    occupancy = occupancy.sort_values(
        ["N_v", "state"], ascending=[False, True]
    ).reset_index(drop=True)

    qc = (
        work.groupby(["included", "stage", "inclusion_reason"], dropna=False)
        .size()
        .rename("samples")
        .reset_index()
    )
    return occupancy, events, qc


def transition_lookup(transitions: pd.DataFrame) -> dict[tuple[str, str], float]:
    return {
        (str(row.source_genotype), str(row.event_added)): float(row.probability)
        for row in transitions.itertuples(index=False)
    }


def theta_next_probabilities(
    signature: str, events: list[str], log_theta: np.ndarray
) -> dict[str, float]:
    state = genotype_vector(signature, events)
    absent = np.flatnonzero(state == 0)
    if len(absent) == 0:
        return {}
    log_hazards = np.array(
        [
            log_theta[idx, idx] + log_theta[idx, state.astype(bool)].sum()
            for idx in absent
        ]
    )
    scaled = np.exp(log_hazards - log_hazards.max())
    probabilities = scaled / scaled.sum()
    return {events[idx]: float(probabilities[pos]) for pos, idx in enumerate(absent)}


def build_one_step_edges(
    occupancy: pd.DataFrame,
    events: list[str],
    lookup: dict[tuple[str, str], float],
    rule: str,
    source_l: dict[str, float],
    probability_scale: float = 1.0,
) -> pd.DataFrame:
    observed_states = set(occupancy["state"])
    rows = []
    for target in occupancy.itertuples(index=False):
        vector = genotype_vector(target.genotype, events)
        for event_idx in np.flatnonzero(vector == 1):
            source_vector = vector.copy()
            source_vector[event_idx] = 0
            source_genotype = genotype_signature(source_vector, events)
            source_state = f"{target.stage}::{source_genotype}"
            if source_state not in observed_states:
                continue
            probability = lookup.get((source_genotype, events[event_idx]), 0.0)
            if probability <= 0:
                continue
            edge_probability = probability_scale * probability
            rows.append(
                {
                    "rule": rule,
                    "source_state": source_state,
                    "target_state": target.state,
                    "predecessor_type": "same_stage_one_event",
                    "event_added": events[event_idx],
                    "step_distance": 1,
                    "edge_probability": edge_probability,
                    "source_L": source_l[source_state],
                    "inflow_contribution": source_l[source_state]
                    * edge_probability,
                }
            )
    return pd.DataFrame(rows)


def build_stage_bridge_edges(
    occupancy: pd.DataFrame,
    events: list[str],
    lookup: dict[tuple[str, str], float],
    source_l: dict[str, float],
    stage_order: list[str],
    stage_mass: float,
    rule: str,
) -> pd.DataFrame:
    observed_states = set(occupancy["state"])
    previous = {
        stage_order[idx]: stage_order[idx - 1]
        for idx in range(1, len(stage_order))
    }
    rows = []
    for target in occupancy.itertuples(index=False):
        if target.stage not in previous:
            continue
        prev_stage = previous[target.stage]
        same_genotype_source = f"{prev_stage}::{target.genotype}"
        if same_genotype_source in observed_states:
            edge_probability = stage_mass / 2
            rows.append(
                {
                    "rule": rule,
                    "source_state": same_genotype_source,
                    "target_state": target.state,
                    "predecessor_type": "previous_stage_same_genotype",
                    "event_added": "STAGE_ADVANCE",
                    "step_distance": 1,
                    "edge_probability": edge_probability,
                    "source_L": source_l[same_genotype_source],
                    "inflow_contribution": source_l[same_genotype_source]
                    * edge_probability,
                }
            )
        vector = genotype_vector(target.genotype, events)
        for event_idx in np.flatnonzero(vector == 1):
            source_vector = vector.copy()
            source_vector[event_idx] = 0
            source_genotype = genotype_signature(source_vector, events)
            source_state = f"{prev_stage}::{source_genotype}"
            if source_state not in observed_states:
                continue
            probability = lookup.get((source_genotype, events[event_idx]), 0.0)
            if probability <= 0:
                continue
            edge_probability = (stage_mass / 2) * probability
            rows.append(
                {
                    "rule": rule,
                    "source_state": source_state,
                    "target_state": target.state,
                    "predecessor_type": "previous_stage_plus_event",
                    "event_added": events[event_idx],
                    "step_distance": 1,
                    "edge_probability": edge_probability,
                    "source_L": source_l[source_state],
                    "inflow_contribution": source_l[source_state]
                    * edge_probability,
                }
            )
    return pd.DataFrame(rows)


def build_two_step_edges(
    occupancy: pd.DataFrame,
    events: list[str],
    log_theta: np.ndarray,
    source_l: dict[str, float],
    two_step_mass: float,
    rule: str,
) -> pd.DataFrame:
    observed_states = set(occupancy["state"])
    rows = []
    probability_cache: dict[str, dict[str, float]] = {}

    def probabilities(signature: str) -> dict[str, float]:
        if signature not in probability_cache:
            probability_cache[signature] = theta_next_probabilities(
                signature, events, log_theta
            )
        return probability_cache[signature]

    for target in occupancy.itertuples(index=False):
        target_vector = genotype_vector(target.genotype, events)
        present = np.flatnonzero(target_vector == 1)
        for left_pos in range(len(present)):
            for right_pos in range(left_pos + 1, len(present)):
                first_idx = present[left_pos]
                second_idx = present[right_pos]
                source_vector = target_vector.copy()
                source_vector[[first_idx, second_idx]] = 0
                source_genotype = genotype_signature(source_vector, events)
                source_state = f"{target.stage}::{source_genotype}"
                if source_state not in observed_states:
                    continue
                first_event, second_event = events[first_idx], events[second_idx]
                p0 = probabilities(source_genotype)

                intermediate_first = source_vector.copy()
                intermediate_first[first_idx] = 1
                signature_first = genotype_signature(intermediate_first, events)
                intermediate_second = source_vector.copy()
                intermediate_second[second_idx] = 1
                signature_second = genotype_signature(intermediate_second, events)
                path_probability = (
                    p0.get(first_event, 0)
                    * probabilities(signature_first).get(second_event, 0)
                    + p0.get(second_event, 0)
                    * probabilities(signature_second).get(first_event, 0)
                )
                if path_probability <= 0:
                    continue
                edge_probability = two_step_mass * path_probability
                rows.append(
                    {
                        "rule": rule,
                        "source_state": source_state,
                        "target_state": target.state,
                        "predecessor_type": "same_stage_two_event",
                        "event_added": f"{first_event}+{second_event}",
                        "step_distance": 2,
                        "edge_probability": edge_probability,
                        "source_L": source_l[source_state],
                        "inflow_contribution": source_l[source_state]
                        * edge_probability,
                    }
                )
    return pd.DataFrame(rows)


def aggregate_inflow(
    occupancy: pd.DataFrame,
    edges: pd.DataFrame,
    rule: str,
    min_count: int,
    min_inflow: float,
) -> pd.DataFrame:
    result = occupancy.copy()
    if edges.empty:
        result["F_hat"] = 0.0
        result["n_predecessors"] = 0
        result["dominant_predecessor"] = ""
        result["dominant_edge_probability"] = 0.0
        result["dominant_contribution"] = 0.0
        result["genotype_inflow"] = 0.0
        result["stage_inflow"] = 0.0
    else:
        totals = (
            edges.groupby("target_state")["inflow_contribution"]
            .sum()
            .rename("F_hat")
        )
        counts = (
            edges.groupby("target_state").size().rename("n_predecessors")
        )
        dominant = (
            edges.sort_values("inflow_contribution", ascending=False)
            .drop_duplicates("target_state")
            .set_index("target_state")
        )
        genotype = (
            edges[
                edges["predecessor_type"].str.startswith("same_stage")
            ]
            .groupby("target_state")["inflow_contribution"]
            .sum()
            .rename("genotype_inflow")
        )
        stage = (
            edges[
                edges["predecessor_type"].str.startswith("previous_stage")
            ]
            .groupby("target_state")["inflow_contribution"]
            .sum()
            .rename("stage_inflow")
        )
        result = result.join(totals, on="state").join(counts, on="state")
        result = result.join(genotype, on="state").join(stage, on="state")
        result["dominant_predecessor"] = result["state"].map(
            dominant["source_state"]
        )
        result["dominant_edge_probability"] = result["state"].map(
            dominant["edge_probability"]
        )
        result["dominant_contribution"] = result["state"].map(
            dominant["inflow_contribution"]
        )
        for column in [
            "F_hat",
            "n_predecessors",
            "genotype_inflow",
            "stage_inflow",
            "dominant_edge_probability",
            "dominant_contribution",
        ]:
            result[column] = result[column].fillna(0)
        result["dominant_predecessor"] = result[
            "dominant_predecessor"
        ].fillna("")

    result["rule"] = rule
    result["count_eligible"] = result["N_v"] >= min_count
    result["inflow_eligible"] = result["F_hat"] >= min_inflow
    result["stable_for_experiment5"] = (
        result["count_eligible"] & result["inflow_eligible"]
    )
    result["flags"] = np.select(
        [
            ~result["count_eligible"],
            result["count_eligible"] & ~result["inflow_eligible"],
        ],
        ["rare_state", "low_or_zero_inflow"],
        default="stable",
    )
    return result.sort_values(["F_hat", "N_v"], ascending=[False, False])


def make_rule_outputs(
    dataset: str,
    config: dict,
    occupancy: pd.DataFrame,
    events: list[str],
    transitions: pd.DataFrame,
    theta: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    min_count = int(config["thresholds"]["minimum_state_count"])
    min_inflow = float(config["thresholds"]["minimum_inflow"])
    lookup = transition_lookup(transitions)
    source_l_main = occupancy.set_index("state")["L_v"].to_dict()

    main_name = config["main_rule"]["name"]
    main_edges = build_one_step_edges(
        occupancy, events, lookup, main_name, source_l_main
    )
    main = aggregate_inflow(
        occupancy, main_edges, main_name, min_count, min_inflow
    )

    stage_cfg = config["sensitivity_rules"]["stage_bridge"]
    stage_mass = float(stage_cfg["stage_mass"])
    stage_name = stage_cfg["name"]
    stage_edges = build_one_step_edges(
        occupancy,
        events,
        lookup,
        stage_name,
        source_l_main,
        probability_scale=1 - stage_mass,
    )
    bridge = build_stage_bridge_edges(
        occupancy,
        events,
        lookup,
        source_l_main,
        config["datasets"][dataset]["stage_order"],
        stage_mass,
        stage_name,
    )
    stage_edges = pd.concat([stage_edges, bridge], ignore_index=True)
    stage_result = aggregate_inflow(
        occupancy, stage_edges, stage_name, min_count, min_inflow
    )

    two_cfg = config["sensitivity_rules"]["two_step"]
    two_mass = float(two_cfg["two_step_mass"])
    two_name = two_cfg["name"]
    two_edges = build_one_step_edges(
        occupancy,
        events,
        lookup,
        two_name,
        source_l_main,
        probability_scale=1 - two_mass,
    )
    two_extra = build_two_step_edges(
        occupancy,
        events,
        theta.to_numpy(dtype=float),
        source_l_main,
        two_mass,
        two_name,
    )
    two_edges = pd.concat([two_edges, two_extra], ignore_index=True)
    two_result = aggregate_inflow(
        occupancy, two_edges, two_name, min_count, min_inflow
    )

    smooth_cfg = config["sensitivity_rules"]["smoothed"]
    smooth_name = smooth_cfg["name"]
    alpha = float(smooth_cfg["occupancy_alpha"])
    k = len(occupancy)
    n = int(occupancy["N_v"].sum())
    smooth_l = (
        (occupancy.set_index("state")["N_v"] + alpha) / (n + alpha * k)
    ).to_dict()
    smooth_edges = build_one_step_edges(
        occupancy, events, lookup, smooth_name, smooth_l
    )
    smooth_occupancy = occupancy.copy()
    smooth_occupancy["L_v"] = smooth_occupancy["state"].map(smooth_l)
    smooth_result = aggregate_inflow(
        smooth_occupancy, smooth_edges, smooth_name, min_count, min_inflow
    )

    return (
        {
            main_name: main,
            stage_name: stage_result,
            two_name: two_result,
            smooth_name: smooth_result,
        },
        {
            main_name: main_edges,
            stage_name: stage_edges,
            two_name: two_edges,
            smooth_name: smooth_edges,
        },
    )


def sensitivity_metrics(
    rule_tables: dict[str, pd.DataFrame], config: dict
) -> pd.DataFrame:
    main_name = config["main_rule"]["name"]
    main = rule_tables[main_name].set_index("state")
    top_k = int(config["thresholds"]["top_k"])
    rows = []
    for rule, table in rule_tables.items():
        if rule == main_name:
            continue
        other = table.set_index("state")
        joined = main[["F_hat", "N_v"]].join(
            other[["F_hat"]], how="inner", rsuffix="_other"
        )
        eligible = joined[
            (joined["N_v"] >= int(config["thresholds"]["minimum_state_count"]))
            & (joined["F_hat"] > 0)
            & (joined["F_hat_other"] > 0)
        ]
        rho = (
            float(spearmanr(eligible["F_hat"], eligible["F_hat_other"]).statistic)
            if len(eligible) >= 3
            else np.nan
        )
        main_top = set(
            joined[joined["F_hat"] > 0].nlargest(top_k, "F_hat").index
        )
        other_top = set(
            joined[joined["F_hat_other"] > 0]
            .nlargest(top_k, "F_hat_other")
            .index
        )
        overlap = len(main_top & other_top) / max(top_k, 1)
        relative_change = (
            np.abs(
                eligible["F_hat_other"] - eligible["F_hat"]
            )
            / eligible["F_hat"].clip(lower=1e-12)
            if len(eligible)
            else pd.Series(dtype=float)
        )
        rows.append(
            {
                "comparison_rule": rule,
                "states_compared": len(eligible),
                "spearman_F_hat": rho,
                "top_k": top_k,
                "top_k_overlap": overlap,
                "median_absolute_relative_change": (
                    float(relative_change.median())
                    if len(relative_change)
                    else np.nan
                ),
                "positive_inflow_states": int((other["F_hat"] > 0).sum()),
                "stable_states": int(other["stable_for_experiment5"].sum()),
            }
        )
    return pd.DataFrame(rows)


def summary_metrics(
    dataset: str, rule_tables: dict[str, pd.DataFrame], config: dict
) -> pd.DataFrame:
    rows = []
    for rule, table in rule_tables.items():
        stable = table[table["stable_for_experiment5"]]
        positive = table[table["F_hat"] > 0]
        correlation_states = table[(table["F_hat"] > 0) & (table["N_v"] >= 5)]
        rho = (
            float(spearmanr(correlation_states["L_v"], correlation_states["F_hat"]).statistic)
            if len(correlation_states) >= 3
            else np.nan
        )
        rows.append(
            {
                "dataset_name": dataset,
                "rule": rule,
                "analysis_samples": int(table["N_v"].sum()),
                "observed_states": len(table),
                "positive_inflow_states": len(positive),
                "zero_inflow_states": int((table["F_hat"] <= 0).sum()),
                "stable_states": len(stable),
                "stable_sample_fraction": float(stable["L_v"].sum()),
                "median_positive_F_hat": (
                    float(positive["F_hat"].median()) if len(positive) else 0.0
                ),
                "max_F_hat": float(table["F_hat"].max()),
                "spearman_L_vs_F": rho,
            }
        )
    return pd.DataFrame(rows)


def top_edge_table(edges: pd.DataFrame, count: int) -> pd.DataFrame:
    if edges.empty:
        return edges
    return edges.nlargest(count, "inflow_contribution").copy()


def plot_dataset(
    dataset: str,
    config: dict,
    rule_tables: dict[str, pd.DataFrame],
    edge_tables: dict[str, pd.DataFrame],
    sensitivity: pd.DataFrame,
    output: Path,
) -> None:
    main_name = config["main_rule"]["name"]
    main = rule_tables[main_name]
    stage_name = config["sensitivity_rules"]["stage_bridge"]["name"]
    stage_result = rule_tables[stage_name].set_index("state")
    stage_colors = config["datasets"][dataset]["stage_colors"]
    color = COHORT_COLORS[dataset]

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.2))
    fig.subplots_adjust(
        left=0.08, right=0.98, bottom=0.09, top=0.90, wspace=0.36, hspace=0.38
    )

    ax = axes[0, 0]
    positive = main[(main["L_v"] > 0) & (main["F_hat"] > 0)]
    for stage, group in positive.groupby("stage"):
        ax.scatter(
            group["F_hat"],
            group["L_v"],
            s=np.clip(10 + 5 * np.sqrt(group["N_v"]), 14, 75),
            c=stage_colors.get(stage, "#999999"),
            alpha=0.72,
            edgecolor=np.where(group["stable_for_experiment5"], "#222222", "none"),
            linewidth=0.45,
            label=stage,
        )
    limits = [
        min(positive["F_hat"].min(), positive["L_v"].min()) * 0.75,
        max(positive["F_hat"].max(), positive["L_v"].max()) * 1.25,
    ]
    ax.plot(limits, limits, color="#777777", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel(r"Inferred relative inflow $\hat{F}_v$")
    ax.set_ylabel(r"Observed occupancy $L_v$")
    ax.set_title("Occupancy versus inferred inflow")
    ax.legend(frameon=False, title="Stage", loc="lower right")
    panel_label(ax, "A")
    sns.despine(ax=ax)

    ax = axes[0, 1]
    top_edges = top_edge_table(
        edge_tables[main_name], int(config["plot"]["top_edges"])
    ).sort_values("inflow_contribution")
    labels = [
        f"{compact_state(source, 2)} -> {compact_state(target, 2).split(': ', 1)[1]}"
        for source, target in zip(
            top_edges["source_state"], top_edges["target_state"]
        )
    ]
    ax.barh(
        np.arange(len(top_edges)),
        top_edges["inflow_contribution"],
        color=color,
        alpha=0.9,
    )
    ax.set_yticks(np.arange(len(top_edges)), labels=labels)
    ax.set_xlabel(r"Edge contribution $L_uP(u\to v)$")
    ax.set_title("Dominant contributors to relative inflow")
    panel_label(ax, "B")
    sns.despine(ax=ax)

    ax = axes[1, 0]
    top_states = main.nlargest(int(config["plot"]["top_states"]), "F_hat")
    comparison = top_states[["state", "F_hat"]].copy()
    comparison["stage_bridge"] = comparison["state"].map(stage_result["F_hat"])
    comparison["stage_component"] = comparison["state"].map(
        stage_result["stage_inflow"]
    )
    comparison = comparison.sort_values("F_hat")
    y = np.arange(len(comparison))
    ax.barh(y, comparison["F_hat"], color="#B7C9E2", label="Rule A")
    ax.scatter(
        comparison["stage_bridge"],
        y,
        color="#CC6677",
        s=20,
        zorder=3,
        label="Rule B total",
    )
    ax.scatter(
        comparison["stage_component"],
        y,
        facecolor="white",
        edgecolor="#AA3377",
        s=20,
        zorder=3,
        label="Stage component",
    )
    ax.set_yticks(y, labels=[compact_state(x, 3) for x in comparison["state"]])
    ax.set_xlabel(r"Relative inflow $\hat{F}_v$")
    ax.set_title("Main inflow and stage-bridge sensitivity")
    ax.legend(frameon=False, loc="lower right")
    panel_label(ax, "C")
    sns.despine(ax=ax)

    ax = axes[1, 1]
    metric_order = [
        "spearman_F_hat",
        "top_k_overlap",
    ]
    labels = {
        "spearman_F_hat": "Rank correlation",
        "top_k_overlap": "Top-10 overlap",
    }
    x = np.arange(len(sensitivity))
    width = 0.34
    for idx, metric in enumerate(metric_order):
        ax.bar(
            x + (idx - 0.5) * width,
            sensitivity[metric],
            width=width,
            label=labels[metric],
            color=["#4477AA", "#EEAA33"][idx],
        )
    ax.set_xticks(
        x,
        labels=[RULE_LABELS.get(rule, rule) for rule in sensitivity["comparison_rule"]],
    )
    ax.set_ylim(0, 1.05)
    ax.axhline(0.6, color="#777777", ls="--", lw=0.8)
    ax.set_ylabel("Agreement with Rule A")
    ax.set_title("Inflow-rule sensitivity")
    ax.legend(frameon=False, loc="lower right")
    panel_label(ax, "D")
    sns.despine(ax=ax)

    fig.suptitle(
        f"{config['datasets'][dataset]['display_name']}: relative inflow estimation",
        fontweight="bold",
        y=0.975,
    )
    save_figure(fig, output, int(config["plot"]["dpi"]))


def plot_combined(
    datasets: list[str], config: dict, result_root: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 7.4))
    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        bottom=0.09,
        top=0.84,
        wspace=0.25,
        hspace=0.34,
    )
    for panel_index, dataset in enumerate(datasets):
        ax = axes.ravel()[panel_index]
        tables = result_root / dataset / "tables"
        main = pd.read_csv(tables / "inflow_table_rule_a_one_step.tsv", sep="\t")
        stage_colors = config["datasets"][dataset]["stage_colors"]

        positive = main[(main["L_v"] > 0) & (main["F_hat"] > 0)]
        for stage, group in positive.groupby("stage"):
            ax.scatter(
                group["F_hat"],
                group["L_v"],
                s=np.clip(10 + 5 * np.sqrt(group["N_v"]), 14, 75),
                c=stage_colors.get(stage, "#999999"),
                alpha=0.72,
                edgecolor=np.where(
                    group["stable_for_experiment5"],
                    "#222222",
                    "none",
                ),
                linewidth=0.45,
                label=stage,
            )
        limits = [
            min(positive["F_hat"].min(), positive["L_v"].min()) * 0.65,
            max(positive["F_hat"].max(), positive["L_v"].max()) * 1.55,
        ]
        ax.plot(limits, limits, color="#777777", ls="--", lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_xlabel(r"Inferred relative inflow $\hat{F}_v$")
        ax.set_ylabel(r"Observed occupancy $L_v$")
        ax.set_title(
            config["datasets"][dataset]["display_name"],
            loc="left",
            fontweight="bold",
        )
        ax.legend(frameon=False, title="Stage", loc="lower right")
        sns.despine(ax=ax)
        ax.text(
            -0.15,
            1.07,
            chr(ord("A") + panel_index),
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
        )

    fig.suptitle(
        "Experiment 4 | Occupancy and inferred relative inflow",
        fontweight="bold",
        y=0.975,
    )
    save_figure(
        fig,
        result_root / "combined_figures" / "Figure_E4_relative_inflow_three_cohorts",
        int(config["plot"]["dpi"]),
    )


def plot_combined_rule_sensitivity(
    datasets: list[str], config: dict, result_root: Path
) -> None:
    rules = [
        config["sensitivity_rules"]["stage_bridge"]["name"],
        config["sensitivity_rules"]["two_step"]["name"],
        config["sensitivity_rules"]["smoothed"]["name"],
    ]
    rule_labels = [RULE_LABELS[rule] for rule in rules]
    cohort_labels = {
        "AACR_LUAD": "LUAD",
        "AACR_COAD": "COAD",
        "AACR_IDC": "IDC",
    }
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        frame = pd.read_csv(
            result_root / dataset / "tables" / "inflow_rule_sensitivity.tsv",
            sep="\t",
        ).set_index("comparison_rule")
        for rule, rule_label in zip(rules, rule_labels):
            record = frame.loc[rule]
            top_k = int(record["top_k"])
            retained = int(round(top_k * float(record["top_k_overlap"])))
            rows.append(
                {
                    "dataset": dataset,
                    "cohort": cohort_labels.get(dataset, dataset),
                    "rule": rule,
                    "rule_label": rule_label,
                    "states": int(record["states_compared"]),
                    "rank": float(record["spearman_F_hat"]),
                    "retained": retained,
                    "top_k": top_k,
                    "change": float(record["median_absolute_relative_change"]),
                }
            )

    categorical = figure_style.categorical_palette(config)
    palette = {
        "lavender": categorical.get("lavender", "#B5AED5"),
        "sky_blue": categorical.get("sky_blue", "#B2E6FD"),
        "sage": categorical.get("sage", "#B8D2CC"),
        "coral": categorical.get("coral", "#E8B2A7"),
        "pale_yellow": categorical.get("pale_yellow", "#FEEBB9"),
    }
    cohort_colors = {
        "AACR_LUAD": palette["lavender"],
        "AACR_COAD": palette["sky_blue"],
        "AACR_IDC": palette["sage"],
    }
    rule_colors = {
        rules[0]: palette["coral"],
        rules[1]: palette["sky_blue"],
        rules[2]: palette["sage"],
    }

    def mix(color_a: str, color_b: str, fraction: float) -> str:
        fraction = float(np.clip(fraction, 0.0, 1.0))
        a = np.asarray(mcolors.to_rgb(color_a))
        b = np.asarray(mcolors.to_rgb(color_b))
        return mcolors.to_hex(a * (1.0 - fraction) + b * fraction)

    def pale(color: str, strength: float) -> str:
        return mix("#FFFFFF", color, strength)

    def rank_fill(value: float) -> str:
        quality = np.clip((value - 0.90) / 0.10, 0.0, 1.0)
        return pale(palette["sage"], 0.18 + 0.45 * quality)

    def retained_fill(value: int, top_k: int) -> str:
        quality = np.clip(value / max(top_k, 1), 0.0, 1.0)
        return pale(palette["sky_blue"], 0.16 + 0.44 * quality)

    def change_fill(value: float) -> str:
        fraction = np.clip(value / 0.25, 0.0, 1.0)
        return mix(pale(palette["sage"], 0.55), pale(palette["coral"], 0.62), fraction)

    columns = [
        ("Cohort", 0.88),
        ("Sensitivity rule", 1.35),
        ("n states", 0.68),
        (r"Rank $\rho$", 0.80),
        ("Top-10", 0.74),
        (r"Median $|\Delta\hat{F}|/\hat{F}$", 1.18),
    ]
    x_edges = np.concatenate([[0.0], np.cumsum([width for _, width in columns])])
    total_width = float(x_edges[-1])
    header_h = 0.52
    row_h = 0.43
    top_margin = 0.86
    bottom_margin = 0.34
    table_top = bottom_margin + len(rows) * row_h + header_h
    total_height = table_top + top_margin

    fig, ax = plt.subplots(figsize=(7.05, 6.55))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.97)
    ax.set_xlim(-0.04, total_width + 0.04)
    ax.set_ylim(0.0, total_height)
    ax.axis("off")

    ax.text(
        0.0,
        total_height - 0.12,
        "Cross-cohort sensitivity of relative inflow",
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="#222222",
    )
    ax.text(
        0.0,
        total_height - 0.43,
        r"Exact audit table; stable behavior means high $\rho$, high Top-10 retention and low median relative change.",
        ha="left",
        va="top",
        fontsize=6.8,
        color="#4B4B4B",
    )

    header_bottom = table_top - header_h
    ax.add_patch(
        plt.Rectangle(
            (0, header_bottom),
            total_width,
            header_h,
            facecolor="#ECEFF1",
            edgecolor="none",
            zorder=0,
        )
    )
    for column_index, (label, _) in enumerate(columns):
        x_mid = (x_edges[column_index] + x_edges[column_index + 1]) / 2
        ax.text(
            x_mid,
            header_bottom + header_h / 2,
            label,
            ha="center",
            va="center",
            fontsize=6.7,
            fontweight="bold",
            color="#263238",
        )

    grid_color = "#D0D0D0"
    group_line_color = "#777777"
    metric_columns = {
        3: lambda row: rank_fill(float(row["rank"])),
        4: lambda row: retained_fill(int(row["retained"]), int(row["top_k"])),
        5: lambda row: change_fill(float(row["change"])),
    }
    for row_index, row in enumerate(rows):
        group_index = row_index // len(rules)
        within_group = row_index % len(rules)
        y_top = header_bottom - row_index * row_h
        y_bottom = y_top - row_h
        if group_index % 2 == 1:
            ax.add_patch(
                plt.Rectangle(
                    (0, y_bottom),
                    total_width,
                    row_h,
                    facecolor="#FAFAFA",
                    edgecolor="none",
                    zorder=0,
                )
            )
        if within_group == 0:
            y_group_bottom = y_top - len(rules) * row_h
            cohort_fill = pale(cohort_colors[row["dataset"]], 0.46)
            ax.add_patch(
                plt.Rectangle(
                    (x_edges[0], y_group_bottom),
                    columns[0][1],
                    len(rules) * row_h,
                    facecolor=cohort_fill,
                    edgecolor="none",
                    zorder=1,
                )
            )
            ax.add_patch(
                plt.Rectangle(
                    (x_edges[0], y_group_bottom),
                    0.055,
                    len(rules) * row_h,
                    facecolor=cohort_colors[row["dataset"]],
                    edgecolor="none",
                    zorder=2,
                )
            )
            ax.text(
                (x_edges[0] + x_edges[1]) / 2 + 0.03,
                y_group_bottom + len(rules) * row_h / 2,
                str(row["cohort"]),
                ha="center",
                va="center",
                fontsize=7.0,
                fontweight="bold",
                color="#263238",
            )
        for column_index, fill_func in metric_columns.items():
            ax.add_patch(
                plt.Rectangle(
                    (x_edges[column_index], y_bottom),
                    x_edges[column_index + 1] - x_edges[column_index],
                    row_h,
                    facecolor=fill_func(row),
                    edgecolor="none",
                    zorder=1,
                )
            )
        rule_x = x_edges[1] + 0.10
        ax.add_patch(
            plt.Rectangle(
                (rule_x, y_bottom + row_h * 0.33),
                0.10,
                row_h * 0.34,
                facecolor=rule_colors[row["rule"]],
                edgecolor="#FFFFFF",
                linewidth=0.4,
                zorder=3,
            )
        )
        ax.text(
            rule_x + 0.16,
            y_bottom + row_h / 2,
            str(row["rule_label"]),
            ha="left",
            va="center",
            fontsize=6.7,
            color="#263238",
        )
        values = [
            "",
            "",
            f"{int(row['states'])}",
            f"{float(row['rank']):.3f}",
            f"{int(row['retained'])}/{int(row['top_k'])}",
            f"{float(row['change']):.2f}",
        ]
        for column_index in range(2, len(columns)):
            x_mid = (x_edges[column_index] + x_edges[column_index + 1]) / 2
            ax.text(
                x_mid,
                y_bottom + row_h / 2,
                values[column_index],
                ha="center",
                va="center",
                fontsize=6.7,
                color="#263238",
            )

    for x_value in x_edges:
        ax.plot(
            [x_value, x_value],
            [bottom_margin, table_top],
            color=grid_color,
            linewidth=0.55,
            zorder=4,
        )
    ax.plot([0, total_width], [table_top, table_top], color="#555555", linewidth=0.75, zorder=4)
    ax.plot([0, total_width], [header_bottom, header_bottom], color="#555555", linewidth=0.75, zorder=4)
    ax.plot([0, total_width], [bottom_margin, bottom_margin], color="#555555", linewidth=0.75, zorder=4)
    for row_index in range(1, len(rows)):
        y_value = header_bottom - row_index * row_h
        is_group_boundary = row_index % len(rules) == 0
        ax.plot(
            [0, total_width],
            [y_value, y_value],
            color=group_line_color if is_group_boundary else grid_color,
            linewidth=0.75 if is_group_boundary else 0.45,
            zorder=4,
        )
    ax.text(
        total_width,
        bottom_margin - 0.16,
        r"$\rho$ = Spearman correlation versus the main one-step rule; Top-10 = retained leading states.",
        ha="right",
        va="top",
        fontsize=5.9,
        color="#555555",
    )

    save_figure(
        fig,
        result_root
        / "combined_figures"
        / "Figure_E4_inflow_rule_sensitivity_three_cohorts",
        int(config["plot"]["dpi"]),
    )


def compact_inflow_edge_label(row: pd.Series) -> str:
    stage, genotype = str(row["source_state"]).split("::", 1)
    stage_label = "P" if stage == "primary" else "M"
    if genotype != "WT":
        events = genotype.split("+")
        if len(events) > 2:
            genotype = "+".join(events[:2]) + "+..."
    return f"{stage_label} | {genotype} -> +{row['event_added']}"


def canonical_inflow_edge(row: pd.Series) -> str:
    stage, genotype = str(row["source_state"]).split("::", 1)
    stage_label = "P" if stage == "primary" else "M"
    if genotype != "WT":
        genotype = "+".join(sorted(genotype.split("+")))
    return f"{stage_label} | {genotype} -> +{row['event_added']}"


def compact_canonical_edge(edge: str) -> str:
    prefix, event_added = edge.rsplit(" -> +", 1)
    stage, genotype = prefix.split(" | ", 1)
    if genotype != "WT":
        events = genotype.split("+")
        if len(events) > 3:
            genotype = "+".join(events[:3]) + "+..."
    return f"{stage} | {genotype} -> +{event_added}"


def plot_combined_dominant_edges(
    datasets: list[str], config: dict, result_root: Path
) -> None:
    per_cohort = 7
    all_tables = {}
    selected_edges = set()
    for dataset in datasets:
        edges = pd.read_csv(
            result_root
            / dataset
            / "tables"
            / "predecessor_edges_rule_a_one_step.tsv",
            sep="\t",
        )
        edges["canonical_edge"] = edges.apply(canonical_inflow_edge, axis=1)
        canonical = (
            edges.groupby("canonical_edge", as_index=False)["inflow_contribution"]
            .sum()
            .sort_values("inflow_contribution", ascending=False)
        )
        all_tables[dataset] = canonical.set_index("canonical_edge")[
            "inflow_contribution"
        ]
        selected_edges.update(canonical.head(per_cohort)["canonical_edge"])

    edge_rows = []
    for edge in selected_edges:
        values = [float(all_tables[dataset].get(edge, 0.0)) for dataset in datasets]
        edge_rows.append(
            {
                "edge": edge,
                "cohort_count": sum(value > 0 for value in values),
                "max_contribution": max(values),
                "total_contribution": sum(values),
            }
        )
    edge_order = [
        row["edge"]
        for row in sorted(
            edge_rows,
            key=lambda row: (
                row["cohort_count"],
                row["max_contribution"],
                row["total_contribution"],
            ),
            reverse=True,
        )
    ][:18]
    matrix = np.array(
        [
            [float(all_tables[dataset].get(edge, 0.0)) for dataset in datasets]
            for edge in edge_order
        ]
    )

    fig, ax = plt.subplots(figsize=(8.7, 7.9))
    fig.subplots_adjust(left=0.40, right=0.84, bottom=0.14, top=0.88)
    x, y = np.meshgrid(np.arange(len(datasets)), np.arange(len(edge_order)))
    maximum = float(matrix.max())
    nonzero = matrix[matrix > 0]
    minimum = float(nonzero.min()) if len(nonzero) else 0.0
    norm = mcolors.PowerNorm(gamma=0.55, vmin=0, vmax=maximum)
    scatter = ax.scatter(
        x.ravel(),
        y.ravel(),
        s=np.where(matrix.ravel() > 0, 35 + 900 * matrix.ravel() / maximum, 0),
        c=matrix.ravel(),
        cmap=sns.light_palette("#7A1F5C", as_cmap=True),
        norm=norm,
        edgecolor=np.where(matrix.ravel() > 0, "#333333", "none"),
        linewidth=0.45,
    )
    for row in range(len(edge_order)):
        for column in range(len(datasets)):
            value = matrix[row, column]
            if value >= maximum * 0.18:
                ax.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5.6,
                    color="white" if value >= maximum * 0.45 else "#222222",
                    fontweight="bold",
                )
    ax.set_xticks(
        np.arange(len(datasets)),
        labels=[
            config["datasets"][dataset]["display_name"] for dataset in datasets
        ],
        rotation=20,
        ha="right",
    )
    ax.set_yticks(
        np.arange(len(edge_order)),
        labels=[compact_canonical_edge(edge) for edge in edge_order],
    )
    ax.invert_yaxis()
    ax.set_xlim(-0.55, len(datasets) - 0.35)
    ax.set_ylim(len(edge_order) - 0.45, -0.95)
    ax.set_xlabel("Cancer cohort")
    ax.set_title(
        "Cross-cohort fingerprint of dominant relative-inflow edges",
        fontweight="bold",
        pad=18,
    )
    ax.grid(color="#E8E8E8", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar_ax = fig.add_axes([0.88, 0.36, 0.018, 0.36])
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label(r"Contribution to target inflow, $L_uP(u\to v)$", fontsize=7)
    cbar.ax.tick_params(labelsize=6, length=2)

    ax.text(
        0,
        1.015,
        "Circle area and color = contribution to target inflow; "
        "P = primary; M = metastatic; blank = edge absent",
        transform=ax.transAxes,
        fontsize=7,
        color="#555555",
        ha="left",
    )
    save_figure(
        fig,
        result_root
        / "combined_figures"
        / "Figure_E4_dominant_inflow_edges_three_cohorts",
        int(config["plot"]["dpi"]),
    )


def write_dataset_report(
    dataset: str,
    config: dict,
    summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    main: pd.DataFrame,
    edges: pd.DataFrame,
    output: Path,
) -> None:
    main_summary = summary[summary["rule"].eq(config["main_rule"]["name"])].iloc[0]
    top_states = main.nlargest(8, "F_hat")
    top_edges = top_edge_table(edges, 8)
    lines = [
        f"# Experiment 4: {config['datasets'][dataset]['display_name']}",
        "",
        "## Primary Rule",
        "",
        "Rule A uses observed same-stage states that differ by exactly one event.",
        "Unknown stages and explicitly excluded model-system specimens are outside the analysis universe.",
        "",
        f"- Analysis samples: {int(main_summary['analysis_samples'])}",
        f"- Observed states: {int(main_summary['observed_states'])}",
        f"- States with positive inflow: {int(main_summary['positive_inflow_states'])}",
        f"- Stable states for Experiment 5: {int(main_summary['stable_states'])}",
        f"- Stable-state sample fraction: {main_summary['stable_sample_fraction']:.3f}",
        f"- Spearman(L, F_hat): {main_summary['spearman_L_vs_F']:.3f}",
        "",
        "## Highest Relative Inflow States",
        "",
    ]
    for row in top_states.itertuples(index=False):
        lines.append(
            f"- {row.state}: F_hat={row.F_hat:.6g}, N={row.N_v}, "
            f"dominant predecessor={row.dominant_predecessor or 'none'}."
        )
    lines.extend(["", "## Dominant Inflow Edges", ""])
    for row in top_edges.itertuples(index=False):
        lines.append(
            f"- {row.source_state} -> {row.target_state}: "
            f"L_u P={row.inflow_contribution:.6g}, P={row.edge_probability:.4f}."
        )
    lines.extend(["", "## Sensitivity", ""])
    for row in sensitivity.itertuples(index=False):
        lines.append(
            f"- {row.comparison_rule}: rho={row.spearman_F_hat:.3f}, "
            f"Top-{row.top_k} overlap={row.top_k_overlap:.2f}, "
            f"stable states={row.stable_states}."
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "F_hat is an internally inferred relative inflow, not an absolute patient flow rate.",
            "States with zero or near-zero inflow are retained but flagged for Experiment 5.",
            "No bottleneck or dwell-time claim is made until R* is computed.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_dataset(dataset: str, config: dict, result_root: Path) -> dict:
    dataset_root = result_root / dataset
    tables = dataset_root / "tables"
    figures = dataset_root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    occupancy, events, stage_qc = prepare_states(dataset, config)
    exp3_tables = Path(config["experiment_03_root"]) / dataset / "tables"
    transitions = pd.read_csv(exp3_tables / "genotype_transition.tsv", sep="\t")
    theta = pd.read_csv(exp3_tables / "theta.tsv", sep="\t", index_col=0)
    if theta.columns.astype(str).tolist() != events:
        raise ValueError(f"{dataset}: Experiment 3 theta event order mismatch")

    rule_tables, edge_tables = make_rule_outputs(
        dataset, config, occupancy, events, transitions, theta
    )
    sensitivity = sensitivity_metrics(rule_tables, config)
    summary = summary_metrics(dataset, rule_tables, config)

    occupancy.to_csv(tables / "state_occupancy_experiment4.tsv", sep="\t", index=False)
    stage_qc.to_csv(tables / "stage_inclusion_qc.tsv", sep="\t", index=False)
    for rule, table in rule_tables.items():
        table.to_csv(tables / f"inflow_table_{rule}.tsv", sep="\t", index=False)
    for rule, edges in edge_tables.items():
        edges.to_csv(
            tables / f"predecessor_edges_{rule}.tsv", sep="\t", index=False
        )
    sensitivity.to_csv(tables / "inflow_rule_sensitivity.tsv", sep="\t", index=False)
    summary.to_csv(tables / "experiment_04_metrics.tsv", sep="\t", index=False)

    plot_dataset(
        dataset,
        config,
        rule_tables,
        edge_tables,
        sensitivity,
        figures / "Figure_E4_relative_inflow",
    )
    write_dataset_report(
        dataset,
        config,
        summary,
        sensitivity,
        rule_tables[config["main_rule"]["name"]],
        edge_tables[config["main_rule"]["name"]],
        dataset_root / "experiment_04_report.md",
    )
    record = summary[summary["rule"].eq(config["main_rule"]["name"])].iloc[0].to_dict()
    record["stage_excluded_samples"] = int(
        stage_qc.loc[~stage_qc["included"], "samples"].sum()
    )
    record["main_edges"] = len(edge_tables[config["main_rule"]["name"]])
    logging.info("%s complete: %s", dataset, record)
    return record


def write_summary(
    result_root: Path, records: list[dict], config: dict
) -> None:
    summary = pd.DataFrame(records)
    summary.to_csv(result_root / "experiment_04_summary.csv", index=False)
    columns = [
        "dataset_name",
        "analysis_samples",
        "observed_states",
        "main_edges",
        "positive_inflow_states",
        "zero_inflow_states",
        "stable_states",
        "stable_sample_fraction",
        "spearman_L_vs_F",
        "stage_excluded_samples",
    ]
    lines = [
        "# Experiment 4 Summary",
        "",
        "Primary results use same-stage, one-event predecessors. Stage bridges, "
        "two-event predecessors, and occupancy smoothing are sensitivity analyses.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in summary.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    (result_root / "experiment_04_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_literature_design_notes(result_root: Path) -> None:
    text = """# Bioinformatics Figure-Design Review for Experiment 4

The Experiment 4 figures use recurring design practices from cancer-progression
method papers in *Bioinformatics*:

1. Compact multi-panel evidence chains: input/model diagnostics are separated
   from biologically interpretable transition results.
2. Stable visual encodings: stage is represented by a fixed color mapping,
   transition contribution by bar length, and uncertainty/sensitivity by
   aligned quantitative panels.
3. Sparse transition summaries: only dominant edges are labeled in the main
   figure, while complete edge tables remain machine-readable.
4. Direct method comparison: alternative predecessor rules are shown on the
   same 0-1 agreement scale using rank correlation and Top-K overlap.
5. Vector-first output and restrained typography: Arial-compatible fonts,
   compact labels, colorblind-conscious palettes, PDF plus 600-dpi PNG.

Articles reviewed:

- Schill et al. Modelling cancer progression using Mutual Hazard Networks.
  Bioinformatics 2020. https://doi.org/10.1093/bioinformatics/btz513
- Modeling metastatic progression from cross-sectional cancer genomics data
  (metMHN). Bioinformatics 2024.
  https://doi.org/10.1093/bioinformatics/btae250
- HyperHMM: efficient inference of evolutionary and progressive dynamics on
  hypercubic transition graphs. Bioinformatics 2023.
  https://doi.org/10.1093/bioinformatics/btac803
- De Sano et al. TRONCO: an R package for the inference of cancer progression
  models from heterogeneous genomic data. Bioinformatics 2016.
  https://doi.org/10.1093/bioinformatics/btw035

The figures borrow organizational principles only; no published figure is
copied or recreated.

## Additional top-journal redesign review

The cross-cohort figures also apply compact matrix and multivariate-encoding
principles used in trajectory benchmarking and cancer-evolution studies:

- Saelens et al., Nature Biotechnology 2019:
  https://www.nature.com/articles/s41587-019-0071-9
- PCAWG Evolution and Heterogeneity Working Group, Nature 2020:
  https://www.nature.com/articles/s41586-019-1907-7
- TRACERx metastatic evolution study, Nature 2023:
  https://doi.org/10.1038/s41586-023-05729-x
- PAGA graph abstraction study, Genome Biology 2019:
  https://doi.org/10.1186/s13059-019-1663-x

The adopted principles are shared coordinates across cohorts, restrained
continuous palettes, redundant quantitative encodings only where they improve
pattern recognition, and compact legends placed next to their visual channel.

## Cross-domain candidate designs

Four additional alternatives were generated after reviewing visual grammars
used in artificial-intelligence benchmarking, computer visualization,
multi-objective optimization, and bioinformatics method comparisons:

1. Pareto stability maps place inflow perturbation and rank correlation on
   quantitative axes and encode retained states in point area and text.
2. Parallel stability profiles normalize all three dimensions so higher values
   consistently indicate robustness.
3. A tri-axis glyph matrix provides the most compact cohort-by-rule overview
   and exposes balanced versus dimension-specific robustness.
4. A benchmark dot table preserves independent numerical axes and maximizes
   auditability without using bar-like baselines.

Relevant design sources include:

- Luecken et al., benchmarking atlas-level data integration in single-cell
  genomics, Nature Methods 2022:
  https://doi.org/10.1038/s41592-021-01336-8
- Borgo et al., glyph-based visualization design guidelines, EuroVis/Computer
  Graphics Forum 2013:
  https://doi.org/10.1111/cgf.12128
- Saelens et al., trajectory-inference benchmarking, Nature Biotechnology 2019:
  https://www.nature.com/articles/s41587-019-0071-9
- PAGA graph abstraction and aligned comparative displays, Genome Biology
  2019: https://doi.org/10.1186/s13059-019-1663-x

## Final legend semantics

- Edge contribution is L_u P(u -> v): source-state relative occupancy
  multiplied by transition probability. These terms sum to the inferred
  target-state inflow F_hat_v. It is not a patient proportion, absolute flow
  rate, or MHN log-effect.
- Top-10 state overlap is the number of states shared by the ten highest-inflow
  states under the main and sensitivity rules, divided by 10. Thus, 0.8 means
  eight of the ten leading states are retained.
- In the dominant-edge matrix, circle area and color redundantly encode the
  same edge-contribution value; one continuous color bar is sufficient.
- The final sensitivity figure is an exact audit table rather than an
  axis-heavy dot plot. It keeps the same row order across cohorts and exposes
  the three key sensitivity metrics directly: Spearman rank correlation,
  retained Top-10 states, and median absolute relative inflow change. Subtle
  cell shading supports scanning, while the numeric values remain the primary
  evidence.
"""
    (result_root / "bioinformatics_figure_design_review.md").write_text(
        text, encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    selected = yaml.safe_load(Path(args.dataset_config).read_text(encoding="utf-8"))
    datasets = [entry["dataset_name"] for entry in selected["included_datasets"]]
    result_root = Path(config["result_root"]).resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    configure_plotting(config)
    setup_logging(result_root)

    records = []
    for dataset in datasets:
        print(f"[Experiment 4] Computing {dataset}...", flush=True)
        records.append(run_dataset(dataset, config, result_root))
        print(f"[Experiment 4] {dataset} complete.", flush=True)
    write_summary(result_root, records, config)
    write_literature_design_notes(result_root)
    plot_combined(datasets, config, result_root)
    plot_combined_rule_sensitivity(datasets, config, result_root)
    plot_combined_dominant_edges(datasets, config, result_root)


if __name__ == "__main__":
    main()

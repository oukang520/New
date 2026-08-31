"""Run Experiment 5: relative dwell R* and observation enrichment O*.

R* is the primary state-level result. O* is an auxiliary residual diagnostic
against a progression-only cMHN simulation and must not be interpreted as a
clinical observation or diagnosis rate.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
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


STAGE_COLORS = {"primary": "#4477AA", "metastatic": "#CC6677"}
PATHWAY_GROUPS = {
    "p53/genome integrity": {"TP53"},
    "DNA repair": {"ATM", "ATRX", "SETD2", "BRCA1", "BRCA2"},
    "RTK-MAPK": {
        "KRAS",
        "NRAS",
        "EGFR",
        "BRAF",
        "NF1",
        "MET",
        "ALK",
        "ERBB2",
        "ERBB4",
        "MAP3K1",
        "MAP2K4",
    },
    "PI3K-AKT": {"PIK3CA", "PTEN", "AKT1"},
    "cell cycle": {"CDKN2A", "RB1", "CCND1"},
    "WNT": {"APC", "CTNNB1"},
    "TGF-beta": {"SMAD4", "TGFBR2"},
    "chromatin": {"ARID1A", "SMARCA4", "KMT2D", "KMT2C", "KDM6A"},
    "stress/metabolism": {"STK11", "KEAP1"},
    "hormone/luminal": {"ESR1", "GATA3", "FOXA1"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Rel-ObsTQ-MHN Experiment 5.")
    parser.add_argument("--config", default="configs/experiment_05.yaml")
    parser.add_argument(
        "--dataset-config", default="configs/selected_experiment_datasets.yaml"
    )
    return parser.parse_args()


def configure_plotting(config: dict) -> None:
    figure_style.configure_matplotlib(config)


def save_figure(fig: plt.Figure, base_path: Path, dpi: int) -> None:
    figure_style.save_figure(fig, base_path, {"plot": {"dpi": dpi}}, dpi=dpi)


def cleanup_figure_outputs(base_path: Path) -> None:
    """Remove stale composite or panel-crop outputs before writing single figures."""
    base_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".png", ".pdf"]:
        candidate = base_path.with_suffix(suffix)
        if candidate.exists():
            candidate.unlink()
    for candidate in base_path.parent.glob(f"{base_path.name}__*"):
        if candidate.suffix.lower() in {".png", ".pdf"}:
            candidate.unlink()


def genotype_from_mask(mask: int, events: list[str]) -> str:
    selected = [event for index, event in enumerate(events) if mask & (1 << index)]
    return "+".join(selected) if selected else "WT"


def compact_state(state: str, max_events: int = 3) -> str:
    stage, genotype = state.split("::", 1)
    events = [] if genotype == "WT" else genotype.split("+")
    if len(events) > max_events:
        genotype = "+".join(events[:max_events]) + "+..."
    prefix = "P" if stage == "primary" else "M"
    return f"{prefix} | {genotype}"


def compact_genotype(genotype: str, max_events: int = 4) -> str:
    events = [] if genotype == "WT" else str(genotype).split("+")
    if len(events) > max_events:
        return "+".join(events[:max_events]) + "+..."
    return str(genotype)


def biological_annotation(genotype: str) -> str:
    events = set() if genotype == "WT" else set(genotype.split("+"))
    labels = [
        label for label, members in PATHWAY_GROUPS.items() if events.intersection(members)
    ]
    return "; ".join(labels) if labels else "other/WT"


def progression_simulation(
    theta: np.ndarray,
    events: list[str],
    event_burdens: np.ndarray,
    observed_states: list[str],
    simulations: int,
    stage_mass: float,
    alpha: float,
    seed: int,
) -> tuple[pd.Series, float]:
    rng = np.random.default_rng(seed)
    sampled_burdens = rng.choice(event_burdens, size=simulations, replace=True)
    transition_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    counts: Counter[str] = Counter()
    observed_set = set(observed_states)

    for burden in sampled_burdens:
        mask = 0
        stage = "primary"
        for _ in range(int(burden)):
            if stage == "primary" and rng.random() < stage_mass:
                stage = "metastatic"
            if mask not in transition_cache:
                present = np.array(
                    [bool(mask & (1 << index)) for index in range(len(events))],
                    dtype=bool,
                )
                absent = np.flatnonzero(~present)
                if len(absent) == 0:
                    transition_cache[mask] = (absent, np.array([], dtype=float))
                else:
                    logits = np.array(
                        [
                            theta[event_index, event_index]
                            + theta[event_index, present].sum()
                            for event_index in absent
                        ],
                        dtype=float,
                    )
                    scaled = np.exp(logits - logits.max())
                    transition_cache[mask] = (absent, scaled / scaled.sum())
            absent, probabilities = transition_cache[mask]
            if len(absent) == 0:
                break
            event_index = int(rng.choice(absent, p=probabilities))
            mask |= 1 << event_index
        if stage == "primary" and rng.random() < stage_mass:
            stage = "metastatic"
        state = f"{stage}::{genotype_from_mask(mask, events)}"
        counts[state] += 1

    observed_simulations = sum(counts[state] for state in observed_states)
    denominator = observed_simulations + alpha * len(observed_states)
    expected = pd.Series(
        {
            state: (counts[state] + alpha) / denominator
            for state in observed_states
        },
        dtype=float,
    )
    support_coverage = observed_simulations / simulations
    return expected, support_coverage


def bootstrap_r_star(
    state_table: pd.DataFrame,
    edges: pd.DataFrame,
    thresholds: dict,
    bootstrap: dict,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    states = state_table["state"].astype(str).tolist()
    state_index = {state: index for index, state in enumerate(states)}
    counts = state_table["N_v"].to_numpy(dtype=int)
    total = int(counts.sum())
    probabilities = counts / total
    edge_source = np.array(
        [state_index.get(state, -1) for state in edges["source_state"].astype(str)],
        dtype=int,
    )
    edge_target = np.array(
        [state_index.get(state, -1) for state in edges["target_state"].astype(str)],
        dtype=int,
    )
    edge_probability = edges["edge_probability"].to_numpy(dtype=float)
    valid_edge = (edge_source >= 0) & (edge_target >= 0)
    edge_source = edge_source[valid_edge]
    edge_target = edge_target[valid_edge]
    edge_probability = edge_probability[valid_edge]

    replicates = int(bootstrap["replicates"])
    epsilon = float(thresholds["epsilon"])
    minimum_count = int(thresholds["minimum_state_count"])
    high_confidence_count = int(thresholds["high_confidence_state_count"])
    minimum_inflow = float(thresholds["minimum_inflow"])
    top_k = int(thresholds["top_k"])
    rng = np.random.default_rng(seed)
    values = np.full((replicates, len(states)), np.nan, dtype=float)
    top_counts = np.zeros(len(states), dtype=int)
    high_confidence_top_counts = np.zeros(len(states), dtype=int)

    for replicate in range(replicates):
        sampled_counts = rng.multinomial(total, probabilities)
        occupancy = sampled_counts / total
        inflow = np.zeros(len(states), dtype=float)
        np.add.at(
            inflow,
            edge_target,
            occupancy[edge_source] * edge_probability,
        )
        eligible = (sampled_counts >= minimum_count) & (inflow >= minimum_inflow)
        raw = occupancy / (inflow + epsilon)
        normalizer = np.median(raw[eligible]) if eligible.any() else np.nan
        if not np.isfinite(normalizer) or normalizer <= 0:
            continue
        values[replicate, eligible] = raw[eligible] / normalizer
        eligible_indices = np.flatnonzero(eligible)
        if len(eligible_indices):
            order = eligible_indices[np.argsort(values[replicate, eligible_indices])[::-1]]
            top_counts[order[:top_k]] += 1
        high_confidence_eligible = eligible & (
            sampled_counts >= high_confidence_count
        )
        high_confidence_indices = np.flatnonzero(high_confidence_eligible)
        if len(high_confidence_indices):
            order = high_confidence_indices[
                np.argsort(values[replicate, high_confidence_indices])[::-1]
            ]
            high_confidence_top_counts[order[:top_k]] += 1

    alpha = (1 - float(bootstrap["confidence_level"])) / 2
    medians = np.full(len(states), np.nan)
    ci_low = np.full(len(states), np.nan)
    ci_high = np.full(len(states), np.nan)
    valid_counts = np.isfinite(values).sum(axis=0)
    for index in np.flatnonzero(valid_counts):
        finite = values[np.isfinite(values[:, index]), index]
        medians[index] = np.median(finite)
        ci_low[index] = np.quantile(finite, alpha)
        ci_high[index] = np.quantile(finite, 1 - alpha)
    summary = pd.DataFrame(
        {
            "state": states,
            "bootstrap_median_R_star": medians,
            "R_star_ci_low": ci_low,
            "R_star_ci_high": ci_high,
            "bootstrap_valid_replicates": valid_counts,
            "stability": top_counts / replicates,
            "stability_high_confidence": high_confidence_top_counts / replicates,
        }
    )
    long_rows = []
    for replicate in range(replicates):
        finite = np.flatnonzero(np.isfinite(values[replicate]))
        for index in finite:
            long_rows.append(
                {
                    "replicate": replicate + 1,
                    "state": states[index],
                    "R_star": values[replicate, index],
                }
            )
    return summary, pd.DataFrame(long_rows)


def next_state_map(edges: pd.DataFrame) -> dict[str, str]:
    result = {}
    ordered = edges.sort_values(
        ["source_state", "inflow_contribution"], ascending=[True, False]
    )
    for source, group in ordered.groupby("source_state"):
        result[str(source)] = "; ".join(
            compact_state(state) for state in group["target_state"].head(3)
        )
    return result


def compute_dataset(
    dataset: str, config: dict, result_root: Path
) -> dict:
    exp4 = Path(config["experiment_04_root"]) / dataset / "tables"
    exp3 = Path(config["experiment_03_root"]) / dataset / "tables"
    output = result_root / dataset
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    inflow = pd.read_csv(exp4 / "inflow_table_rule_a_one_step.tsv", sep="\t")
    edges = pd.read_csv(exp4 / "predecessor_edges_rule_a_one_step.tsv", sep="\t")
    occupancy = pd.read_csv(exp4 / "state_occupancy_experiment4.tsv", sep="\t")
    theta_frame = pd.read_csv(exp3 / "theta.tsv", sep="\t", index_col=0)
    events = theta_frame.columns.astype(str).tolist()
    theta = theta_frame.to_numpy(dtype=float)
    thresholds = config["thresholds"]

    eligible = (
        (inflow["N_v"] >= int(thresholds["minimum_state_count"]))
        & (inflow["F_hat"] >= float(thresholds["minimum_inflow"]))
    )
    epsilon = float(thresholds["epsilon"])
    inflow["R_v"] = inflow["L_v"] / (inflow["F_hat"] + epsilon)
    normalizer = float(inflow.loc[eligible, "R_v"].median())
    inflow["R_star"] = inflow["R_v"] / normalizer
    inflow["log2_R_star"] = np.log2(inflow["R_star"].clip(lower=1e-12))
    inflow["eligible_experiment5"] = eligible
    inflow["high_confidence"] = (
        eligible
        & (inflow["N_v"] >= int(thresholds["high_confidence_state_count"]))
    )

    event_burdens = occupancy.loc[
        occupancy["stage"].isin(["primary", "metastatic"]), "event_count"
    ].to_numpy(dtype=int)
    observed_states = inflow["state"].astype(str).tolist()
    progression = config["progression_only"]
    expected, support_coverage = progression_simulation(
        theta,
        events,
        event_burdens,
        observed_states,
        int(progression["simulations"]),
        float(progression["stage_transition_mass"]),
        float(progression["dirichlet_alpha"]),
        int(config["random_seed"]) + sum(ord(char) for char in dataset),
    )
    inflow["Lhat_progression"] = inflow["state"].map(expected)
    inflow["O_star"] = inflow["L_v"] / (
        inflow["Lhat_progression"] + epsilon
    )
    inflow["log2_O_star"] = np.log2(inflow["O_star"].clip(lower=1e-12))

    sensitivity_rows = []
    expected_by_mass = {}
    for index, stage_mass in enumerate(
        progression["stage_transition_mass_sensitivity"]
    ):
        expected_mass, coverage = progression_simulation(
            theta,
            events,
            event_burdens,
            observed_states,
            int(progression["simulations"]),
            float(stage_mass),
            float(progression["dirichlet_alpha"]),
            int(config["random_seed"]) + 1000 * (index + 1) + sum(ord(c) for c in dataset),
        )
        expected_by_mass[float(stage_mass)] = expected_mass
        o_mass = inflow["L_v"] / (
            inflow["state"].map(expected_mass) + epsilon
        )
        comparison = eligible & np.isfinite(o_mass) & np.isfinite(inflow["O_star"])
        rho = spearmanr(
            inflow.loc[comparison, "O_star"],
            o_mass.loc[comparison],
        ).statistic
        sensitivity_rows.append(
            {
                "stage_transition_mass": float(stage_mass),
                "support_coverage": coverage,
                "spearman_O_star_vs_main": float(rho),
            }
        )
    pd.DataFrame(sensitivity_rows).to_csv(
        tables / "progression_only_sensitivity.tsv", sep="\t", index=False
    )

    bootstrap_summary, bootstrap_long = bootstrap_r_star(
        inflow[["state", "N_v"]].copy(),
        edges,
        thresholds,
        config["bootstrap"],
        int(config["random_seed"]) + 5000 + sum(ord(c) for c in dataset),
    )
    inflow = inflow.merge(bootstrap_summary, on="state", how="left")
    inflow["genotype"] = inflow["genotype"].fillna("WT")
    inflow["clinical_annotation"] = inflow["genotype"].map(biological_annotation)
    inflow["possible_next_states"] = inflow["state"].map(next_state_map(edges)).fillna("")
    inflow["direction_flag"] = np.select(
        [inflow["R_star"] > 1, inflow["R_star"] < 1],
        ["relative_bottleneck", "fast_passing"],
        default="neutral",
    )
    inflow["interpretation_flag"] = np.select(
        [
            (inflow["R_star"] > 1) & (inflow["O_star"] > 1),
            (inflow["R_star"] > 1) & (inflow["O_star"] <= 1),
            (inflow["R_star"] <= 1) & (inflow["O_star"] > 1),
        ],
        [
            "bottleneck_with_observation_enrichment",
            "bottleneck_without_observation_enrichment",
            "observation_enrichment_without_bottleneck",
        ],
        default="fast_or_neutral_without_enrichment",
    )
    inflow.to_csv(tables / "state_scores.tsv", sep="\t", index=False)
    bootstrap_long.to_csv(tables / "bootstrap_R_star.tsv", sep="\t", index=False)

    stable = inflow[inflow["eligible_experiment5"]].copy()
    top_k = int(thresholds["top_k"])
    bottleneck = stable.nlargest(top_k, "R_star").copy()
    bottleneck.insert(0, "rank", range(1, len(bottleneck) + 1))
    bottleneck[
        [
            "rank",
            "state",
            "R_star",
            "R_star_ci_low",
            "R_star_ci_high",
            "stability",
            "stability_high_confidence",
            "N_v",
            "dominant_predecessor",
            "clinical_annotation",
        ]
    ].to_csv(tables / "top_bottleneck_states.tsv", sep="\t", index=False)
    high_confidence_bottleneck = stable[stable["high_confidence"]].nlargest(
        top_k, "R_star"
    ).copy()
    high_confidence_bottleneck.insert(
        0, "rank", range(1, len(high_confidence_bottleneck) + 1)
    )
    high_confidence_bottleneck[
        [
            "rank",
            "state",
            "R_star",
            "R_star_ci_low",
            "R_star_ci_high",
            "stability",
            "stability_high_confidence",
            "N_v",
            "dominant_predecessor",
            "clinical_annotation",
        ]
    ].to_csv(
        tables / "top_bottleneck_states_high_confidence.tsv",
        sep="\t",
        index=False,
    )
    fast = stable.nsmallest(top_k, "R_star").copy()
    fast.insert(0, "rank", range(1, len(fast) + 1))
    fast[
        [
            "rank",
            "state",
            "R_star",
            "R_star_ci_low",
            "R_star_ci_high",
            "stability",
            "stability_high_confidence",
            "N_v",
            "possible_next_states",
        ]
    ].to_csv(tables / "fast_passing_states.tsv", sep="\t", index=False)
    enriched = stable.nlargest(top_k, "O_star").copy()
    enriched.insert(0, "rank", range(1, len(enriched) + 1))
    enriched[
        [
            "rank",
            "state",
            "O_star",
            "R_star",
            "stability",
            "N_v",
            "interpretation_flag",
        ]
    ].to_csv(tables / "high_observation_enrichment.tsv", sep="\t", index=False)

    metrics = {
        "dataset_name": dataset,
        "states_total": int(len(inflow)),
        "states_eligible": int(eligible.sum()),
        "states_high_confidence": int(inflow["high_confidence"].sum()),
        "median_R_raw": normalizer,
        "progression_support_coverage": support_coverage,
        "median_O_star": float(stable["O_star"].median()),
        "top_bottleneck_stability": float(bottleneck["stability"].mean()),
        "top_high_confidence_stability": float(
            high_confidence_bottleneck["stability_high_confidence"].mean()
        ),
        "bootstrap_replicates": int(config["bootstrap"]["replicates"]),
    }
    pd.DataFrame([metrics]).to_csv(
        tables / "experiment_05_metrics.tsv", sep="\t", index=False
    )
    plot_dataset_single(dataset, inflow, config, figures / "Figure_E5_state_scores")
    write_dataset_report(dataset, inflow, metrics, output / "experiment_05_report.md")
    return metrics


def label_selected(
    ax: plt.Axes,
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    count: int = 4,
) -> None:
    selected = pd.concat(
        [
            frame.nlargest(count, "R_star"),
            frame.nsmallest(max(2, count // 2), "R_star"),
        ]
    ).drop_duplicates("state")
    for _, row in selected.iterrows():
        ax.annotate(
            compact_state(row["state"], 2),
            (row[x_column], row[y_column]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=5.2,
            color="#333333",
        )


def plot_dataset(
    dataset: str, scores: pd.DataFrame, config: dict, output: Path
) -> None:
    stable = scores[scores["eligible_experiment5"]].copy()
    thresholds = config["thresholds"]
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 8.0))
    fig.subplots_adjust(
        left=0.09, right=0.985, bottom=0.085, top=0.90, wspace=0.40, hspace=0.42
    )
    r_limit = max(2.0, float(np.nanquantile(np.abs(stable["log2_R_star"]), 0.98)))
    o_limit = max(2.0, float(np.nanquantile(np.abs(stable["log2_O_star"]), 0.98)))
    r_norm = mcolors.TwoSlopeNorm(vmin=-r_limit, vcenter=0, vmax=r_limit)
    o_norm = mcolors.TwoSlopeNorm(vmin=-o_limit, vcenter=0, vmax=o_limit)

    ax = axes[0, 0]
    scatter = ax.scatter(
        stable["F_hat"],
        stable["L_v"],
        c=stable["log2_R_star"],
        s=np.clip(12 + 5 * np.sqrt(stable["N_v"]), 16, 80),
        cmap="coolwarm",
        norm=r_norm,
        edgecolor="#333333",
        linewidth=0.35,
        alpha=0.78,
    )
    limits = [
        min(stable["F_hat"].min(), stable["L_v"].min()) * 0.60,
        max(stable["F_hat"].max(), stable["L_v"].max()) * 1.65,
    ]
    ax.plot(limits, limits, color="#777777", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel(r"Inferred relative inflow $\hat{F}_v$")
    ax.set_ylabel(r"Observed occupancy $L_v$")
    ax.set_title(r"Flow-corrected state landscape")
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label(r"$\log_2 R^*$")
    sns.despine(ax=ax)

    ax = axes[0, 1]
    selected_genotypes = (
        pd.concat(
            [
                stable.nlargest(int(thresholds["heatmap_genotypes"]), "R_star"),
                stable.nsmallest(4, "R_star"),
            ]
        )
        .sort_values("R_star", ascending=False)["genotype"]
        .drop_duplicates()
        .head(int(thresholds["heatmap_genotypes"]))
        .tolist()
    )
    heat = (
        stable[stable["genotype"].isin(selected_genotypes)]
        .pivot_table(index="genotype", columns="stage", values="log2_R_star", aggfunc="first")
        .reindex(selected_genotypes)
        .reindex(columns=["primary", "metastatic"])
    )
    sns.heatmap(
        heat,
        ax=ax,
        cmap="coolwarm",
        center=0,
        vmin=-r_limit,
        vmax=r_limit,
        linewidths=0.5,
        linecolor="white",
        annot=True,
        fmt=".1f",
        annot_kws={"fontsize": 5.3},
        cbar_kws={"label": r"$\log_2 R^*$", "fraction": 0.06, "pad": 0.03},
        mask=heat.isna(),
    )
    ax.set_xlabel("Disease compartment")
    ax.set_ylabel("")
    ax.set_title(r"Stage–genotype $R^*$ map")
    ax.tick_params(axis="y", labelsize=5.5)

    ax = axes[1, 0]
    ranked = pd.concat(
        [
            stable.nlargest(int(thresholds["bottleneck_display"]), "R_star"),
            stable.nsmallest(int(thresholds["fast_display"]), "R_star"),
        ]
    ).drop_duplicates("state")
    ranked = ranked.sort_values("R_star")
    y = np.arange(len(ranked))
    median = np.log2(ranked["bootstrap_median_R_star"].clip(lower=1e-12))
    low = np.log2(ranked["R_star_ci_low"].clip(lower=1e-12))
    high = np.log2(ranked["R_star_ci_high"].clip(lower=1e-12))
    ax.errorbar(
        median,
        y,
        xerr=np.vstack([median - low, high - median]),
        fmt="o",
        color="#333333",
        ecolor="#777777",
        elinewidth=0.8,
        capsize=2,
        markersize=3.8,
    )
    direction_colors = np.where(ranked["R_star"] >= 1, "#B24773", "#4C78A8")
    ax.scatter(
        np.log2(ranked["R_star"]),
        y,
        color=direction_colors,
        s=28,
        edgecolor="#333333",
        linewidth=0.35,
        zorder=3,
    )
    ax.axvline(0, color="#777777", ls="--", lw=0.7)
    ax.set_yticks(y, labels=[compact_state(state) for state in ranked["state"]])
    ax.tick_params(axis="y", labelsize=5.2)
    ax.set_xlabel(r"$\log_2 R^*$ with bootstrap 95% CI")
    ax.set_title("Bottleneck and fast-passing states")
    sns.despine(ax=ax)

    ax = axes[1, 1]
    scatter_o = ax.scatter(
        stable["Lhat_progression"],
        stable["L_v"],
        c=stable["log2_O_star"],
        s=np.clip(12 + 5 * np.sqrt(stable["N_v"]), 16, 80),
        cmap="PiYG_r",
        norm=o_norm,
        edgecolor="#333333",
        linewidth=0.35,
        alpha=0.78,
    )
    limits_o = [
        min(stable["Lhat_progression"].min(), stable["L_v"].min()) * 0.60,
        max(stable["Lhat_progression"].max(), stable["L_v"].max()) * 1.65,
    ]
    ax.plot(limits_o, limits_o, color="#777777", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(limits_o)
    ax.set_ylim(limits_o)
    ax.set_xlabel(r"Progression-only expected occupancy $\hat{L}_v$")
    ax.set_ylabel(r"Observed occupancy $L_v$")
    ax.set_title(r"Observation-enrichment residual")
    cbar = fig.colorbar(scatter_o, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label(r"$\log_2 O^*$")
    sns.despine(ax=ax)

    for index, ax in enumerate(axes.ravel()):
        ax.text(
            -0.15,
            1.08,
            chr(ord("A") + index),
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
        )
    fig.suptitle(
        f"{config['datasets'][dataset]['display_name']} | Experiment 5 state scores",
        fontweight="bold",
        y=0.975,
    )
    save_figure(fig, output, int(config["plot"]["dpi"]))


def plot_combined(
    datasets: list[str], config: dict, result_root: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.6))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.09, top=0.88, wspace=0.25, hspace=0.34)
    for index, (axis, dataset) in enumerate(zip(axes.ravel(), datasets)):
        scores = pd.read_csv(result_root / dataset / "tables" / "state_scores.tsv", sep="\t")
        stable = scores[scores["eligible_experiment5"]].copy()
        scatter = axis.scatter(
            stable["log2_R_star"],
            stable["log2_O_star"],
            s=np.clip(12 + 5 * np.sqrt(stable["N_v"]), 16, 70),
            c=stable["stability_high_confidence"],
            cmap=sns.light_palette("#006D77", as_cmap=True),
            vmin=0,
            vmax=1,
            edgecolor=np.where(stable["stage"].eq("metastatic"), "#CC6677", "#333333"),
            linewidth=0.55,
            alpha=0.76,
        )
        axis.axhline(0, color="#888888", lw=0.7)
        axis.axvline(0, color="#888888", lw=0.7)
        axis.set_xlabel(r"$\log_2 R^*$")
        axis.set_ylabel(r"$\log_2 O^*$")
        axis.set_title(config["datasets"][dataset]["display_name"], loc="left", fontweight="bold")
        high_confidence = stable[stable["high_confidence"]].copy()
        high_confidence["distance"] = np.sqrt(
            high_confidence["log2_R_star"] ** 2
            + high_confidence["log2_O_star"] ** 2
        )
        extreme = high_confidence.nlargest(1, "distance")
        for _, row in extreme.iterrows():
            axis.annotate(
                compact_state(row["state"], 2),
                (row["log2_R_star"], row["log2_O_star"]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=5.2,
            )
        axis.margins(x=0.16, y=0.16)
        axis.text(
            -0.14,
            1.07,
            chr(ord("A") + index),
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
        )
        sns.despine(ax=axis)
    cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("Bootstrap Top-10 stability (N≥10)")
    fig.suptitle(
        r"Experiment 5 | Relative dwell and observation-enrichment landscape",
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.925,
        "Point size = state count; red edge = metastatic state; O* is an auxiliary progression-only residual",
        ha="center",
        fontsize=7,
        color="#555555",
    )
    save_figure(
        fig,
        result_root / "combined_figures" / "Figure_E5_core_results_three_cohorts",
        int(config["plot"]["dpi"]),
    )


def plot_dataset_single(
    dataset: str, scores: pd.DataFrame, config: dict, output: Path
) -> None:
    stable = scores[scores["eligible_experiment5"]].copy()
    thresholds = config["thresholds"]
    cleanup_figure_outputs(output)
    dpi = int(config["plot"]["dpi"])
    r_limit = max(2.0, float(np.nanquantile(np.abs(stable["log2_R_star"]), 0.98)))
    o_limit = max(2.0, float(np.nanquantile(np.abs(stable["log2_O_star"]), 0.98)))
    r_norm = mcolors.TwoSlopeNorm(vmin=-r_limit, vcenter=0, vmax=r_limit)
    o_norm = mcolors.TwoSlopeNorm(vmin=-o_limit, vcenter=0, vmax=o_limit)

    fig, ax = plt.subplots(figsize=(3.25, 3.25))
    fig.subplots_adjust(left=0.17, right=0.84, bottom=0.17, top=0.96)
    scatter = ax.scatter(
        stable["F_hat"],
        stable["L_v"],
        c=stable["log2_R_star"],
        s=np.clip(12 + 5 * np.sqrt(stable["N_v"]), 16, 80),
        cmap="coolwarm",
        norm=r_norm,
        edgecolor="#333333",
        linewidth=0.35,
        alpha=0.78,
    )
    limits = [
        min(stable["F_hat"].min(), stable["L_v"].min()) * 0.60,
        max(stable["F_hat"].max(), stable["L_v"].max()) * 1.65,
    ]
    ax.plot(limits, limits, color="#777777", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel(r"Inferred relative inflow $\hat{F}_v$")
    ax.set_ylabel(r"Observed occupancy $L_v$")
    ax.grid(color="#E6E6E6", lw=0.45)
    ax.set_box_aspect(1)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.035, fraction=0.052)
    cbar.set_label(r"$\log_2 R^*$")
    sns.despine(ax=ax)
    save_figure(fig, output.with_name(f"{output.name}__flow_corrected_state_landscape"), dpi)

    selected_genotypes = (
        pd.concat(
            [
                stable.nlargest(int(thresholds["heatmap_genotypes"]), "R_star"),
                stable.nsmallest(4, "R_star"),
            ]
        )
        .sort_values("R_star", ascending=False)["genotype"]
        .drop_duplicates()
        .head(int(thresholds["heatmap_genotypes"]))
        .tolist()
    )
    heat = (
        stable[stable["genotype"].isin(selected_genotypes)]
        .pivot_table(index="genotype", columns="stage", values="log2_R_star", aggfunc="first")
        .reindex(selected_genotypes)
        .reindex(columns=["primary", "metastatic"])
    )
    heat.index = [compact_genotype(genotype) for genotype in heat.index]
    fig, ax = plt.subplots(figsize=(3.35, 3.35))
    fig.subplots_adjust(left=0.38, right=0.84, bottom=0.13, top=0.95)
    sns.heatmap(
        heat,
        ax=ax,
        cmap="coolwarm",
        center=0,
        vmin=-r_limit,
        vmax=r_limit,
        linewidths=0.5,
        linecolor="white",
        annot=True,
        fmt=".1f",
        annot_kws={"fontsize": 5.3},
        cbar_kws={"label": r"$\log_2 R^*$", "fraction": 0.06, "pad": 0.03},
        mask=heat.isna(),
    )
    ax.set_xlabel("Compartment")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=0)
    ax.tick_params(axis="y", labelsize=5.5)
    save_figure(fig, output.with_name(f"{output.name}__stage_genotype_map"), dpi)

    ranked = pd.concat(
        [
            stable.nlargest(int(thresholds["bottleneck_display"]), "R_star"),
            stable.nsmallest(int(thresholds["fast_display"]), "R_star"),
        ]
    ).drop_duplicates("state")
    ranked = ranked.sort_values("R_star")
    y = np.arange(len(ranked))
    median = np.log2(ranked["bootstrap_median_R_star"].clip(lower=1e-12))
    low = np.log2(ranked["R_star_ci_low"].clip(lower=1e-12))
    high = np.log2(ranked["R_star_ci_high"].clip(lower=1e-12))
    fig, ax = plt.subplots(figsize=(3.65, 3.65))
    fig.subplots_adjust(left=0.44, right=0.96, bottom=0.14, top=0.96)
    ax.errorbar(
        median,
        y,
        xerr=np.vstack([median - low, high - median]),
        fmt="o",
        color="#333333",
        ecolor="#777777",
        elinewidth=0.8,
        capsize=2,
        markersize=3.8,
    )
    direction_colors = np.where(ranked["R_star"] >= 1, "#B24773", "#4C78A8")
    ax.scatter(
        np.log2(ranked["R_star"]),
        y,
        color=direction_colors,
        s=28,
        edgecolor="#333333",
        linewidth=0.35,
        zorder=3,
    )
    ax.axvline(0, color="#777777", ls="--", lw=0.7)
    ax.set_yticks(y, labels=[compact_state(state) for state in ranked["state"]])
    ax.tick_params(axis="y", labelsize=5.2)
    ax.set_xlabel(r"$\log_2 R^*$ with bootstrap 95% CI")
    ax.grid(axis="x", color="#E6E6E6", lw=0.45)
    sns.despine(ax=ax)
    save_figure(fig, output.with_name(f"{output.name}__bottleneck_and_fast_passing_states"), dpi)

    fig, ax = plt.subplots(figsize=(3.25, 3.25))
    fig.subplots_adjust(left=0.17, right=0.84, bottom=0.17, top=0.96)
    scatter_o = ax.scatter(
        stable["Lhat_progression"],
        stable["L_v"],
        c=stable["log2_O_star"],
        s=np.clip(12 + 5 * np.sqrt(stable["N_v"]), 16, 80),
        cmap="PiYG_r",
        norm=o_norm,
        edgecolor="#333333",
        linewidth=0.35,
        alpha=0.78,
    )
    limits_o = [
        min(stable["Lhat_progression"].min(), stable["L_v"].min()) * 0.60,
        max(stable["Lhat_progression"].max(), stable["L_v"].max()) * 1.65,
    ]
    ax.plot(limits_o, limits_o, color="#777777", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(limits_o)
    ax.set_ylim(limits_o)
    ax.set_xlabel(r"Progression-only expected occupancy $\hat{L}_v$")
    ax.set_ylabel(r"Observed occupancy $L_v$")
    ax.grid(color="#E6E6E6", lw=0.45)
    ax.set_box_aspect(1)
    cbar = fig.colorbar(scatter_o, ax=ax, pad=0.035, fraction=0.052)
    cbar.set_label(r"$\log_2 O^*$")
    sns.despine(ax=ax)
    save_figure(fig, output.with_name(f"{output.name}__observation_enrichment_residual"), dpi)


def plot_combined_single(
    datasets: list[str], config: dict, result_root: Path
) -> None:
    base = result_root / "combined_figures" / "Figure_E5_core_results_three_cohorts"
    cleanup_figure_outputs(base)
    dpi = int(config["plot"]["dpi"])
    for dataset in datasets:
        scores = pd.read_csv(result_root / dataset / "tables" / "state_scores.tsv", sep="\t")
        stable = scores[scores["eligible_experiment5"]].copy()
        fig, axis = plt.subplots(figsize=(3.25, 3.25))
        fig.subplots_adjust(left=0.17, right=0.84, bottom=0.17, top=0.96)
        scatter = axis.scatter(
            stable["log2_R_star"],
            stable["log2_O_star"],
            s=np.clip(12 + 5 * np.sqrt(stable["N_v"]), 16, 70),
            c=stable["stability_high_confidence"],
            cmap=sns.light_palette("#006D77", as_cmap=True),
            vmin=0,
            vmax=1,
            edgecolor=np.where(stable["stage"].eq("metastatic"), "#CC6677", "#333333"),
            linewidth=0.55,
            alpha=0.76,
        )
        axis.axhline(0, color="#888888", lw=0.7)
        axis.axvline(0, color="#888888", lw=0.7)
        axis.set_xlabel(r"$\log_2 R^*$")
        axis.set_ylabel(r"$\log_2 O^*$")
        high_confidence = stable[stable["high_confidence"]].copy()
        high_confidence["distance"] = np.sqrt(
            high_confidence["log2_R_star"] ** 2
            + high_confidence["log2_O_star"] ** 2
        )
        extreme = high_confidence.nlargest(1, "distance")
        for _, row in extreme.iterrows():
            axis.annotate(
                compact_state(row["state"], 2),
                (row["log2_R_star"], row["log2_O_star"]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=5.2,
            )
        axis.margins(x=0.16, y=0.16)
        axis.grid(color="#E6E6E6", lw=0.45)
        axis.set_box_aspect(1)
        cbar = fig.colorbar(scatter, ax=axis, pad=0.035, fraction=0.052)
        cbar.set_label("Bootstrap Top-10 stability (N>=10)")
        sns.despine(ax=axis)
        slug = config["datasets"][dataset]["display_name"].lower().replace(" ", "_")
        save_figure(fig, base.with_name(f"{base.name}__{slug}"), dpi)


def write_dataset_report(
    dataset: str, scores: pd.DataFrame, metrics: dict, path: Path
) -> None:
    stable = scores[scores["eligible_experiment5"]]
    top = stable.nlargest(5, "R_star")
    fast = stable.nsmallest(5, "R_star")
    enriched = stable.nlargest(5, "O_star")
    lines = [
        f"# Experiment 5 Report: {dataset}",
        "",
        "R* is a relative dwell index, not an absolute time. O* is an auxiliary",
        "progression-only residual, not a diagnosis or observation probability.",
        "",
        "## Metrics",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in metrics.items())
    lines.extend(["", "## Top Relative Bottlenecks", ""])
    lines.extend(
        f"- {row.state}: R*={row.R_star:.3g}, N={row.N_v}, stability={row.stability:.2f}."
        for row in top.itertuples()
    )
    lines.extend(["", "## Fast-passing States", ""])
    lines.extend(
        f"- {row.state}: R*={row.R_star:.3g}, N={row.N_v}, stability={row.stability:.2f}."
        for row in fast.itertuples()
    )
    lines.extend(["", "## High Observation-enrichment Residuals", ""])
    lines.extend(
        f"- {row.state}: O*={row.O_star:.3g}, R*={row.R_star:.3g}."
        for row in enriched.itertuples()
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(result_root: Path, records: list[dict]) -> None:
    summary = pd.DataFrame(records)
    summary.to_csv(result_root / "experiment_05_summary.csv", index=False)
    lines = [
        "# Experiment 5 Summary",
        "",
        "| " + " | ".join(summary.columns) + " |",
        "| " + " | ".join(["---"] * len(summary.columns)) + " |",
    ]
    for _, row in summary.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in summary.columns) + " |")
    (result_root / "experiment_05_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_protocol_audit(result_root: Path, config: dict) -> None:
    text = f"""# Experiment 5 Protocol Audit

## Locked primary definition

- Eligible states: N_v >= {config['thresholds']['minimum_state_count']} and
  F_hat_v >= {config['thresholds']['minimum_inflow']}.
- R_v = L_v / (F_hat_v + epsilon), epsilon =
  {config['thresholds']['epsilon']}.
- R_star = R_v / median(R_v) across eligible states.
- R_star is relative and has no calendar-time unit.

## Progression-only O_star boundary

The source protocol defines O_star = L_v / Lhat_progression_v but does not
uniquely specify real-cohort observation times or stage-transition intensity.
The preregistered implementation therefore:

1. Simulates monotone cMHN event accumulation from WT.
2. Samples the terminal event burden from the empirical analyzed cohort.
3. Uses the Experiment 4 stage-bridge mass
   ({config['progression_only']['stage_transition_mass']}) for primary-to-
   metastatic progression.
4. Conditions the smoothed expected occupancy on observed state support.
5. Repeats the analysis at stage masses
   {config['progression_only']['stage_transition_mass_sensitivity']}.

O_star is auxiliary and cannot be interpreted as a diagnosis rate.

## Bootstrap boundary

Patient-equivalent multinomial resampling is performed
{config['bootstrap']['replicates']} times. Occupancy, inflow, and R_star are
recomputed with the fitted MHN backbone held fixed. Full MHN refitting is
reserved for the later uncertainty experiment specified by the protocol.

## Figure-design review

The visual evidence chain adopts practices from:

- SCENIC+ multiomic benchmarking, Nature Methods 2023:
  https://www.nature.com/articles/s41592-023-01938-4
- Atlas-level integration benchmarking, Nature Methods 2022:
  https://www.nature.com/articles/s41592-021-01336-8
- Tumor subclonal reconstruction benchmarking, Nature Biotechnology 2024:
  https://www.nature.com/articles/s41587-024-02250-y
- Tumor evolution metrics, Nature Cancer 2024:
  https://www.nature.com/articles/s43018-024-00787-0

Adopted practices are aligned four-panel evidence, ordered state heatmaps,
color-size encoding with explicit scales, residual plots with identity lines,
and compact uncertainty displays. The per-cohort main figure now retains only
A-D: R* construction, stage-genotype R* mapping, state-level R* intervals, and
the O* residual diagnostic. The former R*-O* taxonomy panel is represented by
the four-cohort combined figure, and the former ranking-stability panel is kept
as table-level audit evidence rather than a main-figure panel.
"""
    (result_root / "experiment_05_protocol_audit.md").write_text(
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
    logging.basicConfig(
        filename=result_root / "experiment_05.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    records = []
    for dataset in datasets:
        print(f"[Experiment 5] Computing {dataset}...", flush=True)
        records.append(compute_dataset(dataset, config, result_root))
        print(f"[Experiment 5] {dataset} complete.", flush=True)
    write_summary(result_root, records)
    write_protocol_audit(result_root, config)
    plot_combined_single(datasets, config, result_root)


if __name__ == "__main__":
    main()

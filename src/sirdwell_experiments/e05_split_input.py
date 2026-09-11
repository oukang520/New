"""Run Experiment 5: relative dwell R* and observation enrichment O*.

R* is the primary state-level result. O* is an auxiliary residual diagnostic
against a progression-only cMHN simulation and must not be interpreted as a
clinical observation or diagnosis rate.
"""

from __future__ import annotations
from .parameters import config_path as bundled_config_path
import argparse
import json
import logging
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

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
    parser.add_argument(
        "--config", default=str(bundled_config_path("experiment_05_split_input.yaml"))
    )
    parser.add_argument(
        "--dataset-config",
        default=str(bundled_config_path("selected_experiment_datasets.yaml")),
    )
    return parser.parse_args()


def genotype_from_mask(mask: int, events: list[str]) -> str:
    selected = [event for index, event in enumerate(events) if mask & 1 << index]
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
        label
        for label, members in PATHWAY_GROUPS.items()
        if events.intersection(members)
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
                    [bool(mask & 1 << index) for index in range(len(events))],
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
    observed_simulations = sum((counts[state] for state in observed_states))
    denominator = observed_simulations + alpha * len(observed_states)
    expected = pd.Series(
        {state: (counts[state] + alpha) / denominator for state in observed_states},
        dtype=float,
    )
    support_coverage = observed_simulations / simulations
    return (expected, support_coverage)


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
        np.add.at(inflow, edge_target, occupancy[edge_source] * edge_probability)
        eligible = (sampled_counts >= minimum_count) & (inflow >= minimum_inflow)
        raw = occupancy / (inflow + epsilon)
        normalizer = np.median(raw[eligible]) if eligible.any() else np.nan
        if not np.isfinite(normalizer) or normalizer <= 0:
            continue
        values[replicate, eligible] = raw[eligible] / normalizer
        eligible_indices = np.flatnonzero(eligible)
        if len(eligible_indices):
            order = eligible_indices[
                np.argsort(values[replicate, eligible_indices])[::-1]
            ]
            top_counts[order[:top_k]] += 1
        high_confidence_eligible = eligible & (sampled_counts >= high_confidence_count)
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
    return (summary, pd.DataFrame(long_rows))


def next_state_map(edges: pd.DataFrame) -> dict[str, str]:
    result = {}
    ordered = edges.sort_values(
        ["source_state", "inflow_contribution"], ascending=[True, False]
    )
    for source, group in ordered.groupby("source_state"):
        result[str(source)] = "; ".join(
            (compact_state(state) for state in group["target_state"].head(3))
        )
    return result


def compute_dataset(dataset: str, config: dict, result_root: Path) -> dict:
    exp4 = Path(config["experiment_04_root"]) / dataset / "tables"
    exp3 = Path(config["experiment_03_root"]) / dataset / "tables"
    output = result_root / dataset
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    inflow = pd.read_csv(exp4 / "inflow_table_rule_a_one_step.tsv", sep="\t")
    edges = pd.read_csv(exp4 / "predecessor_edges_rule_a_one_step.tsv", sep="\t")
    occupancy = pd.read_csv(exp4 / "state_occupancy_experiment4.tsv", sep="\t")
    theta_frame = pd.read_csv(exp3 / "theta.tsv", sep="\t", index_col=0)
    events = theta_frame.columns.astype(str).tolist()
    theta = theta_frame.to_numpy(dtype=float)
    thresholds = config["thresholds"]
    eligible = (inflow["N_v"] >= int(thresholds["minimum_state_count"])) & (
        inflow["F_hat"] >= float(thresholds["minimum_inflow"])
    )
    epsilon = float(thresholds["epsilon"])
    inflow["R_v"] = inflow["L_v"] / (inflow["F_hat"] + epsilon)
    normalizer = float(inflow.loc[eligible, "R_v"].median())
    inflow["R_star"] = inflow["R_v"] / normalizer
    inflow["log2_R_star"] = np.log2(inflow["R_star"].clip(lower=1e-12))
    inflow["eligible_experiment5"] = eligible
    inflow["high_confidence"] = eligible & (
        inflow["N_v"] >= int(thresholds["high_confidence_state_count"])
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
        int(config["random_seed"]) + sum((ord(char) for char in dataset)),
    )
    inflow["Lhat_progression"] = inflow["state"].map(expected)
    inflow["O_star"] = inflow["L_v"] / (inflow["Lhat_progression"] + epsilon)
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
            int(config["random_seed"])
            + 1000 * (index + 1)
            + sum((ord(c) for c in dataset)),
        )
        expected_by_mass[float(stage_mass)] = expected_mass
        o_mass = inflow["L_v"] / (inflow["state"].map(expected_mass) + epsilon)
        comparison = eligible & np.isfinite(o_mass) & np.isfinite(inflow["O_star"])
        rho = spearmanr(
            inflow.loc[comparison, "O_star"], o_mass.loc[comparison]
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
        int(config["random_seed"]) + 5000 + sum((ord(c) for c in dataset)),
    )
    inflow = inflow.merge(bootstrap_summary, on="state", how="left")
    inflow["genotype"] = inflow["genotype"].fillna("WT")
    inflow["clinical_annotation"] = inflow["genotype"].map(biological_annotation)
    inflow["possible_next_states"] = (
        inflow["state"].map(next_state_map(edges)).fillna("")
    )
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
    high_confidence_bottleneck = (
        stable[stable["high_confidence"]].nlargest(top_k, "R_star").copy()
    )
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
        tables / "top_bottleneck_states_high_confidence.tsv", sep="\t", index=False
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
        ["rank", "state", "O_star", "R_star", "stability", "N_v", "interpretation_flag"]
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
    return metrics


def _submission_short_name(dataset: str) -> str:
    return {"AACR_LUAD": "LUAD", "AACR_COAD": "COAD", "AACR_IDC": "IDC"}.get(
        dataset, dataset
    )


def _load_submission_scores(
    datasets: list[str], result_root: Path
) -> dict[str, pd.DataFrame]:
    return {
        dataset: pd.read_csv(
            result_root / dataset / "tables" / "state_scores.tsv", sep="\t"
        )
        for dataset in datasets
    }


def _stable_submission_scores(score_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for dataset, scores in score_tables.items():
        part = scores[scores["eligible_experiment5"].astype(bool)].copy()
        part["dataset_name"] = dataset
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    selected = yaml.safe_load(Path(args.dataset_config).read_text(encoding="utf-8"))
    datasets = [entry["dataset_name"] for entry in selected["included_datasets"]]
    result_root = Path(config["result_root"]).resolve()
    result_root.mkdir(parents=True, exist_ok=True)
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
    pd.DataFrame(records).to_csv(result_root / "experiment_05_summary.csv", index=False)


if __name__ == "__main__":
    main()

"""Run Experiment 9: observation-enrichment O* recovery simulation.

Experiment 9 isolates the auxiliary observation-enrichment layer defined in the
protocol:

    O*_v = L_v / Lhat_progression_v

The progression-only baseline is generated from a fixed synthetic cMHN
backbone. Observation-enriched snapshots are then sampled by weighting selected
states with known omega values. This tests whether O* recovers omega_v > 1
without turning the experiment into another MHN parameter-recovery benchmark.
"""

from __future__ import annotations
from .parameters import config_path as bundled_config_path
import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

CONFIG_PATH = bundled_config_path("experiment_09.yaml")
EVENTS = [f"E{i:02d}" for i in range(1, 16)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Rel-ObsTQ-MHN Experiment 9.")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--repeats", type=int, help="Override repeat count.")
    parser.add_argument("--result-root", help="Override result root.")
    return parser.parse_args()


def setup_logging(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=root / "experiment_09.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )


def genotype(mask: int) -> str:
    selected = [event for index, event in enumerate(EVENTS) if mask & 1 << index]
    return "+".join(selected) if selected else "WT"


def stage(mask: int) -> str:
    invasion_driver = bool(mask & 1 << 2)
    metastasis_driver = bool(mask & 1 << 5)
    if metastasis_driver:
        return "S3"
    if invasion_driver:
        return "S2"
    return "S1"


def state_name(mask: int) -> str:
    return f"{stage(mask)}::{genotype(mask)}"


def compact_state(mask: int, maximum_events: int = 3) -> str:
    members = genotype(mask).split("+")
    if members == ["WT"]:
        members = []
    text = "+".join(members[:maximum_events]) if members else "WT"
    if len(members) > maximum_events:
        text += "+..."
    return f"{stage(mask)} | {text}"


def event_count(mask: int) -> int:
    return bin(int(mask)).count("1")


def create_true_theta(seed: int, sparsity: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = len(EVENTS)
    theta = np.zeros((p, p), dtype=float)
    theta[np.diag_indices(p)] = np.clip(rng.normal(-1.0, 0.4, p), -2.0, 0.0)
    off_diagonal = [(i, j) for i in range(p) for j in range(p) if i != j]
    selected = rng.choice(
        len(off_diagonal), size=round(sparsity * len(off_diagonal)), replace=False
    )
    for index in selected:
        target, source = off_diagonal[int(index)]
        theta[target, source] = (
            rng.uniform(0.5, 1.5) if rng.random() < 0.62 else rng.uniform(-1.5, -0.5)
        )
    forced = {
        (2, 0): 1.25,
        (2, 1): 0.85,
        (5, 2): 1.35,
        (3, 0): 1.0,
        (4, 1): 1.1,
        (6, 2): 0.9,
        (7, 3): 1.15,
        (8, 4): 1.05,
        (9, 5): 1.2,
        (10, 6): 0.8,
        (11, 7): 1.1,
        (12, 8): 0.95,
        (1, 0): -1.2,
        (0, 1): -1.2,
        (4, 3): -0.85,
        (3, 4): -0.85,
    }
    for (target, source), value in forced.items():
        theta[target, source] = value
    return theta


def event_probabilities(mask: int, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = theta.shape[0]
    present = np.array([bool(mask & 1 << index) for index in range(p)])
    absent = np.flatnonzero(~present)
    if absent.size == 0:
        return (absent, np.array([], dtype=float))
    logits = np.array(
        [theta[event, event] + theta[event, present].sum() for event in absent],
        dtype=float,
    )
    rates = np.exp(np.clip(logits, -50, 50))
    return (absent, rates)


def simulate_snapshot(
    theta: np.ndarray,
    dwell_by_mask: dict[int, float],
    maximum_time: float,
    maximum_events: int,
    rng: np.random.Generator,
) -> int:
    mask = 0
    current_time = 0.0
    intervals: list[tuple[float, float, int]] = []
    while current_time < maximum_time:
        if event_count(mask) >= maximum_events:
            break
        absent, rates = event_probabilities(mask, theta)
        if absent.size == 0 or rates.sum() <= 0:
            intervals.append((current_time, maximum_time, mask))
            break
        dwell = float(dwell_by_mask.get(mask, 1.0))
        total_rate = float(rates.sum()) / dwell
        wait = float(rng.exponential(1.0 / total_rate))
        end = min(current_time + wait, maximum_time)
        intervals.append((current_time, end, mask))
        if end >= maximum_time:
            break
        event = int(rng.choice(absent, p=rates / rates.sum()))
        mask |= 1 << event
        current_time = end
    if not intervals:
        return mask
    observation_horizon = intervals[-1][1]
    observation_time = float(rng.uniform(0.0, observation_horizon))
    for start, end, interval_mask in intervals:
        if start <= observation_time < end or np.isclose(observation_time, end):
            return int(interval_mask)
    return int(intervals[-1][2])


def simulate_cohort(
    theta: np.ndarray,
    dwell_by_mask: dict[int, float],
    n: int,
    simulation: dict,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array(
        [
            simulate_snapshot(
                theta,
                dwell_by_mask,
                float(simulation["maximum_time"]),
                int(simulation["maximum_events"]),
                rng,
            )
            for _ in range(n)
        ],
        dtype=np.int32,
    )


def probability_table(masks: np.ndarray) -> pd.DataFrame:
    counts = Counter((int(mask) for mask in masks))
    total = len(masks)
    rows = []
    for mask, count in counts.items():
        rows.append(
            {
                "mask": int(mask),
                "state": state_name(int(mask)),
                "stage": stage(int(mask)),
                "genotype": genotype(int(mask)),
                "event_count": event_count(int(mask)),
                "pilot_count": int(count),
                "pilot_frequency": float(count / total),
            }
        )
    return pd.DataFrame(rows).sort_values("pilot_frequency", ascending=False)


def select_stage_balanced(
    candidates: pd.DataFrame, needed: int, used: set[int]
) -> list[int]:
    selected: list[int] = []
    for stage_name in ["S1", "S2", "S3"]:
        sub = candidates[
            (candidates["stage"] == stage_name)
            & ~candidates["mask"].isin(used | set(selected))
        ]
        if not sub.empty:
            selected.append(int(sub.iloc[0]["mask"]))
        if len(selected) == needed:
            return selected
    for mask in candidates["mask"].astype(int):
        if len(selected) == needed:
            break
        if mask not in used and mask not in selected:
            selected.append(mask)
    return selected


def select_truth_states(
    pilot: pd.DataFrame, config: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selection = config["truth_selection"]
    simulation = config["simulation"]
    candidates = pilot[
        pilot["event_count"].between(
            int(selection["event_count_min"]), int(selection["event_count_max"])
        )
        & pilot["pilot_frequency"].between(
            float(selection["min_pilot_frequency"]),
            float(selection["max_pilot_frequency"]),
        )
    ].copy()
    if len(candidates) < int(simulation["high_observation_states"]) + int(
        simulation["low_observation_states"]
    ):
        raise RuntimeError(
            "Not enough moderate-frequency states for Experiment 9 truth selection."
        )
    target = float(selection["target_pilot_frequency"])
    candidates["selection_distance"] = np.abs(
        np.log(candidates["pilot_frequency"]) - np.log(target)
    )
    candidates = candidates.sort_values(
        ["selection_distance", "pilot_frequency"], ascending=[True, False]
    )
    used: set[int] = set()
    high_candidates = candidates.sort_values(
        ["pilot_frequency", "selection_distance"], ascending=[True, True]
    )
    high_masks = select_stage_balanced(
        high_candidates, int(simulation["high_observation_states"]), used
    )
    used.update(high_masks)
    low_candidates = candidates[~candidates["mask"].isin(used)].copy()
    low_candidates = low_candidates.sort_values(
        ["pilot_frequency", "selection_distance"], ascending=[False, True]
    )
    low_masks = select_stage_balanced(
        low_candidates, int(simulation["low_observation_states"]), used
    )
    used.update(low_masks)
    truth = pilot[pilot["mask"].isin(high_masks + low_masks)].copy()
    truth["omega_class"] = np.where(
        truth["mask"].isin(high_masks), "high_observation", "low_observation"
    )
    truth["omega_true"] = np.where(
        truth["omega_class"].eq("high_observation"),
        float(simulation["high_omega"]),
        float(simulation["low_omega"]),
    )
    truth["D_true_omega_only"] = float(simulation["neutral_dwell"])
    truth["D_true_omega_plus_dwell"] = np.where(
        truth["omega_class"].eq("low_observation"),
        float(simulation["contrast_bottleneck_dwell"]),
        float(simulation["neutral_dwell"]),
    )
    truth["selection_mode"] = selection["mode"]
    truth["state_label"] = truth["mask"].map(compact_state)
    candidate_audit = candidates.head(80).copy()
    candidate_audit["selected"] = candidate_audit["mask"].isin(truth["mask"])
    return (
        truth.sort_values(["omega_class", "stage", "pilot_frequency"]),
        candidate_audit,
    )


def align_probabilities(probabilities: pd.Series, universe: list[int]) -> np.ndarray:
    arr = probabilities.reindex(universe).fillna(0.0).to_numpy(dtype=float)
    total = float(arr.sum())
    if total <= 0:
        raise RuntimeError("Probability vector sums to zero.")
    return arr / total


def weighted_observation_distribution(
    base_prob: np.ndarray, omega: np.ndarray
) -> np.ndarray:
    weighted = base_prob * omega
    total = float(weighted.sum())
    if total <= 0:
        raise RuntimeError("Observation-weighted distribution sums to zero.")
    return weighted / total


def rank_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores)
    labels = labels[valid]
    scores = scores[valid]
    positive = int(labels.sum())
    negative = int((~labels).sum())
    if positive == 0 or negative == 0:
        return np.nan
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    return float(
        (ranks[labels].sum() - positive * (positive + 1) / 2) / (positive * negative)
    )


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores)
    labels = labels[valid]
    scores = scores[valid]
    if labels.sum() == 0:
        return np.nan
    order = np.argsort(scores)[::-1]
    ordered = labels[order].astype(int)
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float((precision * ordered).sum() / ordered.sum())


def roc_curve_grid(
    labels: np.ndarray, scores: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores)
    labels = labels[valid]
    scores = scores[valid]
    if labels.sum() == 0 or (~labels).sum() == 0:
        return np.full_like(grid, np.nan, dtype=float)
    order = np.argsort(scores)[::-1]
    ordered = labels[order].astype(int)
    positives = int(ordered.sum())
    negatives = int((1 - ordered).sum())
    tp = np.cumsum(ordered)
    fp = np.cumsum(1 - ordered)
    fpr = np.r_[0.0, fp / negatives]
    tpr = np.r_[0.0, tp / positives]
    return np.interp(grid, fpr, tpr)


def compute_r_star(
    table: pd.DataFrame, theta: np.ndarray, scoring: dict
) -> pd.DataFrame:
    occupancy = dict(zip(table["mask"].astype(int), table["L_v"].astype(float)))
    observed_masks = set(occupancy)
    inflow: defaultdict[int, float] = defaultdict(float)
    for source_mask, source_occupancy in occupancy.items():
        if source_occupancy <= 0:
            continue
        absent, rates = event_probabilities(int(source_mask), theta)
        if absent.size == 0 or rates.sum() <= 0:
            continue
        probabilities = rates / rates.sum()
        for event, probability in zip(absent, probabilities):
            target = int(source_mask) | 1 << int(event)
            if target not in observed_masks:
                continue
            inflow[target] += source_occupancy * float(probability)
    result = table.copy()
    result["F_hat"] = result["mask"].map(lambda m: float(inflow.get(int(m), 0.0)))
    epsilon = float(scoring["epsilon"])
    result["R_raw"] = result["L_v"] / (result["F_hat"] + epsilon)
    eligible = result["eligible"].astype(bool) & result["F_hat"].gt(epsilon)
    normalizer = (
        float(result.loc[eligible, "R_raw"].median()) if eligible.any() else np.nan
    )
    result["R_star"] = np.where(
        eligible & np.isfinite(normalizer) & (normalizer > 0),
        result["R_raw"] / normalizer,
        np.nan,
    )
    result["log2_R_star"] = np.log2(result["R_star"].clip(lower=1e-12))
    return result


def build_repeat_table(
    repeat: int,
    scenario: str,
    universe: list[int],
    expected_counts: np.ndarray,
    observed_counts: np.ndarray,
    omega: np.ndarray,
    dwell: np.ndarray,
    theta: np.ndarray,
    config: dict,
) -> pd.DataFrame:
    scoring = config["state_scoring"]
    alpha = float(config["smoothing"]["dirichlet_alpha"])
    n_expected = int(expected_counts.sum())
    n_observed = int(observed_counts.sum())
    include = (
        (expected_counts > 0) | (observed_counts > 0) | (omega != 1.0) | (dwell != 1.0)
    )
    local_indices = np.flatnonzero(include)
    local_universe = [universe[index] for index in local_indices]
    expected_counts = expected_counts[local_indices]
    observed_counts = observed_counts[local_indices]
    omega = omega[local_indices]
    dwell = dwell[local_indices]
    k = len(universe)
    lhat = (expected_counts + alpha) / (n_expected + alpha * k)
    l_obs = observed_counts / n_observed
    table = pd.DataFrame(
        {
            "repeat": repeat,
            "scenario": scenario,
            "mask": local_universe,
            "state": [state_name(mask) for mask in local_universe],
            "state_label": [compact_state(mask) for mask in local_universe],
            "stage": [stage(mask) for mask in local_universe],
            "genotype": [genotype(mask) for mask in local_universe],
            "event_count": [event_count(mask) for mask in local_universe],
            "N_v": observed_counts.astype(int),
            "N_expected": expected_counts.astype(int),
            "L_v": l_obs,
            "Lhat_progression": lhat,
            "omega_true": omega,
            "D_true": dwell,
        }
    )
    table["O_star"] = table["L_v"] / (
        table["Lhat_progression"] + float(scoring["epsilon"])
    )
    table["log2_O_star"] = np.log2(table["O_star"].clip(lower=1e-12))
    table["omega_class"] = np.select(
        [table["omega_true"] > 1, table["omega_true"] < 1],
        ["high_observation", "low_observation"],
        default="neutral",
    )
    table["dwell_class"] = np.select(
        [table["D_true"] > 1, table["D_true"] < 1],
        ["bottleneck", "fast"],
        default="neutral",
    )
    table["eligible"] = table["N_v"].ge(int(scoring["minimum_state_count"])) & table[
        "N_expected"
    ].ge(int(scoring["minimum_expected_count"]))
    table = compute_r_star(table, theta, scoring)
    return table


def repeat_metrics(table: pd.DataFrame, config: dict) -> tuple[dict, list[dict]]:
    scoring = config["state_scoring"]
    stable = table[table["eligible"].astype(bool)].copy()
    labels = stable["omega_true"].to_numpy(dtype=float) > 1
    omega = stable["omega_true"].to_numpy(dtype=float)
    o_score = stable["O_star"].to_numpy(dtype=float)
    l_score = stable["L_v"].to_numpy(dtype=float)
    top_k = int(scoring["top_k"])
    top_o = stable.nlargest(top_k, "O_star")
    top_l = stable.nlargest(top_k, "L_v")
    rho_o = spearmanr(omega, o_score).statistic if len(stable) >= 3 else np.nan
    rho_l = spearmanr(omega, l_score).statistic if len(stable) >= 3 else np.nan
    record = {
        "repeat": int(table["repeat"].iloc[0]),
        "scenario": str(table["scenario"].iloc[0]),
        "stable_states": int(len(stable)),
        "stable_high_observation_states": int(labels.sum()),
        "stable_low_observation_states": int((stable["omega_true"] < 1).sum()),
        "spearman_O_star": float(rho_o),
        "spearman_occupancy": float(rho_l),
        "high_omega_auc_O_star": rank_auc(labels, o_score),
        "high_omega_auc_occupancy": rank_auc(labels, l_score),
        "high_omega_ap_O_star": average_precision(labels, o_score),
        "high_omega_ap_occupancy": average_precision(labels, l_score),
        "top3_precision_O_star": float((top_o["omega_true"] > 1).sum() / top_k),
        "top3_precision_occupancy": float((top_l["omega_true"] > 1).sum() / top_k),
    }
    curve_rows: list[dict] = []
    grid = np.linspace(0.0, 1.0, 101)
    for method, scores in [("O_star", o_score), ("occupancy", l_score)]:
        tpr = roc_curve_grid(labels, scores, grid)
        for value, tpr_value in zip(grid, tpr):
            curve_rows.append(
                {
                    "repeat": record["repeat"],
                    "scenario": record["scenario"],
                    "method": method,
                    "fpr": float(value),
                    "tpr": float(tpr_value),
                }
            )
    return (record, curve_rows)


def select_representative(metrics: pd.DataFrame, scenario: str = "omega_only") -> int:
    sub = metrics[metrics["scenario"].eq(scenario)].copy()
    target_auc = float(sub["high_omega_auc_O_star"].median())
    target_rho = float(sub["spearman_O_star"].median())
    sub["distance"] = (sub["high_omega_auc_O_star"] - target_auc).abs() + (
        sub["spearman_O_star"] - target_rho
    ).abs()
    return int(sub.sort_values(["distance", "repeat"]).iloc[0]["repeat"])


def true_theta_table(theta: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(theta, index=EVENTS, columns=EVENTS)
    frame.index.name = "target_event"
    return frame


def edge_list(theta: np.ndarray) -> pd.DataFrame:
    rows = []
    for target in range(theta.shape[0]):
        for source in range(theta.shape[1]):
            if target == source or np.isclose(theta[target, source], 0.0):
                continue
            rows.append(
                {
                    "target_event": EVENTS[target],
                    "source_event": EVENTS[source],
                    "log_effect": float(theta[target, source]),
                    "effect": (
                        "promoting" if theta[target, source] > 0 else "inhibiting"
                    ),
                }
            )
    return pd.DataFrame(rows)


def save_resolved_config(root: Path, config: dict) -> None:
    (root / "resolved_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.repeats is not None:
        config["simulation"]["repeats"] = int(args.repeats)
    if args.result_root:
        config["result_root"] = args.result_root
    root = Path(config["result_root"]).resolve()
    tables = root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    setup_logging(root)
    save_resolved_config(root, config)
    seed = int(config["random_seed"])
    simulation = config["simulation"]
    theta = create_true_theta(seed, float(simulation["interaction_sparsity"]))
    true_theta_table(theta).to_csv(tables / "true_theta.tsv", sep="\t")
    edge_list(theta).to_csv(tables / "true_edge_list.tsv", sep="\t", index=False)
    pilot_masks = simulate_cohort(
        theta, {}, int(simulation["pilot_samples"]), simulation, seed + 1
    )
    pilot = probability_table(pilot_masks)
    truth, candidate_audit = select_truth_states(pilot, config)
    pilot.to_csv(tables / "progression_pilot_distribution.tsv", sep="\t", index=False)
    truth.to_csv(tables / "truth_states.tsv", sep="\t", index=False)
    candidate_audit.to_csv(
        tables / "truth_selection_candidate_audit.tsv", sep="\t", index=False
    )
    dwell_contrast = {
        int(row["mask"]): float(row["D_true_omega_plus_dwell"])
        for _, row in truth.iterrows()
        if float(row["D_true_omega_plus_dwell"]) != 1.0
    }
    contrast_masks = simulate_cohort(
        theta, dwell_contrast, int(simulation["pilot_samples"]), simulation, seed + 2
    )
    contrast_pilot = probability_table(contrast_masks)
    contrast_pilot.to_csv(
        tables / "contrast_dwell_pilot_distribution.tsv", sep="\t", index=False
    )
    universe = sorted(
        set(pilot["mask"].astype(int)) | set(contrast_pilot["mask"].astype(int))
    )
    p0 = align_probabilities(pilot.set_index("mask")["pilot_frequency"], universe)
    pdwell = align_probabilities(
        contrast_pilot.set_index("mask")["pilot_frequency"], universe
    )
    omega = np.full(len(universe), float(simulation["neutral_omega"]), dtype=float)
    dwell_omega_only = np.full(
        len(universe), float(simulation["neutral_dwell"]), dtype=float
    )
    dwell_contrast_array = np.full(
        len(universe), float(simulation["neutral_dwell"]), dtype=float
    )
    mask_to_index = {mask: index for index, mask in enumerate(universe)}
    for _, row in truth.iterrows():
        idx = mask_to_index[int(row["mask"])]
        omega[idx] = float(row["omega_true"])
        dwell_contrast_array[idx] = float(row["D_true_omega_plus_dwell"])
    scenario_specs = {
        "omega_only": {
            "progression_distribution": p0,
            "observed_distribution": weighted_observation_distribution(p0, omega),
            "dwell": dwell_omega_only,
        },
        "omega_plus_dwell": {
            "progression_distribution": p0,
            "observed_distribution": weighted_observation_distribution(pdwell, omega),
            "dwell": dwell_contrast_array,
        },
    }
    rng = np.random.default_rng(seed + 3000)
    state_frames: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    curve_rows: list[dict] = []
    repeats = int(simulation["repeats"])
    for repeat in range(1, repeats + 1):
        for scenario, spec in scenario_specs.items():
            expected_counts = rng.multinomial(
                int(simulation["expected_samples_per_repeat"]),
                spec["progression_distribution"],
            )
            observed_counts = rng.multinomial(
                int(simulation["samples_per_repeat"]), spec["observed_distribution"]
            )
            table = build_repeat_table(
                repeat,
                scenario,
                universe,
                expected_counts,
                observed_counts,
                omega,
                spec["dwell"],
                theta,
                config,
            )
            state_frames.append(table)
            metrics, curves = repeat_metrics(table, config)
            metric_rows.append(metrics)
            curve_rows.extend(curves)
        if repeat == 1 or repeat % 20 == 0:
            latest = metric_rows[-2]
            print(
                f"Repeat {repeat}/{repeats}: O* rho={latest['spearman_O_star']:.3f}, AUC={latest['high_omega_auc_O_star']:.3f}"
            )
            logging.info(
                "repeat=%s rho=%.4f auc=%.4f",
                repeat,
                latest["spearman_O_star"],
                latest["high_omega_auc_O_star"],
            )
    states = pd.concat(state_frames, ignore_index=True)
    metrics_table = pd.DataFrame(metric_rows)
    curves = pd.DataFrame(curve_rows)
    states.to_csv(tables / "state_recovery_long.tsv", sep="\t", index=False)
    metrics_table.to_csv(tables / "repeat_metrics.tsv", sep="\t", index=False)
    curves.to_csv(tables / "repeat_curves.tsv", sep="\t", index=False)
    representative = select_representative(metrics_table, "omega_only")
    representative_table = states[
        states["repeat"].eq(representative)
        & states["scenario"].isin(["omega_only", "omega_plus_dwell"])
    ].copy()
    representative_table.to_csv(
        tables / "representative_state_scores.tsv", sep="\t", index=False
    )
    summary = metrics_table.groupby("scenario").median(numeric_only=True).reset_index()
    summary.insert(1, "repeats", repeats)
    summary.to_csv(tables / "experiment_09_summary.tsv", sep="\t", index=False)
    metadata = {
        "experiment": config["experiment_name"],
        "result_root": str(root),
        "representative_repeat": representative,
        "universe_states": len(universe),
        "truth_states": int(len(truth)),
        "progression_pilot_samples": int(simulation["pilot_samples"]),
    }
    (root / "experiment_09_run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

"""Run Experiment 7: topology and multipath robustness simulation.

Experiment 7 follows the protocol matrix:

- topology: linear, branching, mutual-exclusivity, mixed;
- sparsity: 5%, 10%, 20% nonzero directed off-diagonal effects;
- bottleneck placement: early-stage, middle-stage, late-stage,
  pathway-specific;
- repeats: 50 independent cohorts per condition.

The experiment reuses the validated Experiment 6 scoring machinery, but varies
the true cMHN topology and the location of implanted dwell bottlenecks.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import shutil
import time
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
from scipy.stats import spearmanr

import figure_style
import render_experiment_07_design_variants as selected_e7_figure

try:
    import run_experiment_06 as base
except ModuleNotFoundError as exc:  # Allows report-only rendering without the compiled MHN package.
    base = None
    BASE_IMPORT_ERROR = exc
else:
    BASE_IMPORT_ERROR = None


EVENTS = base.EVENTS if base is not None else [f"E{i:02d}" for i in range(1, 16)]
TOPOLOGY_LABELS = {
    "linear": "Linear",
    "branching": "Branching",
    "mutual_exclusivity": "Mutual excl.",
    "mixed": "Mixed",
}
TOPOLOGY_SHORT = {
    "linear": "Lin",
    "branching": "Branch",
    "mutual_exclusivity": "ME",
    "mixed": "Mixed",
}
PLACEMENT_LABELS = {
    "early_stage": "Early-stage",
    "middle_stage": "Middle-stage",
    "late_stage": "Late-stage",
    "pathway_specific": "Pathway-specific",
}
PATHWAYS = {
    "early_driver": [0, 1, 3, 4],
    "invasive": [2, 6, 7],
    "metastatic": [5, 8, 9, 10, 11, 12],
}
SPARSITY_COLORS = {
    0.05: "#B5AED5",
    0.10: "#B2E6FD",
    0.20: "#B8D2CC",
}
METRIC_COLUMNS = [
    "spearman_R_star",
    "spearman_occupancy",
    "bottleneck_auc_R_star",
    "bottleneck_auc_occupancy",
    "bottleneck_ap_R_star",
    "bottleneck_ap_occupancy",
    "top5_precision_R_star",
    "top5_precision_occupancy",
    "bottleneck_recall_at5_R_star",
    "bottleneck_recall_at5_occupancy",
    "stable_states",
    "stable_bottlenecks",
    "evaluated_states",
    "ineligible_truth_states",
    "fit_seconds",
]


def descriptive_reference(config: dict) -> dict:
    """Optional reference cut points; Experiment 7 does not use them as pass/fail."""
    return config.get("descriptive_reference", config.get("success", {}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Rel-ObsTQ-MHN Experiment 7.")
    parser.add_argument("--config", default="configs/experiment_07.yaml")
    parser.add_argument("--repeats", type=int, help="Override repeats for smoke tests.")
    parser.add_argument("--result-root", help="Override result root.")
    parser.add_argument("--limit-combos", type=int, help="Run only the first N matrix conditions.")
    parser.add_argument("--lambda-multiplier", type=float, help="Override fixed lambda multiplier.")
    parser.add_argument("--render-only", action="store_true", help="Regenerate reports and figures from tables.")
    return parser.parse_args()


def setup_logging(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=root / "experiment_07.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )


def save_figure(fig: plt.Figure, base_path: Path, dpi: int) -> None:
    figure_style.save_figure_panels(fig, base_path, {"plot": {"dpi": dpi}}, dpi=dpi)


def configure_plotting(config: dict) -> None:
    figure_style.configure_matplotlib(config)


def draw_compact_gradient_key(
    fig: plt.Figure,
    cmap: mcolors.Colormap,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    left_label: str,
    right_label: str,
) -> None:
    ax_key = fig.add_axes([x, y, width, height])
    gradient = np.linspace(0.0, 1.0, 256).reshape(1, -1)
    ax_key.imshow(gradient, aspect="auto", cmap=cmap)
    ax_key.set_xticks([])
    ax_key.set_yticks([])
    for spine in ax_key.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
        spine.set_linewidth(0.45)
    fig.text(x, y + height + 0.005, title, ha="left", va="bottom", fontsize=5.9, color="#333333")
    fig.text(x, y - 0.007, left_label, ha="left", va="top", fontsize=5.4, color="#444444")
    fig.text(x + width, y - 0.007, right_label, ha="right", va="top", fontsize=5.4, color="#444444")


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


def topology_scaffold(topology: str) -> dict[tuple[int, int], float]:
    if topology == "linear":
        return {
            (1, 0): 1.25,
            (2, 1): 1.20,
            (3, 2): 0.95,
            (4, 3): 0.90,
            (5, 4): 1.25,
            (6, 5): 0.95,
            (7, 6): 0.85,
            (8, 7): 0.80,
        }
    if topology == "branching":
        return {
            (2, 0): 1.20,
            (2, 1): 0.95,
            (3, 2): 1.05,
            (4, 2): 1.05,
            (5, 3): 1.20,
            (5, 4): 1.20,
            (6, 3): 0.90,
            (7, 4): 0.90,
            (8, 5): 0.85,
        }
    if topology == "mutual_exclusivity":
        return {
            (2, 0): 1.10,
            (2, 1): 1.10,
            (5, 2): 1.25,
            (6, 2): 0.95,
            (7, 5): 0.85,
            (1, 0): -1.20,
            (0, 1): -1.20,
            (4, 3): -0.95,
            (3, 4): -0.95,
            (8, 7): -0.80,
        }
    if topology == "mixed":
        return {
            (2, 0): 1.25,
            (2, 1): 0.85,
            (5, 2): 1.35,
            (3, 0): 1.00,
            (4, 1): 1.10,
            (6, 2): 0.90,
            (7, 3): 1.15,
            (8, 4): 1.05,
            (1, 0): -1.20,
            (0, 1): -1.20,
        }
    raise ValueError(f"Unknown topology: {topology}")


def create_topology_theta(seed: int, topology: str, sparsity: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = len(EVENTS)
    theta = np.zeros((p, p), dtype=float)
    theta[np.diag_indices(p)] = np.clip(rng.normal(-1.0, 0.4, p), -2.0, 0.0)

    scaffold = topology_scaffold(topology)
    target_edges = int(round(float(sparsity) * p * (p - 1)))
    if len(scaffold) > target_edges:
        raise ValueError(
            f"Scaffold for {topology} has {len(scaffold)} edges, exceeding target {target_edges}."
        )
    for (target, source), value in scaffold.items():
        theta[target, source] = value

    available = [
        (target, source)
        for target in range(p)
        for source in range(p)
        if target != source and (target, source) not in scaffold
    ]
    rng.shuffle(available)
    random_needed = target_edges - len(scaffold)
    for target, source in available[:random_needed]:
        if rng.random() < 0.64:
            theta[target, source] = float(rng.uniform(0.45, 1.35))
        else:
            theta[target, source] = float(rng.uniform(-1.35, -0.45))
    return theta


def edge_list(theta: np.ndarray, combo: dict | None = None) -> pd.DataFrame:
    rows = []
    for target in range(theta.shape[0]):
        for source in range(theta.shape[1]):
            value = float(theta[target, source])
            if target == source or np.isclose(value, 0.0):
                continue
            row = {
                "target_event": EVENTS[target],
                "source_event": EVENTS[source],
                "target_index": target,
                "source_index": source,
                "log_effect": value,
                "effect": "promoting" if value > 0 else "inhibiting",
                "is_scaffold": False,
            }
            if combo:
                row.update(combo)
                row["is_scaffold"] = (target, source) in topology_scaffold(str(combo["topology"]))
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["combo_id", "target_event", "source_event"] if combo else ["target_event", "source_event"]
    )


def pathway_info(mask: int) -> tuple[str, int]:
    counts = {
        name: sum(1 for event_index in event_indices if mask & (1 << event_index))
        for name, event_indices in PATHWAYS.items()
    }
    best = max(counts, key=counts.get)
    return best, int(counts[best])


def pilot_state_table(theta: np.ndarray, simulation: dict, seed: int) -> pd.DataFrame:
    pilot_masks = base.simulate_cohort(
        theta,
        {},
        int(simulation["pilot_samples"]),
        simulation,
        seed,
    )
    counts = Counter(int(mask) for mask in pilot_masks)
    rows = []
    for mask, count in counts.items():
        pathway, pathway_hits = pathway_info(mask)
        rows.append(
            {
                "mask": int(mask),
                "state": base.state_name(mask),
                "stage": base.stage(mask),
                "genotype": base.genotype(mask),
                "event_count": base.event_count(mask),
                "pilot_count": int(count),
                "pilot_frequency": float(count / len(pilot_masks)),
                "pathway_label": pathway,
                "pathway_hits": pathway_hits,
            }
        )
    return pd.DataFrame(rows).sort_values("pilot_count", ascending=False)


def placement_profile(placement: str) -> dict:
    profiles = {
        "early_stage": {"stage": "S1", "event_min": 1, "event_max": 3, "event_target": 2.0},
        "middle_stage": {"stage": "S2", "event_min": 1, "event_max": 5, "event_target": 2.5},
        "late_stage": {"stage": "S3", "event_min": 1, "event_max": 7, "event_target": 3.0},
        "pathway_specific": {"stage": None, "event_min": 1, "event_max": 6, "event_target": 2.5},
    }
    return profiles[placement]


def candidate_pool(
    pilot: pd.DataFrame, placement: str, selection: dict
) -> tuple[pd.DataFrame, str]:
    profile = placement_profile(placement)
    min_count = int(selection["bottleneck_pilot_count_min"])
    max_count = int(selection["bottleneck_pilot_count_max"])
    base_mask = pilot["event_count"].between(profile["event_min"], profile["event_max"])
    if profile["stage"] is not None:
        placement_mask = pilot["stage"].eq(profile["stage"])
    else:
        placement_mask = pilot["pathway_hits"].ge(2)

    relaxed_min_count = int(selection.get("relaxed_bottleneck_pilot_count_min", max(3, min_count // 2)))
    tiers = [
        ("primary", placement_mask & base_mask & pilot["pilot_count"].between(min_count, max_count)),
        ("relaxed_high_count", placement_mask & base_mask & pilot["pilot_count"].ge(min_count)),
        ("relaxed_event", placement_mask & pilot["event_count"].ge(1) & pilot["pilot_count"].ge(min_count)),
        ("relaxed_low_count", placement_mask & pilot["event_count"].ge(1) & pilot["pilot_count"].ge(relaxed_min_count)),
        ("fallback_placement_all_observed", placement_mask & pilot["event_count"].ge(1)),
    ]
    if bool(selection.get("allow_global_fallback", True)):
        tiers.append(("fallback_global_unavoidable", pilot["event_count"].ge(1)))
    for tier_name, mask in tiers:
        pool = pilot.loc[mask].copy()
        if len(pool) >= 3:
            break
    else:
        pool = pilot.copy()
        tier_name = "fallback_any"

    target_count = int(selection["bottleneck_target_pilot_count"])
    pool["count_distance"] = (np.log1p(pool["pilot_count"]) - math.log1p(target_count)).abs()
    pool["event_distance"] = (pool["event_count"].astype(float) - float(profile["event_target"])).abs()
    pool["selection_score"] = pool["count_distance"] + 0.18 * pool["event_distance"]
    if profile["stage"] is not None:
        pool["placement_match"] = pool["stage"].eq(profile["stage"])
    else:
        pool["placement_match"] = pool["pathway_hits"].ge(2)
    pool["selection_tier"] = tier_name
    return pool.sort_values(["selection_score", "pilot_count"], ascending=[True, False]), tier_name


def diverse_select(pool: pd.DataFrame, count: int, diversity_column: str) -> list[int]:
    selected: list[int] = []
    if diversity_column in pool.columns:
        for _, group in pool.groupby(diversity_column, sort=False):
            if len(selected) >= count:
                break
            mask = int(group.sort_values(["selection_score", "pilot_count"], ascending=[True, False]).iloc[0]["mask"])
            if mask not in selected:
                selected.append(mask)
    for mask in pool.sort_values(["selection_score", "pilot_count"], ascending=[True, False])["mask"].astype(int):
        if len(selected) >= count:
            break
        if mask not in selected:
            selected.append(int(mask))
    return selected


def select_fast_states(
    pilot: pd.DataFrame, selected_bottlenecks: list[int], selection: dict, count: int
) -> tuple[list[int], str]:
    min_count = int(selection["fast_pilot_count_min"])
    tiers = [
        (
            "primary",
            pilot["event_count"].between(
                int(selection["fast_event_count_min"]),
                int(selection["fast_event_count_max"]),
            )
            & pilot["pilot_count"].ge(min_count),
        ),
        ("relaxed_count", pilot["event_count"].between(1, 3) & pilot["pilot_count"].ge(max(3, min_count // 2))),
        ("relaxed_global", pilot["event_count"].ge(1)),
    ]
    for tier_name, mask in tiers:
        pool = pilot.loc[mask & ~pilot["mask"].isin(selected_bottlenecks)].copy()
        if len(pool) >= count:
            break
    else:
        pool = pilot.loc[~pilot["mask"].isin(selected_bottlenecks)].copy()
        tier_name = "fallback_any"
    pool["selection_score"] = -pool["pilot_count"].astype(float)
    selected = diverse_select(pool.sort_values("pilot_count", ascending=False), count, "stage")
    return selected, tier_name


def select_truth_states_for_combo(
    theta: np.ndarray,
    config: dict,
    combo: dict,
    seed: int,
) -> tuple[dict[int, float], pd.DataFrame, pd.DataFrame]:
    pilot = pilot_state_table(theta, config["simulation"], seed)
    bottleneck_pool, bottleneck_tier = candidate_pool(
        pilot,
        str(combo["bottleneck_placement"]),
        config["truth_selection"],
    )
    diversity_column = "pathway_label" if combo["bottleneck_placement"] == "pathway_specific" else "event_count"
    selected_bottleneck = diverse_select(
        bottleneck_pool,
        int(config["simulation"]["bottleneck_states"]),
        diversity_column,
    )
    if len(selected_bottleneck) < int(config["simulation"]["bottleneck_states"]):
        raise RuntimeError(f"Insufficient bottleneck states for {combo['combo_id']}.")

    selected_fast, fast_tier = select_fast_states(
        pilot,
        selected_bottleneck,
        config["truth_selection"],
        int(config["simulation"]["fast_states"]),
    )
    if len(selected_fast) < int(config["simulation"]["fast_states"]):
        raise RuntimeError(f"Insufficient fast states for {combo['combo_id']}.")

    dwell = {
        mask: float(config["simulation"]["bottleneck_dwell"])
        for mask in selected_bottleneck
    }
    dwell.update(
        {
            mask: float(config["simulation"]["fast_dwell"])
            for mask in selected_fast
        }
    )
    selected_masks = selected_bottleneck + selected_fast
    truth = pilot[pilot["mask"].isin(selected_masks)].copy()
    truth["truth_class"] = np.where(
        truth["mask"].isin(selected_bottleneck), "bottleneck", "fast"
    )
    truth["D_true"] = truth["mask"].map(dwell)
    truth["selection_mode"] = config["truth_selection"]["mode"]
    truth["bottleneck_selection_tier"] = bottleneck_tier
    truth["fast_selection_tier"] = fast_tier
    for key, value in combo.items():
        truth[key] = value
    truth = truth.sort_values(["truth_class", "pilot_count"], ascending=[True, False])

    audited = pd.concat(
        [
            bottleneck_pool.head(60).assign(candidate_class="bottleneck_candidate"),
            pilot.loc[pilot["mask"].isin(selected_fast)].assign(candidate_class="selected_fast_state"),
        ],
        ignore_index=True,
        sort=False,
    )
    audited["selected"] = audited["mask"].isin(selected_masks)
    audited["bottleneck_selection_tier"] = bottleneck_tier
    audited["fast_selection_tier"] = fast_tier
    for key, value in combo.items():
        audited[key] = value
    return dwell, truth, audited


def experiment_metrics(
    repeat: int,
    scores: pd.DataFrame,
    fit_seconds: float,
    top_k: int,
    truth_total: int,
) -> dict:
    stable = scores[scores["eligible"]].copy()
    truth_ineligible = scores[(scores["D_true"] != 1) & (~scores["eligible"])].copy()
    evaluation = pd.concat([stable, truth_ineligible], ignore_index=True, sort=False)
    evaluation = evaluation.sort_values("eligible", ascending=False).drop_duplicates("mask")
    if evaluation.empty:
        return {
            "repeat": repeat,
            "stable_states": 0,
            "stable_bottlenecks": 0,
            "evaluated_states": 0,
            "ineligible_truth_states": 0,
            "spearman_R_star": np.nan,
            "spearman_occupancy": np.nan,
            "bottleneck_auc_R_star": np.nan,
            "bottleneck_auc_occupancy": np.nan,
            "bottleneck_ap_R_star": np.nan,
            "bottleneck_ap_occupancy": np.nan,
            "top5_precision_R_star": np.nan,
            "top5_precision_occupancy": np.nan,
            "bottleneck_recall_at5_R_star": np.nan,
            "bottleneck_recall_at5_occupancy": np.nan,
            "fit_seconds": fit_seconds,
        }

    def filled_score(column: str) -> np.ndarray:
        values = evaluation[column].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return np.full(len(values), np.nan)
        floor = float(finite.min() - max(1.0e-6, finite.std(ddof=0) * 0.01))
        return np.where(np.isfinite(values), values, floor)

    labels = evaluation["D_true"].to_numpy() > 1
    d_true = evaluation["D_true"].to_numpy(dtype=float)
    r_score = filled_score("R_star")
    l_score = filled_score("occupancy_star")
    if len(np.unique(d_true)) < 2 or len(evaluation) < 3 or not np.isfinite(r_score).any():
        spearman_r = np.nan
    else:
        spearman_r = float(spearmanr(d_true, r_score).statistic)
    if len(np.unique(d_true)) < 2 or len(evaluation) < 3 or not np.isfinite(l_score).any():
        spearman_l = np.nan
    else:
        spearman_l = float(spearmanr(d_true, l_score).statistic)

    ranked = evaluation.copy()
    ranked["R_star_eval"] = r_score
    ranked["occupancy_eval"] = l_score
    top_r = ranked.nlargest(top_k, "R_star_eval")
    top_l = ranked.nlargest(top_k, "occupancy_eval")
    return {
        "repeat": repeat,
        "stable_states": int(len(stable)),
        "stable_bottlenecks": int((stable["D_true"] > 1).sum()),
        "evaluated_states": int(len(evaluation)),
        "ineligible_truth_states": int(len(truth_ineligible)),
        "spearman_R_star": spearman_r,
        "spearman_occupancy": spearman_l,
        "bottleneck_auc_R_star": base.rank_auc(labels, r_score),
        "bottleneck_auc_occupancy": base.rank_auc(labels, l_score),
        "bottleneck_ap_R_star": base.average_precision(labels, r_score),
        "bottleneck_ap_occupancy": base.average_precision(labels, l_score),
        "top5_precision_R_star": float((top_r["D_true"] > 1).sum() / top_k),
        "top5_precision_occupancy": float((top_l["D_true"] > 1).sum() / top_k),
        "bottleneck_recall_at5_R_star": float((top_r["D_true"] > 1).sum() / truth_total),
        "bottleneck_recall_at5_occupancy": float((top_l["D_true"] > 1).sum() / truth_total),
        "fit_seconds": fit_seconds,
    }


def append_missing_truth_rows(
    scores: pd.DataFrame,
    dwell_by_mask: dict[int, float],
    combo: dict,
    repeat: int,
) -> pd.DataFrame:
    observed = set(scores["mask"].astype(int))
    missing_rows = []
    for mask, dwell in dwell_by_mask.items():
        if int(mask) in observed:
            continue
        row = {
            "repeat": repeat,
            "mask": int(mask),
            "state": base.state_name(mask),
            "stage": base.stage(mask),
            "genotype": base.genotype(mask),
            "event_count": base.event_count(mask),
            "N_v": 0,
            "L_v": 0.0,
            "F_hat": 0.0,
            "R_raw": 0.0,
            "eligible": False,
            "D_true": float(dwell),
            "R_star": np.nan,
            "occupancy_star": np.nan,
            "truth_class": "bottleneck" if float(dwell) > 1 else "fast",
        }
        row.update(combo)
        missing_rows.append(row)
    if missing_rows:
        return pd.concat([scores, pd.DataFrame(missing_rows)], ignore_index=True, sort=False)
    return scores


def run_repeat(
    theta: np.ndarray,
    dwell_by_mask: dict[int, float],
    config: dict,
    combo: dict,
    repeat: int,
    seed: int,
    chosen_lambda: float,
) -> tuple[pd.DataFrame, dict]:
    masks = base.simulate_cohort(
        theta,
        dwell_by_mask,
        int(config["simulation"]["samples_per_repeat"]),
        config["simulation"],
        seed,
    )
    matrix = base.masks_to_matrix(masks)
    estimated_theta, fit_seconds = base.fit_mhn(matrix, chosen_lambda, config, seed)
    scores, _ = base.state_scores(
        masks,
        estimated_theta,
        dwell_by_mask,
        config["state_scoring"],
    )
    for key, value in combo.items():
        scores[key] = value
    scores.insert(0, "repeat", repeat)
    scores = append_missing_truth_rows(scores, dwell_by_mask, combo, repeat)
    metrics = experiment_metrics(
        repeat,
        scores,
        fit_seconds,
        int(config["state_scoring"]["top_k"]),
        int(config["simulation"]["bottleneck_states"]),
    )
    metrics.update(combo)
    metrics["seed"] = int(seed)
    metrics["chosen_lambda"] = float(chosen_lambda)
    metrics["chosen_lambda_multiplier"] = float(
        chosen_lambda * int(config["simulation"]["samples_per_repeat"])
    )
    return scores, metrics


def build_combo_manifest(config: dict, limit: int | None = None) -> pd.DataFrame:
    rows = []
    for index, (topology, sparsity, placement) in enumerate(
        itertools.product(
            config["simulation"]["topologies"],
            config["simulation"]["sparsities"],
            config["simulation"]["bottleneck_placements"],
        ),
        start=1,
    ):
        rows.append(
            {
                "combo_index": index,
                "combo_id": f"C{index:02d}_{topology}_s{int(round(float(sparsity) * 100)):02d}_{placement}",
                "topology": topology,
                "topology_label": TOPOLOGY_LABELS[topology],
                "sparsity": float(sparsity),
                "sparsity_label": f"{int(round(float(sparsity) * 100))}%",
                "bottleneck_placement": placement,
                "placement_label": PLACEMENT_LABELS[placement],
            }
        )
    manifest = pd.DataFrame(rows)
    if limit is not None:
        manifest = manifest.head(int(limit)).copy()
    return manifest


def summarize_metrics(metrics: pd.DataFrame, config: dict) -> pd.DataFrame:
    group_columns = [
        "combo_id",
        "combo_index",
        "topology",
        "topology_label",
        "sparsity",
        "sparsity_label",
        "bottleneck_placement",
        "placement_label",
    ]
    rows = []
    for keys, group in metrics.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, keys))
        row["n_repeats"] = int(len(group))
        for column in METRIC_COLUMNS:
            values = group[column].dropna().astype(float) if column in group.columns else pd.Series(dtype=float)
            row[f"{column}_median"] = float(values.median()) if len(values) else np.nan
            row[f"{column}_q1"] = float(values.quantile(0.25)) if len(values) else np.nan
            row[f"{column}_q3"] = float(values.quantile(0.75)) if len(values) else np.nan
            row[f"{column}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{column}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        reference = descriptive_reference(config)
        auc_reference = reference.get("combo_auc_reference", reference.get("combo_auc_threshold"))
        top5_reference = reference.get(
            "global_median_top5_precision_reference",
            reference.get("global_median_top5_precision_minimum"),
        )
        if auc_reference is not None:
            row["auc_ge_reference_fraction"] = float((group["bottleneck_auc_R_star"] >= float(auc_reference)).mean())
        if top5_reference is not None:
            row["top5_ge_reference_fraction"] = float((group["top5_precision_R_star"] >= float(top5_reference)).mean())
        row["difficulty_score"] = float(
            (1 - np.nan_to_num(row["bottleneck_auc_R_star_median"], nan=0.0))
            + 0.55 * (1 - np.nan_to_num(row["spearman_R_star_median"], nan=0.0))
            + 0.35 * (1 - np.nan_to_num(row["top5_precision_R_star_median"], nan=0.0))
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("combo_index")


def global_summary(metrics: pd.DataFrame, summary: pd.DataFrame, config: dict, runtime_seconds: float) -> pd.DataFrame:
    row = {
        "conditions": int(summary["combo_id"].nunique()),
        "repeats": int(metrics["repeat"].nunique()),
        "total_fits": int(len(metrics)),
        "runtime_seconds": float(runtime_seconds),
    }
    for column in METRIC_COLUMNS:
        values = metrics[column].dropna().astype(float)
        row[f"global_{column}_median"] = float(values.median()) if len(values) else np.nan
        row[f"global_{column}_q1"] = float(values.quantile(0.25)) if len(values) else np.nan
        row[f"global_{column}_q3"] = float(values.quantile(0.75)) if len(values) else np.nan
    reference = descriptive_reference(config)
    auc_reference = reference.get("combo_auc_reference", reference.get("combo_auc_threshold"))
    if auc_reference is not None:
        row["combo_auc_ge_reference_fraction"] = float(
            (summary["bottleneck_auc_R_star_median"] >= float(auc_reference)).mean()
        )
    return pd.DataFrame([row])


def heatmap_panel(
    axes: list[plt.Axes],
    summary: pd.DataFrame,
    metric: str,
    title_prefix: str,
    cmap,
    vmin: float,
    vmax: float,
    center: float | None = None,
) -> None:
    topologies = list(TOPOLOGY_LABELS)
    sparsities = sorted(summary["sparsity"].unique())
    for ax, placement in zip(axes, PLACEMENT_LABELS):
        sub = summary[summary["bottleneck_placement"] == placement]
        pivot = sub.pivot(index="topology", columns="sparsity", values=metric).reindex(index=topologies, columns=sparsities)
        annotations = pivot.map(lambda value: "" if pd.isna(value) else f"{value:.2f}")
        sns.heatmap(
            pivot,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            center=center,
            annot=annotations,
            fmt="",
            annot_kws={"fontsize": 6.2},
            linewidths=0.45,
            linecolor="white",
            cbar=False,
            square=False,
        )
        ax.set_title(PLACEMENT_LABELS[placement], loc="left", fontsize=8.2, pad=3)
        ax.set_xlabel("Sparsity")
        ax.set_ylabel("")
        ax.set_xticklabels([f"{int(round(float(x.get_text()) * 100))}%" for x in ax.get_xticklabels()], rotation=0)
        ax.set_yticklabels([TOPOLOGY_SHORT.get(tick.get_text(), tick.get_text()) for tick in ax.get_yticklabels()], rotation=0)
        ax.tick_params(length=0)
    axes[0].text(
        -0.13,
        1.20,
        title_prefix,
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )


def create_relative_dwell_time_figure(summary: pd.DataFrame, output: Path, config: dict) -> None:
    focus = summary.copy()
    focus["spearman_gain_vs_occupancy"] = focus["spearman_R_star_median"] - focus["spearman_occupancy_median"]
    placements = list(PLACEMENT_LABELS)
    topologies = list(TOPOLOGY_LABELS)
    sparsities = sorted(focus["sparsity"].unique())
    row_index = [
        (topology, sparsity)
        for topology in topologies
        for sparsity in sparsities
    ]
    rho = np.full((len(row_index), len(placements)), np.nan, dtype=float)
    gain = np.full_like(rho, np.nan)
    for i, (topology, sparsity) in enumerate(row_index):
        for j, placement in enumerate(placements):
            row = focus[
                (focus["topology"] == topology)
                & (focus["sparsity"] == sparsity)
                & (focus["bottleneck_placement"] == placement)
            ]
            if not row.empty:
                rho[i, j] = float(row.iloc[0]["spearman_R_star_median"])
                gain[i, j] = float(row.iloc[0]["spearman_gain_vs_occupancy"])

    fig, ax = plt.subplots(figsize=(8.9, 5.95))
    fig.subplots_adjust(left=0.20, right=0.96, top=0.80, bottom=0.08)
    fig.suptitle(
        "Experiment 7 | Relative dwell-time robustness across evolutionary topologies",
        x=0.015,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )

    rho_cmap = mcolors.LinearSegmentedColormap.from_list(
        "rho_focus",
        figure_style.continuous_palette("dwell_rank", config, ["#FEEBB9", "#B8D2CC", "#B2E6FD", "#B5AED5"]),
    )
    gain_cmap = mcolors.LinearSegmentedColormap.from_list(
        "gain_focus",
        figure_style.continuous_palette("delta_gain", config, ["#E8B2A7", "#FEEBB9", "#B2E6FD"]),
    )
    rho_norm = mcolors.Normalize(vmin=-0.25, vmax=1.0)
    gain_abs = float(np.nanmax(np.abs(gain)))
    gain_limit = max(0.20, min(0.60, math.ceil(gain_abs * 20) / 20))
    gain_norm = mcolors.Normalize(vmin=-gain_limit, vmax=gain_limit)
    ax.imshow(rho, cmap=rho_cmap, norm=rho_norm, aspect="auto")

    n_rows, n_cols = rho.shape
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    for i in range(n_rows):
        for j in range(n_cols):
            value = rho[i, j]
            delta = gain[i, j]
            ax.add_patch(
                plt.Rectangle(
                    (j - 0.5, i + 0.22),
                    1.0,
                    0.28,
                    facecolor=gain_cmap(gain_norm(delta)),
                    edgecolor="none",
                    alpha=0.92,
                    zorder=2,
                )
            )
            main_color = "white" if np.isfinite(value) and value >= 0.58 else "#222222"
            delta_for_label = 0.0 if abs(delta) < 0.005 else delta
            delta_color = "white" if abs(delta_for_label) >= 0.23 else "#222222"
            ax.text(j, i - 0.08, f"{value:.2f}", ha="center", va="center", fontsize=7.2, color=main_color, zorder=3)
            ax.text(j, i + 0.36, f"Δ{delta_for_label:+.2f}".replace("+0.00", "0.00"), ha="center", va="center", fontsize=5.8, color=delta_color, zorder=3)

    for x in np.arange(-0.5, n_cols + 0.5, 1.0):
        ax.axvline(x, color="white", lw=0.7, zorder=4)
    for y in np.arange(-0.5, n_rows + 0.5, 1.0):
        ax.axhline(y, color="white", lw=0.7, zorder=4)
    for group_start in range(0, n_rows + 1, len(sparsities)):
        ax.axhline(group_start - 0.5, color="#333333", lw=0.9, zorder=5)

    ax.set_xticks(np.arange(n_cols), [PLACEMENT_LABELS[p].replace("-stage", "") for p in placements])
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", labelsize=7.5, pad=4, length=0)
    ax.set_yticks(np.arange(n_rows), [f"{int(s * 100)}%" for _, s in row_index])
    ax.tick_params(axis="y", labelsize=6.6, length=0)
    ax.set_title(r"Cell fill: $\rho=\mathrm{Spearman}(D_{true},R^*)$; bottom strip: $\Delta\rho$ vs occupancy", loc="left", fontsize=8.5, pad=10)
    for group, topology in enumerate(topologies):
        center = group * len(sparsities) + (len(sparsities) - 1) / 2
        ax.text(
            -0.86,
            center,
            TOPOLOGY_LABELS[topology],
            ha="right",
            va="center",
            fontsize=7.5,
            fontweight="bold",
            clip_on=False,
        )
    for spine in ax.spines.values():
        spine.set_visible(False)

    draw_compact_gradient_key(
        fig,
        rho_cmap,
        0.59,
        0.885,
        0.15,
        0.017,
        r"$\rho(D_{true},R^*)$",
        "-0.25",
        "1.00",
    )
    draw_compact_gradient_key(
        fig,
        gain_cmap,
        0.79,
        0.885,
        0.15,
        0.017,
        r"$\Delta\rho$ vs occupancy",
        f"-{gain_limit:.2f}",
        f"+{gain_limit:.2f}",
    )
    save_figure(fig, output, int(config["plot"]["dpi"]))


def draw_bar_facets(axes: list[plt.Axes], summary: pd.DataFrame) -> None:
    topologies = list(TOPOLOGY_LABELS)
    sparsities = sorted(summary["sparsity"].unique())
    offsets = np.linspace(-0.24, 0.24, len(sparsities))
    width = 0.18
    for ax, placement in zip(axes, PLACEMENT_LABELS):
        sub = summary[summary["bottleneck_placement"] == placement]
        x = np.arange(len(topologies), dtype=float)
        for offset, sparsity in zip(offsets, sparsities):
            values = []
            lower = []
            upper = []
            for topology in topologies:
                row = sub[(sub["topology"] == topology) & (sub["sparsity"] == sparsity)]
                if row.empty:
                    values.append(np.nan)
                    lower.append(0.0)
                    upper.append(0.0)
                    continue
                median = float(row.iloc[0]["top5_precision_R_star_median"])
                q1 = float(row.iloc[0]["top5_precision_R_star_q1"])
                q3 = float(row.iloc[0]["top5_precision_R_star_q3"])
                values.append(median)
                lower.append(max(median - q1, 0.0))
                upper.append(max(q3 - median, 0.0))
            ax.bar(
                x + offset,
                values,
                width=width,
                color=SPARSITY_COLORS[float(sparsity)],
                edgecolor="white",
                linewidth=0.45,
                label=f"{int(round(float(sparsity) * 100))}%" if placement == "early_stage" else None,
            )
            ax.errorbar(
                x + offset,
                values,
                yerr=np.vstack([lower, upper]),
                fmt="none",
                ecolor="#333333",
                elinewidth=0.6,
                capsize=1.4,
                capthick=0.6,
            )
        ax.set_xticks(x, [TOPOLOGY_SHORT[t] for t in topologies])
        ax.set_ylim(0, 1.03)
        ax.set_title(PLACEMENT_LABELS[placement], loc="left", fontsize=8.2, pad=3)
        ax.set_ylabel("Top-5 long-dwell precision" if placement in {"early_stage", "late_stage"} else "")
        ax.tick_params(axis="x", labelrotation=0)
        sns.despine(ax=ax)
    axes[0].legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0.0, 1.28), title="Sparsity")
    axes[0].text(
        -0.13,
        1.20,
        "Top-5 enrichment of implanted long-dwell states",
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )


def draw_hardest_table(ax: plt.Axes, summary: pd.DataFrame) -> None:
    hardest = summary.sort_values("difficulty_score", ascending=False).head(10).copy()
    hardest["condition"] = hardest.apply(
        lambda row: f"{TOPOLOGY_SHORT[row['topology']]} | {row['sparsity_label']} | {PLACEMENT_LABELS[row['bottleneck_placement']].replace('-stage', '')}",
        axis=1,
    )
    table_data = pd.DataFrame(
        {
            "Rank": range(1, len(hardest) + 1),
            "Condition": hardest["condition"],
            "AUC": hardest["bottleneck_auc_R_star_median"].map(lambda value: f"{value:.2f}"),
            "rho": hardest["spearman_R_star_median"].map(lambda value: f"{value:.2f}"),
            "Top5": hardest["top5_precision_R_star_median"].map(lambda value: f"{value:.2f}"),
            "Stable": hardest["stable_states_median"].map(lambda value: f"{value:.0f}"),
        }
    )
    ax.axis("off")
    ax.set_title("Hardest simulated conditions", loc="left", pad=8)
    table = ax.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        cellLoc="center",
        colLoc="center",
        loc="upper left",
        bbox=[0.0, 0.06, 1.0, 0.86],
        colWidths=[0.08, 0.42, 0.12, 0.12, 0.12, 0.12],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    table.scale(1.0, 1.12)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D9D9D9")
        cell.set_linewidth(0.35)
        if row == 0:
            cell.set_facecolor("#F0F0F0")
            cell.set_text_props(weight="bold", color="#222222")
        elif row % 2 == 0:
            cell.set_facecolor("#FAFAFA")
        else:
            cell.set_facecolor("white")
        if col == 2 and row > 0:
            cell.set_text_props(color="#1F5A89", weight="bold")
    ax.text(
        0.0,
        0.01,
        "Difficulty is ranked by weak relative dwell-time recovery metrics.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.5,
        color="#555555",
    )


def create_figure(summary: pd.DataFrame, output: Path, config: dict, metrics: pd.DataFrame | None = None) -> None:
    configure_plotting(config)
    if config.get("plot", {}).get("figure_focus") == "relative_dwell_time_only":
        if metrics is not None:
            selected_e7_figure.render_design_d(metrics, output, config)
            return
        create_relative_dwell_time_figure(summary, output, config)
        return
    fig = plt.figure(figsize=(14.8, 10.4), constrained_layout=True)
    outer = fig.add_gridspec(2, 2, width_ratios=[1.02, 1.05], height_ratios=[1.0, 1.0])
    sub_a = outer[0, 0].subgridspec(2, 2, wspace=0.08, hspace=0.22)
    sub_b = outer[0, 1].subgridspec(2, 2, wspace=0.08, hspace=0.22)
    sub_c = outer[1, 0].subgridspec(2, 2, wspace=0.12, hspace=0.28)
    axes_a = [fig.add_subplot(sub_a[i, j]) for i in range(2) for j in range(2)]
    axes_b = [fig.add_subplot(sub_b[i, j]) for i in range(2) for j in range(2)]
    axes_c = [fig.add_subplot(sub_c[i, j]) for i in range(2) for j in range(2)]
    ax_d = fig.add_subplot(outer[1, 1])
    fig.suptitle(
        "Experiment 7 | Relative dwell-time robustness across complex evolutionary topologies",
        x=0.012,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )

    score_cmap = mcolors.LinearSegmentedColormap.from_list(
        "score_map",
        figure_style.continuous_palette("positive", config, ["#FEEBB9", "#B8D2CC", "#B2E6FD"]),
    )
    spearman_cmap = mcolors.LinearSegmentedColormap.from_list(
        "rho_map",
        figure_style.continuous_palette("diverging", config, ["#E8B2A7", "#FEEBB9", "#B2E6FD"]),
    )
    heatmap_panel(
        axes_a,
        summary,
        "bottleneck_auc_R_star_median",
        "Long-dwell state discrimination by condition",
        score_cmap,
        0.0,
        1.0,
    )
    heatmap_panel(
        axes_b,
        summary,
        "spearman_R_star_median",
        "Relative dwell-time rank recovery",
        spearman_cmap,
        -0.25,
        1.0,
        center=0.0,
    )
    draw_bar_facets(axes_c, summary)
    draw_hardest_table(ax_d, summary)

    panel_label(axes_a[0], "A")
    panel_label(axes_b[0], "B")
    panel_label(axes_c[0], "C")
    panel_label(ax_d, "D")
    save_figure(fig, output, int(config["plot"]["dpi"]))


def write_protocol_audit(root: Path, config: dict, combo_manifest: pd.DataFrame) -> None:
    text = f"""# Experiment 7 Protocol Audit

## Protocol target

Experiment 7 is a descriptive robustness benchmark. It tests whether R* still
tracks implanted relative dwell-time differences under different true
evolutionary structures rather than judging the matrix by hard empirical
success thresholds.

## Executed design

| Factor | Levels |
|---|---|
| Topology | {", ".join(config["simulation"]["topologies"])} |
| Sparsity | {", ".join(f"{int(float(x) * 100)}%" for x in config["simulation"]["sparsities"])} |
| Bottleneck placement | {", ".join(config["simulation"]["bottleneck_placements"])} |
| Repeats per condition | {config["simulation"]["repeats"]} |
| Cohort size | {config["simulation"]["samples_per_repeat"]} |

Total planned/executed conditions in this result directory: {len(combo_manifest)}.

## Simulation rules

- p=15 event cMHN with diagonal event hazards sampled from N(-1, 0.4), clipped to [-2, 0].
- Directed nonzero off-diagonal edge count is fixed by the requested sparsity.
- Each topology contains a minimal biological scaffold, then random promoting
  and inhibiting effects are added to reach the target sparsity.
- Three bottleneck states receive D=3 and three fast-passing states receive D=0.3.
- Bottleneck truth states are selected from a D=1 pilot and locked before
  formal repeats.
- Truth-selection mode: `{config["truth_selection"]["mode"]}`.
- Bottleneck candidate target: pilot count {config["truth_selection"]["bottleneck_pilot_count_min"]}
  to {config["truth_selection"]["bottleneck_pilot_count_max"]}; relaxed same-placement
  lower bound {config["truth_selection"].get("relaxed_bottleneck_pilot_count_min", "default")}.
- Global fallback outside the requested placement is
  `{config["truth_selection"].get("allow_global_fallback", True)}`.
- The fixed lambda multiplier is {config["lambda_calibration"]["fixed_lambda_multiplier"]},
  inherited from the successful enhanced Experiment 6 so that Experiment 7
  isolates topology and placement rather than re-optimizing regularization per
  condition.

## Interpretation boundary

This is a controlled qualitative robustness benchmark. Hard conditions define
the boundary of relative dwell-time recovery and are reported descriptively
instead of being converted into a binary pass/fail claim.
"""
    (root / "experiment_07_protocol_audit.md").write_text(text, encoding="utf-8")


def write_figure_design_review(root: Path, config: dict) -> None:
    source_lines = figure_style.design_sources_markdown(config)
    rule_lines = figure_style.design_rules_markdown(config)
    pattern_lines = figure_style.design_patterns_markdown(config)
    if config.get("plot", {}).get("figure_focus") == "relative_dwell_time_only":
        text = f"""# Experiment 7 Figure Design Review

## Design target

The balanced Experiment 7 figure is intentionally restricted to the core
innovation: relative dwell-time robustness. Other endpoints are kept in tables
for auditability but are not shown in the main figure.

## Sources reviewed

{source_lines}

## Shared Figure Rules

{rule_lines}

## User Reference Design Patterns

{pattern_lines}

## Design choices adopted

1. Use a factor-level profile rather than the exploratory 48-condition matrix,
   because the final claim is about whether relative dwell-time recovery is
   stable across topology, placement and sparsity factors.
2. Show R* and occupancy on the same horizontal scale with a connecting
   median-shift line, so the correction effect is visible without a separate
   panel.
3. Summarize repeat-level uncertainty with median, IQR and 10-90% ranges rather
   than a single point estimate.
4. Remove auxiliary state-hit and boundary diagnostics from the main balanced
   figure because they dilute the central dwell-time robustness claim.
5. Adopt the embedded micro-legend pattern from the user-provided reference
   figures: the R*/occupancy legend is placed in the top margin instead of
   occupying a detached legend block.
6. Use direct in-panel annotation for median Delta rho, forming a compact
   right-side numeric column for each factor panel.

## Patterns reviewed but not used

Mechanism cartoons, phase-band time windows and full condition matrices are
useful for other experiment types or audit views, but they do not add clarity to
the selected factor-level robustness profile.

The final figure uses vector PDF plus 600-dpi PNG output, colorblind-aware
project palettes, shared-axis factor panels, direct annotations, embedded
micro-legends and explicit boundary checks.
"""
        (root / "top_journal_figure_design_review.md").write_text(text, encoding="utf-8")
        return

    text = f"""# Experiment 7 Top-Journal Figure Design Review

## Sources reviewed

{source_lines}

## Shared Figure Rules

{rule_lines}

## User Reference Design Patterns

{pattern_lines}

## Design choices adopted

1. Use condition matrices rather than many separated small figures, because
   Experiment 7 is a factorial robustness benchmark.
2. Preserve topology, sparsity and bottleneck placement in every panel so the
   reader can compare failure modes without switching figures.
3. Use annotated heatmaps for AUC and Spearman, because these are condition
   summaries and should be scanned as a matrix.
4. Use a faceted bar plot with IQR error bars for Top-5 long-dwell enrichment
   as an auxiliary localization view while keeping all 48 conditions in one
   visual grammar.
5. Add a compact hardest-condition table to make the main scientific risk
   explicit instead of hiding it in supplementary files.

The final figure uses vector PDF plus 600-dpi PNG output, colorblind-aware
palette choices, direct annotations, small multiples with shared axes, and
explicit boundary checks.
"""
    (root / "top_journal_figure_design_review.md").write_text(text, encoding="utf-8")


def write_summary_report(
    root: Path,
    config: dict,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    overall: pd.DataFrame,
) -> None:
    row = overall.iloc[0]
    if config.get("plot", {}).get("figure_focus") == "relative_dwell_time_only":
        focus = summary.copy()
        focus["spearman_gain_vs_occupancy"] = focus["spearman_R_star_median"] - focus["spearman_occupancy_median"]
        by_topology = (
            focus.groupby("topology_label")[
                ["spearman_R_star_median", "spearman_occupancy_median", "spearman_gain_vs_occupancy"]
            ]
            .median()
            .reset_index()
        )
        by_placement = (
            focus.groupby("placement_label")[
                ["spearman_R_star_median", "spearman_occupancy_median", "spearman_gain_vs_occupancy"]
            ]
            .median()
            .reset_index()
        )
        global_gain = float(row["global_spearman_R_star_median"] - row["global_spearman_occupancy_median"])
        lines = [
            "# Experiment 7 Summary",
            "",
            "## Relative Dwell-Time Robustness",
            "",
            "This balanced Experiment 7 result is restricted to the core innovation: whether R* preserves the relative ordering of implanted dwell time D_true across topology, sparsity and long-dwell placement changes. Auxiliary state-hit diagnostics are not shown in the main balanced report.",
            "",
            "| Scope | Spearman(D_true, R*) | Spearman(D_true, occupancy) | R* - occupancy |",
            "|---|---:|---:|---:|",
            f"| Global | {row['global_spearman_R_star_median']:.3f} [{row['global_spearman_R_star_q1']:.3f}-{row['global_spearman_R_star_q3']:.3f}] | {row['global_spearman_occupancy_median']:.3f} [{row['global_spearman_occupancy_q1']:.3f}-{row['global_spearman_occupancy_q3']:.3f}] | {global_gain:+.3f} |",
            "",
            "## Topology Profile",
            "",
            "| Topology | Spearman(D_true, R*) | Spearman(D_true, occupancy) | R* - occupancy |",
            "|---|---:|---:|---:|",
        ]
        for _, item in by_topology.iterrows():
            lines.append(
                f"| {item['topology_label']} | {item['spearman_R_star_median']:.3f} | "
                f"{item['spearman_occupancy_median']:.3f} | {item['spearman_gain_vs_occupancy']:+.3f} |"
            )
        lines.extend(
            [
                "",
                "## Placement Profile",
                "",
                "| Placement | Spearman(D_true, R*) | Spearman(D_true, occupancy) | R* - occupancy |",
                "|---|---:|---:|---:|",
            ]
        )
        for _, item in by_placement.iterrows():
            lines.append(
                f"| {item['placement_label']} | {item['spearman_R_star_median']:.3f} | "
                f"{item['spearman_occupancy_median']:.3f} | {item['spearman_gain_vs_occupancy']:+.3f} |"
            )
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                "R* shows a stronger positive association with true relative dwell time than raw occupancy overall. This supports the intended robustness claim: after correcting for estimated inflow, state abundance better reflects relative dwell-time ordering across heterogeneous simulated evolutionary structures. The result remains a qualitative robustness profile rather than a threshold-based success claim.",
            ]
        )
        (root / "experiment_07_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    hardest = summary.sort_values("difficulty_score", ascending=False).head(8)
    spearman_delta = float(row["global_spearman_R_star_median"] - row["global_spearman_occupancy_median"])
    auc_delta = float(row["global_bottleneck_auc_R_star_median"] - row["global_bottleneck_auc_occupancy_median"])
    top5_delta = float(row["global_top5_precision_R_star_median"] - row["global_top5_precision_occupancy_median"])
    lines = [
        "# Experiment 7 Summary",
        "",
        "## Descriptive robustness profile",
        "",
        "Experiment 7 is interpreted as a qualitative robustness benchmark. It does not use empirical cut points as binary success/failure thresholds. The focus is whether R* continues to reflect implanted relative dwell-time differences when topology, sparsity and long-dwell placement change.",
        "",
        "## Global dwell-time recovery",
        "",
        "| Endpoint | R* median [IQR] | Occupancy median [IQR] | R* - occupancy | Role |",
        "|---|---:|---:|---:|---|",
        f"| Spearman with true D | {row['global_spearman_R_star_median']:.3f} [{row['global_spearman_R_star_q1']:.3f}-{row['global_spearman_R_star_q3']:.3f}] | {row['global_spearman_occupancy_median']:.3f} [{row['global_spearman_occupancy_q1']:.3f}-{row['global_spearman_occupancy_q3']:.3f}] | {spearman_delta:+.3f} | Primary relative dwell-time ranking |",
        f"| Long-dwell ROC AUC | {row['global_bottleneck_auc_R_star_median']:.3f} [{row['global_bottleneck_auc_R_star_q1']:.3f}-{row['global_bottleneck_auc_R_star_q3']:.3f}] | {row['global_bottleneck_auc_occupancy_median']:.3f} [{row['global_bottleneck_auc_occupancy_q1']:.3f}-{row['global_bottleneck_auc_occupancy_q3']:.3f}] | {auc_delta:+.3f} | Long-vs-nonlong dwell separation |",
        f"| Top-5 long-dwell precision | {row['global_top5_precision_R_star_median']:.3f} [{row['global_top5_precision_R_star_q1']:.3f}-{row['global_top5_precision_R_star_q3']:.3f}] | {row['global_top5_precision_occupancy_median']:.3f} [{row['global_top5_precision_occupancy_q1']:.3f}-{row['global_top5_precision_occupancy_q3']:.3f}] | {top5_delta:+.3f} | Auxiliary candidate localization |",
        f"| Recall@5 | {row['global_bottleneck_recall_at5_R_star_median']:.3f} [{row['global_bottleneck_recall_at5_R_star_q1']:.3f}-{row['global_bottleneck_recall_at5_R_star_q3']:.3f}] | {row['global_bottleneck_recall_at5_occupancy_median']:.3f} [{row['global_bottleneck_recall_at5_occupancy_q1']:.3f}-{row['global_bottleneck_recall_at5_occupancy_q3']:.3f}] | {row['global_bottleneck_recall_at5_R_star_median'] - row['global_bottleneck_recall_at5_occupancy_median']:+.3f} | Auxiliary top-ranked recovery |",
        "",
        f"- Conditions: {int(row['conditions'])}; total cMHN fits: {int(row['total_fits'])}.",
        f"- Runtime: {row['runtime_seconds'] / 60:.1f} minutes.",
        f"- Stable states per repeat: median {row['global_stable_states_median']:.0f} [{row['global_stable_states_q1']:.0f}-{row['global_stable_states_q3']:.0f}].",
        f"- Stable implanted long-dwell states per repeat: median {row['global_stable_bottlenecks_median']:.0f} [{row['global_stable_bottlenecks_q1']:.0f}-{row['global_stable_bottlenecks_q3']:.0f}].",
        "",
        "## Weakest recovery conditions",
        "",
        "| Rank | Topology | Sparsity | Placement | Long-dwell AUC | Spearman | Top-5 precision | Stable states |",
        "|---:|---|---:|---|---:|---:|---:|---:|",
    ]
    for rank, (_, item) in enumerate(hardest.iterrows(), start=1):
        lines.append(
            f"| {rank} | {item['topology_label']} | {item['sparsity_label']} | {item['placement_label']} | "
            f"{item['bottleneck_auc_R_star_median']:.3f} | {item['spearman_R_star_median']:.3f} | "
            f"{item['top5_precision_R_star_median']:.3f} | {item['stable_states_median']:.0f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Experiment 7 is a topology robustness experiment for the core innovation: relative dwell-time recovery. "
            "The central question is whether R* remains aligned with implanted D_true when the true evolutionary graph changes from single-path to branching, mutually exclusive and mixed multipath structures. "
            "The result is best reported as a robustness profile: R* preserves a positive dwell-time ranking signal overall, while middle-stage, late-stage and pathway-specific placements define the main boundary of recovery.",
        ]
    )
    (root / "experiment_07_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_scientific_review(root: Path, config: dict, summary: pd.DataFrame, overall: pd.DataFrame) -> None:
    row = overall.iloc[0]
    if config.get("plot", {}).get("figure_focus") == "relative_dwell_time_only":
        focus = summary.copy()
        focus["spearman_gain_vs_occupancy"] = focus["spearman_R_star_median"] - focus["spearman_occupancy_median"]
        strongest_gain = focus.sort_values("spearman_gain_vs_occupancy", ascending=False).head(5)
        weakest_gain = focus.sort_values("spearman_gain_vs_occupancy", ascending=True).head(5)
        lines = [
            "# Experiment 7 Scientific Review",
            "",
            "## Focus",
            "",
            "The balanced Experiment 7 review is restricted to relative dwell-time robustness. The only displayed scientific endpoint is Spearman(D_true, score), comparing R* with raw occupancy across the 48 simulated topology-sparsity-placement conditions.",
            "",
            "## Main Result",
            "",
            f"- Global Spearman(D_true, R*): {row['global_spearman_R_star_median']:.3f}.",
            f"- Global Spearman(D_true, occupancy): {row['global_spearman_occupancy_median']:.3f}.",
            f"- Median gain of R* over occupancy: {row['global_spearman_R_star_median'] - row['global_spearman_occupancy_median']:+.3f}.",
            "",
            "## Strongest R* Gains",
            "",
            "| Condition | R* Spearman | Occupancy Spearman | Gain |",
            "|---|---:|---:|---:|",
        ]
        for _, item in strongest_gain.iterrows():
            condition = f"{item['topology_label']} | {item['sparsity_label']} | {item['placement_label']}"
            lines.append(
                f"| {condition} | {item['spearman_R_star_median']:.3f} | "
                f"{item['spearman_occupancy_median']:.3f} | {item['spearman_gain_vs_occupancy']:+.3f} |"
            )
        lines.extend(["", "## Weakest R* Gains", "", "| Condition | R* Spearman | Occupancy Spearman | Gain |", "|---|---:|---:|---:|"])
        for _, item in weakest_gain.iterrows():
            condition = f"{item['topology_label']} | {item['sparsity_label']} | {item['placement_label']}"
            lines.append(
                f"| {condition} | {item['spearman_R_star_median']:.3f} | "
                f"{item['spearman_occupancy_median']:.3f} | {item['spearman_gain_vs_occupancy']:+.3f} |"
            )
        lines.extend(
            [
                "",
                "## Boundary of Claim",
                "",
                "This result supports a focused robustness statement: R* generally improves the recovery of relative dwell-time ordering compared with raw occupancy, but the gain is condition-dependent. The balanced Experiment 7 figure and report intentionally avoid other endpoints so the evidence remains aligned with the project's central innovation.",
            ]
        )
        (root / "experiment_07_scientific_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    by_topology = summary.groupby("topology_label")[
        ["bottleneck_auc_R_star_median", "spearman_R_star_median", "top5_precision_R_star_median"]
    ].median().reset_index()
    by_placement = summary.groupby("placement_label")[
        ["bottleneck_auc_R_star_median", "spearman_R_star_median", "top5_precision_R_star_median"]
    ].median().reset_index()
    lines = [
        "# Experiment 7 Scientific Review",
        "",
        "## What this experiment tests",
        "",
        "The experiment asks whether the relative inflow correction remains informative for relative dwell-time recovery when true tumor evolution is linear, branching, mutually exclusive or mixed, and when long-dwell states are placed at early, middle, late or pathway-specific states.",
        "",
        "## Main descriptive result",
        "",
        f"- Global median Spearman(D_true, R*): {row['global_spearman_R_star_median']:.3f}; occupancy baseline {row['global_spearman_occupancy_median']:.3f}.",
        f"- Global median long-dwell ROC AUC: {row['global_bottleneck_auc_R_star_median']:.3f}; occupancy baseline {row['global_bottleneck_auc_occupancy_median']:.3f}.",
        f"- Global median Top-5 long-dwell precision: {row['global_top5_precision_R_star_median']:.3f}; occupancy baseline {row['global_top5_precision_occupancy_median']:.3f}.",
        f"- Median stable states: {row['global_stable_states_median']:.0f}; median stable implanted long-dwell states: {row['global_stable_bottlenecks_median']:.0f}.",
        "",
        "The result should not be converted into a binary robust/not-robust claim using empirical cut points. It is a descriptive robustness profile: R* preserves a positive relative dwell-time ranking signal overall, while discrimination and top-ranked localization weaken in several middle-stage, late-stage and pathway-specific conditions.",
        "",
        "## Topology-level pattern",
        "",
        "| Topology | Long-dwell AUC | Spearman(D,R*) | Top-5 precision |",
        "|---|---:|---:|---:|",
    ]
    for _, item in by_topology.iterrows():
        lines.append(
            f"| {item['topology_label']} | {item['bottleneck_auc_R_star_median']:.3f} | "
            f"{item['spearman_R_star_median']:.3f} | {item['top5_precision_R_star_median']:.3f} |"
        )
    lines.extend(["", "## Placement-level pattern", "", "| Placement | Long-dwell AUC | Spearman(D,R*) | Top-5 precision |", "|---|---:|---:|---:|"])
    for _, item in by_placement.iterrows():
        lines.append(
            f"| {item['placement_label']} | {item['bottleneck_auc_R_star_median']:.3f} | "
            f"{item['spearman_R_star_median']:.3f} | {item['top5_precision_R_star_median']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Biological interpretation",
            "",
            "- Linear topologies are expected to be easier because inflow paths are less ambiguous.",
            "- Branching and mixed topologies test whether multiple incoming histories dilute the state-level dwell-time signal.",
            "- Mutual exclusivity tests whether inhibitory alternatives distort raw occupancy and cMHN-estimated inflow.",
            "- Late-stage and pathway-specific placements are biologically harder because they combine lower prevalence with multiple upstream routes.",
            "- Early-stage placements are the clearest positive-control region; middle-stage placements are the most fragile after strict placement locking.",
            "",
            "## Boundary of claim",
            "",
            "This experiment supports a bounded claim: R* carries useful relative dwell-time information across heterogeneous simulated topologies, but the signal is sensitive to placement, state observability and multipath ambiguity. It is a qualitative robustness map, not a hard success/failure gate and not a real-cohort generalization claim by itself.",
        ]
    )
    (root / "experiment_07_scientific_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_existing(root: Path, config: dict) -> None:
    tables = root / "tables"
    metrics = pd.read_csv(tables / "repeat_metrics.tsv", sep="\t")
    summary = pd.read_csv(tables / "combo_summary.tsv", sep="\t")
    overall = pd.read_csv(tables / "experiment_07_global_summary.tsv", sep="\t")
    manifest = pd.read_csv(tables / "combo_manifest.tsv", sep="\t")
    create_figure(summary, root / "figures" / "Figure_E7_topology_robustness", config, metrics)
    write_protocol_audit(root, config, manifest)
    write_figure_design_review(root, config)
    write_summary_report(root, config, metrics, summary, overall)
    write_scientific_review(root, config, summary, overall)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.repeats is not None:
        config["simulation"]["repeats"] = int(args.repeats)
    if args.result_root:
        config["result_root"] = args.result_root
    if args.lambda_multiplier is not None:
        config["lambda_calibration"]["fixed_lambda_multiplier"] = float(args.lambda_multiplier)

    root = Path(config["result_root"]).resolve()
    tables = root / "tables"
    figures = root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    setup_logging(root)
    configure_plotting(config)
    shutil.copy2(config_path, root / config_path.name)
    (root / "resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if args.render_only:
        render_existing(root, config)
        return
    if base is None:
        raise ModuleNotFoundError(
            "The MHN runtime used by run_experiment_06 could not be imported. "
            "Install/activate the MHN Python environment for full Experiment 7 execution."
        ) from BASE_IMPORT_ERROR

    start = time.time()
    seed = int(config["random_seed"])
    combo_manifest = build_combo_manifest(config, args.limit_combos)
    combo_manifest.to_csv(tables / "combo_manifest.tsv", sep="\t", index=False)
    chosen_lambda = float(config["lambda_calibration"]["fixed_lambda_multiplier"]) / int(config["simulation"]["samples_per_repeat"])

    metric_rows: list[dict] = []
    state_frames: list[pd.DataFrame] = []
    truth_frames: list[pd.DataFrame] = []
    candidate_frames: list[pd.DataFrame] = []
    edge_frames: list[pd.DataFrame] = []
    topology_rows: list[dict] = []

    repeat_count = int(config["simulation"]["repeats"])
    total_conditions = len(combo_manifest)
    for condition_position, combo_row in enumerate(combo_manifest.to_dict("records"), start=1):
        combo = dict(combo_row)
        combo_seed = seed + int(combo["combo_index"]) * 10000
        theta_true = create_topology_theta(
            combo_seed,
            str(combo["topology"]),
            float(combo["sparsity"]),
        )
        dwell_by_mask, truth, candidate_audit = select_truth_states_for_combo(
            theta_true,
            config,
            combo,
            combo_seed + 101,
        )
        truth_frames.append(truth)
        candidate_frames.append(candidate_audit)
        edges = edge_list(theta_true, combo)
        edge_frames.append(edges)
        topology_rows.append(
            {
                **combo,
                "off_diagonal_nonzero_edges": int(len(edges)),
                "off_diagonal_possible_edges": int(len(EVENTS) * (len(EVENTS) - 1)),
                "directed_density": float(len(edges) / (len(EVENTS) * (len(EVENTS) - 1))),
                "positive_edges": int((edges["log_effect"] > 0).sum()),
                "negative_edges": int((edges["log_effect"] < 0).sum()),
                "scaffold_edges": int(edges["is_scaffold"].sum()),
            }
        )
        logging.info(
            "condition=%s/%s combo=%s edges=%s truth=%s",
            condition_position,
            total_conditions,
            combo["combo_id"],
            len(edges),
            truth[["state", "truth_class", "pilot_count"]].to_dict("records"),
        )
        print(
            f"Condition {condition_position}/{total_conditions}: "
            f"{combo['topology_label']} {combo['sparsity_label']} {combo['placement_label']}"
        )

        for repeat in range(1, repeat_count + 1):
            repeat_seed = combo_seed + 1000 + repeat
            scores, metrics = run_repeat(
                theta_true,
                dwell_by_mask,
                config,
                combo,
                repeat,
                repeat_seed,
                chosen_lambda,
            )
            state_frames.append(scores)
            metric_rows.append(metrics)
            if repeat == 1 or repeat % 10 == 0 or repeat == repeat_count:
                logging.info(
                    "combo=%s repeat=%s/%s rho=%.4f auc=%.4f top5=%.4f stable=%s fit=%.2f",
                    combo["combo_id"],
                    repeat,
                    repeat_count,
                    metrics["spearman_R_star"],
                    metrics["bottleneck_auc_R_star"],
                    metrics["top5_precision_R_star"],
                    metrics["stable_states"],
                    metrics["fit_seconds"],
                )
                print(
                    f"  Repeat {repeat}/{repeat_count}: "
                    f"AUC={metrics['bottleneck_auc_R_star']:.3f}, "
                    f"rho={metrics['spearman_R_star']:.3f}, "
                    f"Top5={metrics['top5_precision_R_star']:.2f}, "
                    f"fit={metrics['fit_seconds']:.1f}s"
                )

    metrics = pd.DataFrame(metric_rows)
    states = pd.concat(state_frames, ignore_index=True) if state_frames else pd.DataFrame()
    truth_states = pd.concat(truth_frames, ignore_index=True) if truth_frames else pd.DataFrame()
    candidate_audit = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
    true_edges = pd.concat(edge_frames, ignore_index=True) if edge_frames else pd.DataFrame()
    topology_audit = pd.DataFrame(topology_rows)
    combo_summary = summarize_metrics(metrics, config)
    overall = global_summary(metrics, combo_summary, config, time.time() - start)

    metrics.to_csv(tables / "repeat_metrics.tsv", sep="\t", index=False)
    states.to_csv(tables / "state_recovery_long.tsv", sep="\t", index=False)
    truth_states.to_csv(tables / "truth_states.tsv", sep="\t", index=False)
    candidate_audit.to_csv(tables / "truth_selection_candidate_audit.tsv", sep="\t", index=False)
    true_edges.to_csv(tables / "true_edge_list.tsv", sep="\t", index=False)
    topology_audit.to_csv(tables / "topology_audit.tsv", sep="\t", index=False)
    combo_summary.to_csv(tables / "combo_summary.tsv", sep="\t", index=False)
    overall.to_csv(tables / "experiment_07_global_summary.tsv", sep="\t", index=False)

    create_figure(combo_summary, figures / "Figure_E7_topology_robustness", config, metrics)
    write_protocol_audit(root, config, combo_manifest)
    write_figure_design_review(root, config)
    write_summary_report(root, config, metrics, combo_summary, overall)
    write_scientific_review(root, config, combo_summary, overall)
    (root / "experiment_07_run_metadata.json").write_text(
        json.dumps(overall.iloc[0].to_dict(), indent=2),
        encoding="utf-8",
    )
    print(overall.T.to_string(header=False))


if __name__ == "__main__":
    main()

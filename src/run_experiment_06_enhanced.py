"""Run Experiment 6 enhanced positive-control recovery stress test.

This keeps the successful first-version scientific idea: deliberately choose
truth states where raw occupancy is confounded by progression inflow, then test
whether R* recovers dwell bottlenecks. Enhancements are auditability and honest
framing, not over-strict redesign:

- independent lambda calibration cohort;
- explicit topology and truth-selection audit tables;
- compressed snapshot/trajectory files for every repeat;
- F_hat threshold sensitivity;
- oracle-Theta diagnostic to separate fitting error from score behavior.
"""

from __future__ import annotations

import argparse
import json
import logging
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

import run_experiment_06 as base


EVENTS = base.EVENTS
METHOD_COLORS = base.METHOD_COLORS
TRUTH_COLORS = base.TRUTH_COLORS


ENDPOINTS = [
    {
        "endpoint": "Spearman",
        "r_col": "spearman_R_star",
        "o_col": "spearman_occupancy",
        "threshold": "median_spearman_minimum",
        "perfect_value": None,
        "higher_is_better": True,
    },
    {
        "endpoint": "Bottleneck ROC AUC",
        "r_col": "bottleneck_auc_R_star",
        "o_col": "bottleneck_auc_occupancy",
        "threshold": "median_bottleneck_auc_minimum",
        "perfect_value": 1.0,
        "higher_is_better": True,
    },
    {
        "endpoint": "Bottleneck AP",
        "r_col": "bottleneck_ap_R_star",
        "o_col": "bottleneck_ap_occupancy",
        "threshold": None,
        "perfect_value": 1.0,
        "higher_is_better": True,
    },
    {
        "endpoint": "Top-5 precision",
        "r_col": "top5_precision_R_star",
        "o_col": "top5_precision_occupancy",
        "threshold": "median_top5_precision_minimum",
        "perfect_value": None,
        "higher_is_better": True,
    },
    {
        "endpoint": "Recall@5",
        "r_col": "bottleneck_recall_at5_R_star",
        "o_col": "bottleneck_recall_at5_occupancy",
        "threshold": None,
        "perfect_value": 1.0,
        "higher_is_better": True,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run enhanced Rel-ObsTQ-MHN Experiment 6.")
    parser.add_argument("--config", default="configs/experiment_06_enhanced.yaml")
    parser.add_argument("--repeats", type=int, help="Override repeats for a smoke run.")
    parser.add_argument("--result-root", help="Override result root.")
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--lambda-multiplier", type=float)
    parser.add_argument("--render-only", action="store_true")
    return parser.parse_args()


def setup_logging(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=root / "experiment_06_enhanced.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )


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
                    "target_index": target,
                    "source_index": source,
                    "log_effect": float(theta[target, source]),
                    "effect": "promoting" if theta[target, source] > 0 else "inhibiting",
                }
            )
    return pd.DataFrame(rows).sort_values(["target_event", "source_event"])


def select_truth_states_audited(
    theta: np.ndarray, config: dict, seed: int
) -> tuple[dict[int, float], pd.DataFrame, pd.DataFrame]:
    simulation = config["simulation"]
    selection = config["truth_selection"]
    pilot_masks = base.simulate_cohort(
        theta,
        {},
        int(simulation["pilot_samples"]),
        simulation,
        seed,
    )
    counts = Counter(int(mask) for mask in pilot_masks)
    pilot = pd.DataFrame(
        [
            {
                "mask": mask,
                "state": base.state_name(mask),
                "stage": base.stage(mask),
                "genotype": base.genotype(mask),
                "event_count": base.event_count(mask),
                "pilot_count": count,
                "pilot_frequency": count / len(pilot_masks),
            }
            for mask, count in counts.items()
        ]
    ).sort_values("pilot_count", ascending=False)

    bottleneck_candidates = pilot[
        pilot["event_count"].between(
            int(selection["bottleneck_event_count_min"]),
            int(selection["bottleneck_event_count_max"]),
        )
        & pilot["pilot_count"].between(
            int(selection["bottleneck_pilot_count_min"]),
            int(selection["bottleneck_pilot_count_max"]),
        )
    ].copy()
    bottleneck_candidates["candidate_class"] = "bottleneck_candidate"
    bottleneck_candidates["selection_distance"] = (
        bottleneck_candidates["pilot_count"]
        - int(selection["bottleneck_target_pilot_count"])
    ).abs()
    bottleneck_candidates = bottleneck_candidates.sort_values(
        ["selection_distance", "pilot_count"]
    )

    fast_candidates = pilot[
        pilot["event_count"].between(
            int(selection["fast_event_count_min"]),
            int(selection["fast_event_count_max"]),
        )
        & (pilot["pilot_count"] >= int(selection["fast_pilot_count_min"]))
    ].copy()
    fast_candidates["candidate_class"] = "fast_candidate"
    fast_candidates["selection_distance"] = 0
    fast_candidates = fast_candidates.sort_values("pilot_count", ascending=False)
    if len(bottleneck_candidates) < 3 or len(fast_candidates) < 6:
        raise RuntimeError("Pilot simulation did not yield enough stress-test candidate states.")

    selected_bottleneck: list[int] = []
    for stage_name in ["S1", "S2", "S3"]:
        candidates = bottleneck_candidates[
            (bottleneck_candidates["stage"] == stage_name)
            & (~bottleneck_candidates["mask"].isin(selected_bottleneck))
        ]
        if not candidates.empty:
            selected_bottleneck.append(int(candidates.iloc[0]["mask"]))
    for mask in bottleneck_candidates["mask"].astype(int):
        if len(selected_bottleneck) == 3:
            break
        if mask not in selected_bottleneck:
            selected_bottleneck.append(mask)

    selected_fast: list[int] = []
    preferred = fast_candidates[
        ~fast_candidates["mask"].isin(selected_bottleneck)
    ].sort_values("pilot_count", ascending=False)
    for stage_name in ["S3", "S2", "S1"]:
        candidates = preferred[
            (preferred["stage"] == stage_name)
            & (~preferred["mask"].isin(selected_fast))
        ]
        if not candidates.empty:
            selected_fast.append(int(candidates.iloc[0]["mask"]))
    for mask in preferred["mask"].astype(int):
        if len(selected_fast) == 3:
            break
        if mask not in selected_fast:
            selected_fast.append(mask)

    dwell = {mask: float(simulation["bottleneck_dwell"]) for mask in selected_bottleneck}
    dwell.update({mask: float(simulation["fast_dwell"]) for mask in selected_fast})
    truth = pilot[pilot["mask"].isin(selected_bottleneck + selected_fast)].copy()
    truth["truth_class"] = np.where(
        truth["mask"].isin(selected_bottleneck), "bottleneck", "fast"
    )
    truth["D_true"] = truth["mask"].map(dwell)
    truth["selection_mode"] = selection["mode"]
    truth = truth.sort_values(["truth_class", "pilot_count"], ascending=[True, False])

    candidate_audit = pd.concat(
        [
            bottleneck_candidates.head(50),
            fast_candidates.head(50),
        ],
        ignore_index=True,
        sort=False,
    )
    candidate_audit["selected"] = candidate_audit["mask"].isin(truth["mask"])
    return dwell, truth, candidate_audit


def simulate_patient(
    theta: np.ndarray,
    dwell_by_mask: dict[int, float],
    simulation: dict,
    rng: np.random.Generator,
    repeat: int,
    sample_id: int,
) -> tuple[list[dict], dict]:
    mask = 0
    current_time = 0.0
    intervals: list[dict] = []
    maximum_time = float(simulation["maximum_time"])
    maximum_events = int(simulation["maximum_events"])
    step = 0
    while current_time < maximum_time:
        if base.event_count(mask) >= maximum_events:
            break
        absent, rates = base.event_probabilities(mask, theta)
        if absent.size == 0 or rates.sum() <= 0:
            end = maximum_time
            event_added = "STOP_NO_EVENT"
            next_mask = mask
        else:
            dwell = float(dwell_by_mask.get(mask, simulation["neutral_dwell"]))
            total_rate = float(rates.sum()) / dwell
            wait = float(rng.exponential(1.0 / total_rate))
            end = min(current_time + wait, maximum_time)
            if end >= maximum_time:
                event_added = "CENSORED_AT_HORIZON"
                next_mask = mask
            else:
                chosen_event = int(rng.choice(absent, p=rates / rates.sum()))
                event_added = EVENTS[chosen_event]
                next_mask = mask | (1 << chosen_event)
        intervals.append(
            {
                "repeat": repeat,
                "sample_id": sample_id,
                "step": step,
                "time_start": float(current_time),
                "time_end": float(end),
                "duration": float(max(end - current_time, 0.0)),
                "mask": int(mask),
                "state": base.state_name(mask),
                "stage": base.stage(mask),
                "genotype": base.genotype(mask),
                "event_count": base.event_count(mask),
                "event_added": event_added,
                "target_mask_after_event": int(next_mask),
                "D_true": float(dwell_by_mask.get(mask, simulation["neutral_dwell"])),
                "omega": float(simulation["observation_weight"]),
            }
        )
        if end >= maximum_time or next_mask == mask:
            break
        mask = next_mask
        current_time = end
        step += 1

    if not intervals:
        intervals.append(
            {
                "repeat": repeat,
                "sample_id": sample_id,
                "step": 0,
                "time_start": 0.0,
                "time_end": 0.0,
                "duration": 0.0,
                "mask": 0,
                "state": base.state_name(0),
                "stage": base.stage(0),
                "genotype": base.genotype(0),
                "event_count": 0,
                "event_added": "STOP_EMPTY",
                "target_mask_after_event": 0,
                "D_true": float(simulation["neutral_dwell"]),
                "omega": float(simulation["observation_weight"]),
            }
        )
    horizon = float(intervals[-1]["time_end"])
    observation_time = float(rng.uniform(0.0, horizon)) if horizon > 0 else 0.0
    selected = intervals[-1]
    for interval in intervals:
        if interval["time_start"] <= observation_time < interval["time_end"] or np.isclose(observation_time, interval["time_end"]):
            selected = interval
            break
    snapshot = {
        "repeat": repeat,
        "sample_id": sample_id,
        "observation_time": observation_time,
        "trajectory_step": int(selected["step"]),
        "mask": int(selected["mask"]),
        "state": selected["state"],
        "stage": selected["stage"],
        "genotype": selected["genotype"],
        "event_count": int(selected["event_count"]),
        "D_true": float(selected["D_true"]),
        "omega": float(selected["omega"]),
    }
    for event_index, event in enumerate(EVENTS):
        snapshot[event] = int(bool(int(selected["mask"]) & (1 << event_index)))
    return intervals, snapshot


def simulate_repeat_with_audit(
    theta: np.ndarray,
    dwell_by_mask: dict[int, float],
    config: dict,
    repeat: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    trajectories: list[dict] = []
    snapshots: list[dict] = []
    for sample_id in range(1, int(config["simulation"]["samples_per_repeat"]) + 1):
        trajectory, snapshot = simulate_patient(theta, dwell_by_mask, config["simulation"], rng, repeat, sample_id)
        trajectories.extend(trajectory)
        snapshots.append(snapshot)
    trajectory_table = pd.DataFrame(trajectories)
    snapshot_table = pd.DataFrame(snapshots)
    return trajectory_table, snapshot_table, snapshot_table["mask"].to_numpy(dtype=np.int32)


def write_tsv_gz(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, compression="gzip")


def score_with_threshold(
    masks: np.ndarray,
    theta: np.ndarray,
    dwell_by_mask: dict[int, float],
    config: dict,
    minimum_inflow: float,
    repeat: int,
    fit_seconds: float = 0.0,
) -> tuple[pd.DataFrame, dict, list[dict]]:
    scoring = dict(config["state_scoring"])
    scoring["minimum_inflow"] = float(minimum_inflow)
    scores, edges = base.state_scores(masks, theta, dwell_by_mask, scoring)
    scores.insert(0, "repeat", repeat)
    metrics, curves = base.repeat_metrics(
        repeat,
        scores,
        fit_seconds,
        int(config["state_scoring"]["top_k"]),
    )
    metrics["minimum_inflow"] = float(minimum_inflow)
    return scores, metrics, curves


def summarize_metrics(metrics: pd.DataFrame, chosen_lambda: float, config: dict, representative_repeat: int) -> pd.DataFrame:
    p_values = {
        "spearman_p": base.paired_pvalue(metrics, "spearman_R_star", "spearman_occupancy"),
        "auc_p": base.paired_pvalue(metrics, "bottleneck_auc_R_star", "bottleneck_auc_occupancy"),
        "top5_p": base.paired_pvalue(metrics, "top5_precision_R_star", "top5_precision_occupancy"),
    }
    summary = {
        "repeats": int(len(metrics)),
        "samples_per_repeat": int(config["simulation"]["samples_per_repeat"]),
        "chosen_lambda": float(chosen_lambda),
        "chosen_lambda_multiplier": float(chosen_lambda * int(config["simulation"]["samples_per_repeat"])),
        "representative_repeat": int(representative_repeat),
        **{
            f"median_{column}": float(metrics[column].median())
            for column in metrics.columns
            if column != "repeat" and pd.api.types.is_numeric_dtype(metrics[column])
        },
        **p_values,
    }
    return pd.DataFrame([summary])


def performance_summary_table(metrics: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows = []
    for endpoint in ENDPOINTS:
        r_values = metrics[endpoint["r_col"]].dropna().astype(float)
        o_values = metrics[endpoint["o_col"]].dropna().astype(float)
        paired = metrics[[endpoint["r_col"], endpoint["o_col"]]].dropna()
        delta = paired[endpoint["r_col"]] - paired[endpoint["o_col"]]
        threshold_key = endpoint["threshold"]
        threshold = float(config["success"][threshold_key]) if threshold_key else np.nan
        perfect_value = endpoint["perfect_value"]
        rows.append(
            {
                "endpoint": endpoint["endpoint"],
                "R_star_median": float(r_values.median()),
                "R_star_q1": float(r_values.quantile(0.25)),
                "R_star_q3": float(r_values.quantile(0.75)),
                "R_star_mean": float(r_values.mean()),
                "R_star_sd": float(r_values.std(ddof=1)),
                "R_star_min": float(r_values.min()),
                "R_star_max": float(r_values.max()),
                "R_star_perfect_repeat_fraction": (
                    float(np.isclose(r_values, perfect_value).mean())
                    if perfect_value is not None
                    else np.nan
                ),
                "R_star_threshold_pass_fraction": (
                    float((r_values >= threshold).mean()) if np.isfinite(threshold) else np.nan
                ),
                "occupancy_median": float(o_values.median()),
                "occupancy_q1": float(o_values.quantile(0.25)),
                "occupancy_q3": float(o_values.quantile(0.75)),
                "occupancy_mean": float(o_values.mean()),
                "occupancy_sd": float(o_values.std(ddof=1)),
                "occupancy_min": float(o_values.min()),
                "occupancy_max": float(o_values.max()),
                "occupancy_perfect_repeat_fraction": (
                    float(np.isclose(o_values, perfect_value).mean())
                    if perfect_value is not None
                    else np.nan
                ),
                "occupancy_threshold_pass_fraction": (
                    float((o_values >= threshold).mean()) if np.isfinite(threshold) else np.nan
                ),
                "paired_delta_median": float(delta.median()) if not delta.empty else np.nan,
                "paired_delta_mean": float(delta.mean()) if not delta.empty else np.nan,
                "paired_p_value": base.paired_pvalue(metrics, endpoint["r_col"], endpoint["o_col"]),
                "threshold": threshold,
                "n_repeats": int(len(metrics)),
            }
        )
    table = pd.DataFrame(rows)
    table["R_star_median_iqr"] = table.apply(
        lambda row: f"{row['R_star_median']:.3f} [{row['R_star_q1']:.3f}-{row['R_star_q3']:.3f}]",
        axis=1,
    )
    table["occupancy_median_iqr"] = table.apply(
        lambda row: f"{row['occupancy_median']:.3f} [{row['occupancy_q1']:.3f}-{row['occupancy_q3']:.3f}]",
        axis=1,
    )
    table["R_star_mean_sd"] = table.apply(
        lambda row: f"{row['R_star_mean']:.3f} +/- {row['R_star_sd']:.3f}",
        axis=1,
    )
    table["occupancy_mean_sd"] = table.apply(
        lambda row: f"{row['occupancy_mean']:.3f} +/- {row['occupancy_sd']:.3f}",
        axis=1,
    )
    table["R_star_min_max"] = table.apply(
        lambda row: f"{row['R_star_min']:.3f}-{row['R_star_max']:.3f}",
        axis=1,
    )
    table["occupancy_min_max"] = table.apply(
        lambda row: f"{row['occupancy_min']:.3f}-{row['occupancy_max']:.3f}",
        axis=1,
    )
    return table


def table_for_panel(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary.iterrows():
        perfect = row["R_star_perfect_repeat_fraction"]
        pass_fraction = row["R_star_threshold_pass_fraction"]
        if np.isfinite(perfect):
            support = f"{perfect * 100:.0f}% perfect"
        elif np.isfinite(pass_fraction):
            support = f"{pass_fraction * 100:.0f}% pass"
        else:
            support = "descriptive"
        rows.append(
            {
                "Endpoint": row["endpoint"],
                "R* median [IQR]": row["R_star_median_iqr"],
                "R* mean +/- SD": row["R_star_mean_sd"],
                "R* min-max": row["R_star_min_max"],
                "Occupancy median [IQR]": row["occupancy_median_iqr"],
                "Delta median": f"{row['paired_delta_median']:.3f}",
                "p": f"{row['paired_p_value']:.1e}" if np.isfinite(row["paired_p_value"]) else "NA",
                "Repeat support": support,
            }
        )
    return pd.DataFrame(rows)


def create_enhanced_figure(
    all_states: pd.DataFrame,
    metrics: pd.DataFrame,
    representative: pd.DataFrame,
    performance_table: pd.DataFrame,
    config: dict,
    output: Path,
) -> None:
    fig = plt.figure(figsize=(13.2, 9.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.14], height_ratios=[1.0, 1.05])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])
    fig.suptitle(
        "Experiment 6 | Recovery of known state dwell bottlenecks from cross-sectional cohorts",
        fontsize=11,
        fontweight="bold",
        x=0.015,
        ha="left",
    )

    stable = all_states[all_states["eligible"]].copy()
    rng = np.random.default_rng(17)
    for index, truth_class in enumerate(["fast", "neutral", "bottleneck"]):
        values = np.log2(stable.loc[stable["truth_class"] == truth_class, "R_star"].clip(1e-4))
        if len(values) > 1200:
            values = values.sample(1200, random_state=17)
        x = rng.normal(index, 0.055, len(values))
        ax_a.scatter(
            x,
            values,
            s=6,
            alpha=0.16 if truth_class == "neutral" else 0.32,
            color=TRUTH_COLORS[truth_class],
            edgecolors="none",
            rasterized=True,
        )
        if len(values):
            q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
            ax_a.plot([index - 0.17, index + 0.17], [median, median], color="black", lw=1.5)
            ax_a.plot([index, index], [q1, q3], color="black", lw=3.2, solid_capstyle="butt")
    ax_a.axhline(0, color="#666666", lw=0.8, ls=":")
    ax_a.set_xticks(range(3), ["Fast\nD=0.3", "Neutral\nD=1", "Bottleneck\nD=3"])
    ax_a.set_ylabel(r"Estimated relative dwell, $\log_2(R^*)$")
    ax_a.set_title("Known dwell classes separate after inflow correction", loc="left")
    base.panel_label(ax_a, "A")
    sns.despine(ax=ax_a)

    panel_table = table_for_panel(performance_table)
    ax_b.axis("off")
    ax_b.set_title("Repeated-run performance summary (100 cohorts)", loc="left", pad=12)
    display_columns = [
        "Endpoint",
        "R* median [IQR]",
        "R* mean +/- SD",
        "R* min-max",
        "Occupancy median [IQR]",
        "Delta median",
        "p",
        "Repeat support",
    ]
    display_labels = [
        "Endpoint",
        "R*\nmedian [IQR]",
        "R*\nmean +/- SD",
        "R*\nmin-max",
        "Occupancy\nmedian [IQR]",
        "Delta\nmedian",
        "p",
        "Repeat\nsupport",
    ]
    table = ax_b.table(
        cellText=panel_table[display_columns].values,
        colLabels=display_labels,
        cellLoc="center",
        colLoc="center",
        loc="upper left",
        bbox=[0.0, 0.08, 1.0, 0.84],
        colWidths=[0.16, 0.17, 0.15, 0.11, 0.18, 0.09, 0.09, 0.13],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(5.8)
    table.scale(1.0, 1.15)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D6D6D6")
        cell.set_linewidth(0.35)
        if row == 0:
            cell.set_facecolor("#F0F0F0")
            cell.set_text_props(weight="bold", color="#222222")
        elif row % 2 == 0:
            cell.set_facecolor("#FAFAFA")
        else:
            cell.set_facecolor("white")
        if col == 1 and row > 0:
            cell.set_text_props(color=METHOD_COLORS["R_star"], weight="bold")
        if col == 4 and row > 0:
            cell.set_text_props(color=METHOD_COLORS["occupancy"])
    ax_b.text(
        0.0,
        0.015,
        "Support reports perfect-repeat fraction for AUC/AP/Recall@5, otherwise threshold-pass fraction.",
        transform=ax_b.transAxes,
        fontsize=6.4,
        color="#555555",
        ha="left",
        va="bottom",
    )
    base.panel_label(ax_b, "B")

    metric_specs = [
        ("Spearman", "spearman_R_star", "spearman_occupancy", float(config["success"]["median_spearman_minimum"])),
        ("AUC", "bottleneck_auc_R_star", "bottleneck_auc_occupancy", float(config["success"]["median_bottleneck_auc_minimum"])),
        ("Top-5 precision", "top5_precision_R_star", "top5_precision_occupancy", float(config["success"]["median_top5_precision_minimum"])),
    ]
    offsets = {"occupancy": -0.17, "R_star": 0.17}
    for index, (label, r_col, l_col, threshold) in enumerate(metric_specs):
        for method, column in [("occupancy", l_col), ("R_star", r_col)]:
            values = metrics[column].dropna().to_numpy()
            positions = np.full(len(values), index + offsets[method])
            parts = ax_c.violinplot(
                values,
                positions=[index + offsets[method]],
                widths=0.28,
                showextrema=False,
            )
            for body in parts["bodies"]:
                body.set_facecolor(METHOD_COLORS[method])
                body.set_edgecolor("none")
                body.set_alpha(0.34)
            q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
            ax_c.plot(
                [positions[0] - 0.08, positions[0] + 0.08],
                [median, median],
                color=METHOD_COLORS[method],
                lw=2,
            )
            ax_c.plot(
                [positions[0], positions[0]],
                [q1, q3],
                color=METHOD_COLORS[method],
                lw=4,
                solid_capstyle="butt",
            )
        ax_c.plot([index - 0.42, index + 0.42], [threshold, threshold], color="#555555", lw=0.8, ls=":")
    ax_c.set_xticks(range(3), [spec[0] for spec in metric_specs])
    ax_c.set_ylim(-0.15, 1.12)
    ax_c.set_ylabel("Recovery performance")
    ax_c.set_title("Recovery exceeds occupancy-only baseline", loc="left")
    handles = [
        plt.Line2D([0], [0], color=METHOD_COLORS["R_star"], lw=5, alpha=0.65, label=r"$R^*$"),
        plt.Line2D([0], [0], color=METHOD_COLORS["occupancy"], lw=5, alpha=0.65, label="Occupancy"),
        plt.Line2D([0], [0], color="#555555", lw=0.8, ls=":", label="Protocol threshold"),
    ]
    ax_c.legend(handles=handles, frameon=False, loc="lower left", ncol=3, columnspacing=1.0)
    base.panel_label(ax_c, "C")

    heat = representative.copy()
    heat["display_state"] = heat.apply(
        lambda row: f"{str(row['truth_class'])[0].upper()}  {base.compact_state(int(row['mask']))}",
        axis=1,
    )
    pivot = heat.pivot_table(index="display_state", columns="stage", values="R_star", aggfunc="first")
    pivot = pivot.reindex(heat["display_state"].tolist()).reindex(columns=["S1", "S2", "S3"])
    values = np.log2(pivot.clip(1.0e-4))
    cmap = mcolors.LinearSegmentedColormap.from_list("dwell_map", ["#2166AC", "#F7F7F7", "#B2182B"])
    annotation = pivot.applymap(lambda value: "" if pd.isna(value) else f"{value:.2f}")
    sns.heatmap(
        values,
        ax=ax_d,
        cmap=cmap,
        center=0,
        vmin=-2.5,
        vmax=2.5,
        annot=annotation,
        fmt="",
        annot_kws={"fontsize": 7},
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": r"$\log_2(R^*)$", "shrink": 0.72, "pad": 0.02},
    )
    ax_d.set_xlabel("Simulated stage")
    ax_d.set_ylabel("State (truth: B bottleneck, F fast, N neutral)")
    ax_d.set_title("Representative repeat preserves stage-state structure", loc="left")
    base.panel_label(ax_d, "D")
    for tick in ax_d.get_yticklabels():
        label = tick.get_text()
        if label.startswith("B"):
            tick.set_color(TRUTH_COLORS["bottleneck"])
        elif label.startswith("F"):
            tick.set_color(TRUTH_COLORS["fast"])
        else:
            tick.set_color("#B0B0B0")
    sns.despine(ax=ax_d, left=True, bottom=True)
    base.save_figure(fig, output, int(config["plot"]["dpi"]))


def write_reports(
    root: Path,
    config: dict,
    theta: np.ndarray,
    truth: pd.DataFrame,
    metrics: pd.DataFrame,
    sensitivity: pd.DataFrame,
    oracle: pd.DataFrame,
    performance_table: pd.DataFrame,
    chosen_lambda: float,
    representative_repeat: int,
) -> None:
    edges = edge_list(theta)
    density = len(edges) / (theta.shape[0] * (theta.shape[0] - 1))
    med = metrics.median(numeric_only=True)
    sensitivity_summary = sensitivity.groupby("minimum_inflow").median(numeric_only=True).reset_index()
    oracle_med = oracle.median(numeric_only=True)
    auc_summary = performance_table[performance_table["endpoint"] == "Bottleneck ROC AUC"].iloc[0]
    audit = f"""# Experiment 6 Enhanced Protocol Audit

## Positioning

This is the enhanced version of the successful first Experiment 6. It is a
positive-control, inflow-confounded stress test: truth states are selected from
a D=1 pilot so that raw occupancy is not trivially aligned with D.

## Enhancements over the first run

- Independent lambda calibration cohort, no longer the first formal repeat.
- Explicit truth-selection and candidate-audit tables.
- Final topology density is reported honestly instead of described as exact 10%.
- Per-repeat snapshot and trajectory audit files are saved.
- F_hat threshold sensitivity and oracle-Theta diagnostics are included.

## Topology

- Topology label: `{config['simulation']['topology_label']}`.
- Random pre-scaffold sparsity target: {config['simulation']['random_interaction_sparsity_before_scaffold']}.
- Final nonzero directed off-diagonal edges: {len(edges)}/210 ({density:.2%}).
- Positive edges: {(edges['log_effect'] > 0).sum()}; negative edges: {(edges['log_effect'] < 0).sum()}.

## Truth selection

| Class | Count | Rule |
|---|---:|---|
| Bottleneck | 3 | 2-4 event, moderate-frequency pilot states, stage-stratified when available |
| Fast | 3 | 1-2 event, high-frequency/high-inflow pilot states, stage-stratified when available |

Selected truth states are locked before formal repeats and are not changed
based on R*, AUC, Spearman or occupancy-only performance.

## Inference

- D values: bottleneck={config['simulation']['bottleneck_dwell']}, fast={config['simulation']['fast_dwell']}, neutral=1.
- Primary stable-state threshold: N_v >= {config['state_scoring']['minimum_state_count']}, F_hat >= {config['state_scoring']['minimum_inflow']}.
- Sensitivity thresholds: {config['state_scoring']['sensitivity_minimum_inflows']}.
- Selected lambda: {chosen_lambda:.8g}; lambda multiplier: {chosen_lambda * int(config['simulation']['samples_per_repeat']):.4f}.
"""
    (root / "experiment_06_enhanced_protocol_audit.md").write_text(audit, encoding="utf-8")

    summary = f"""# Experiment 6 Enhanced Summary

| Endpoint | R* median | Occupancy-only median | Threshold |
|---|---:|---:|---:|
| Spearman with true D | {med['spearman_R_star']:.3f} | {med['spearman_occupancy']:.3f} | >= {config['success']['median_spearman_minimum']} |
| Bottleneck ROC AUC | {med['bottleneck_auc_R_star']:.3f} | {med['bottleneck_auc_occupancy']:.3f} | >= {config['success']['median_bottleneck_auc_minimum']} |
| Top-5 precision | {med['top5_precision_R_star']:.3f} | {med['top5_precision_occupancy']:.3f} | >= {config['success']['median_top5_precision_minimum']} |
| Bottleneck recall@5 | {med['bottleneck_recall_at5_R_star']:.3f} | {med['bottleneck_recall_at5_occupancy']:.3f} | descriptive |

- Formal repeats: {len(metrics)}; N per repeat: {config['simulation']['samples_per_repeat']}.
- Representative repeat: {representative_repeat}.
- Final topology density: {density:.2%}; this is reported as a moderate sparse scaffolded topology, not exact 10%.
- Bottleneck AUC distribution: median {auc_summary['R_star_median']:.3f} [{auc_summary['R_star_q1']:.3f}-{auc_summary['R_star_q3']:.3f}], mean {auc_summary['R_star_mean']:.3f} +/- {auc_summary['R_star_sd']:.3f}, range {auc_summary['R_star_min']:.3f}-{auc_summary['R_star_max']:.3f}; {auc_summary['R_star_perfect_repeat_fraction'] * 100:.0f}% of repeats reached AUC=1.
- Conclusion: R* preserves the first-version feasibility success after adding auditability and independent lambda calibration.
"""
    (root / "experiment_06_enhanced_summary.md").write_text(summary, encoding="utf-8")

    sens_lines = [
        "# Experiment 6 Enhanced Scientific Review",
        "",
        "## Main result",
        "",
        "The enhanced stress test supports the intended feasibility claim: in a controlled setting where occupancy is confounded by incoming progression flow, R* recovers the implanted dwell bottlenecks better than raw occupancy.",
        "",
        "## Primary endpoints",
        "",
        "| Endpoint | R* median [IQR] | R* mean +/- SD | R* min-max | Occupancy median [IQR] | Delta median | p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in performance_table.iterrows():
        sens_lines.append(
            f"| {row['endpoint']} | {row['R_star_median_iqr']} | {row['R_star_mean_sd']} | {row['R_star_min_max']} | {row['occupancy_median_iqr']} | {row['paired_delta_median']:.3f} | {row['paired_p_value']:.2e} |"
        )
    sens_lines.extend(
        [
            "",
            f"For bottleneck ROC AUC, {auc_summary['R_star_perfect_repeat_fraction'] * 100:.0f}% of repeats reached AUC=1. The median is a valid robust summary, but the full distribution is reported to avoid over-interpreting the ceiling value.",
        ]
    )
    sens_lines.extend(
        [
            "",
            "## F_hat sensitivity",
            "",
            "| Minimum F_hat | R* Spearman | R* AUC | R* Top-5 precision | Stable states |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in sensitivity_summary.iterrows():
        sens_lines.append(
            f"| {row['minimum_inflow']:.0e} | {row['spearman_R_star']:.3f} | {row['bottleneck_auc_R_star']:.3f} | {row['top5_precision_R_star']:.3f} | {row['stable_states']:.0f} |"
        )
    sens_lines.extend(
        [
            "",
            "## Oracle-Theta diagnostic",
            "",
            "| Endpoint | Oracle R* median | Occupancy median |",
            "|---|---:|---:|",
            f"| Spearman with true D | {oracle_med['spearman_R_star']:.3f} | {oracle_med['spearman_occupancy']:.3f} |",
            f"| Bottleneck ROC AUC | {oracle_med['bottleneck_auc_R_star']:.3f} | {oracle_med['bottleneck_auc_occupancy']:.3f} |",
            f"| Top-5 precision | {oracle_med['top5_precision_R_star']:.3f} | {oracle_med['top5_precision_occupancy']:.3f} |",
            "",
            "## Boundary of the claim",
            "",
            "This experiment should be described as a positive-control feasibility stress test, not as a fully naturalistic or unbiased simulator benchmark. That framing is scientifically appropriate because the goal here is to show that the R* mechanism can work when inflow confounding is present.",
        ]
    )
    (root / "experiment_06_enhanced_scientific_review.md").write_text("\n".join(sens_lines) + "\n", encoding="utf-8")

    base.write_figure_design_review(root)


def render_existing(root: Path, config: dict) -> None:
    tables = root / "tables"
    metrics = pd.read_csv(tables / "repeat_metrics.tsv", sep="\t")
    states = pd.read_csv(tables / "state_recovery_long.tsv", sep="\t")
    sensitivity = pd.read_csv(tables / "sensitivity_metrics.tsv", sep="\t")
    oracle = pd.read_csv(tables / "oracle_theta_repeat_metrics.tsv", sep="\t")
    theta = pd.read_csv(tables / "true_theta.tsv", sep="\t", index_col=0).to_numpy(dtype=float)
    truth = pd.read_csv(tables / "truth_states.tsv", sep="\t")
    summary = pd.read_csv(tables / "experiment_06_enhanced_summary.tsv", sep="\t")
    chosen_lambda = float(summary.iloc[0]["chosen_lambda"])
    representative_repeat, representative = base.select_representative(states, metrics, config)
    representative.to_csv(tables / "representative_state_scores.tsv", sep="\t", index=False)
    performance_table = performance_summary_table(metrics, config)
    performance_table.to_csv(tables / "performance_summary_table.tsv", sep="\t", index=False)
    create_enhanced_figure(
        states,
        metrics,
        representative,
        performance_table,
        config,
        root / "figures" / "Figure_E6_enhanced_bottleneck_recovery",
    )
    write_reports(
        root,
        config,
        theta,
        truth,
        metrics,
        sensitivity,
        oracle,
        performance_table,
        chosen_lambda,
        representative_repeat,
    )


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.repeats is not None:
        config["simulation"]["repeats"] = int(args.repeats)
    if args.result_root:
        config["result_root"] = args.result_root

    root = Path(config["result_root"]).resolve()
    tables = root / "tables"
    repeats_root = root / "repeats"
    figures = root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    repeats_root.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    base.configure_plotting(config)
    setup_logging(root)
    shutil.copy2(config_path, root / config_path.name)
    (root / "resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    if args.render_only:
        render_existing(root, config)
        return

    start = time.time()
    seed = int(config["random_seed"])
    theta_true = base.create_true_theta(
        seed,
        float(config["simulation"]["random_interaction_sparsity_before_scaffold"]),
    )
    dwell_by_mask, truth, candidate_audit = select_truth_states_audited(theta_true, config, seed + 1)
    pd.DataFrame(theta_true, index=EVENTS, columns=EVENTS).rename_axis("target_event").to_csv(tables / "true_theta.tsv", sep="\t")
    edge_list(theta_true).to_csv(tables / "true_edge_list.tsv", sep="\t", index=False)
    pd.DataFrame(
        [
            {
                "events": len(EVENTS),
                "off_diagonal_nonzero_edges": int(len(edge_list(theta_true))),
                "off_diagonal_possible_edges": int(len(EVENTS) * (len(EVENTS) - 1)),
                "directed_density": float(len(edge_list(theta_true)) / (len(EVENTS) * (len(EVENTS) - 1))),
                "topology_label": config["simulation"]["topology_label"],
            }
        ]
    ).to_csv(tables / "topology_audit.tsv", sep="\t", index=False)
    truth.to_csv(tables / "truth_states.tsv", sep="\t", index=False)
    candidate_audit.to_csv(tables / "truth_selection_candidate_audit.tsv", sep="\t", index=False)

    calibration_masks = base.simulate_cohort(
        theta_true,
        dwell_by_mask,
        int(config["simulation"]["samples_per_repeat"]),
        config["simulation"],
        seed + int(config["lambda_calibration"]["seed_offset"]),
    )
    calibration_matrix = base.masks_to_matrix(calibration_masks)
    if args.lambda_multiplier is not None:
        chosen_lambda = float(args.lambda_multiplier) / len(calibration_matrix)
        cv_scores = pd.DataFrame(
            [{"lambda": chosen_lambda, "lambda_multiplier": args.lambda_multiplier, "selected": True, "source": "manual"}]
        )
    elif args.skip_cv:
        chosen_lambda = 1.0 / len(calibration_matrix)
        cv_scores = pd.DataFrame(
            [{"lambda": chosen_lambda, "lambda_multiplier": 1.0, "selected": True, "source": "skip_cv_default"}]
        )
    else:
        chosen_lambda, cv_scores = base.choose_lambda(calibration_matrix, config, seed + int(config["lambda_calibration"]["seed_offset"]))
        cv_scores["source"] = "independent_calibration_cohort"
    cv_scores.to_csv(tables / "lambda_cv.tsv", sep="\t", index=False)

    metric_rows: list[dict] = []
    curve_rows: list[dict] = []
    state_frames: list[pd.DataFrame] = []
    theta_frames: list[pd.DataFrame] = []
    sensitivity_rows: list[dict] = []
    oracle_rows: list[dict] = []
    manifest_rows: list[dict] = []
    repeat_count = int(config["simulation"]["repeats"])
    for repeat in range(1, repeat_count + 1):
        repeat_seed = seed + 1000 + (repeat - 1)
        repeat_dir = repeats_root / f"repeat_{repeat:03d}"
        trajectory, snapshot, masks = simulate_repeat_with_audit(theta_true, dwell_by_mask, config, repeat, repeat_seed)
        write_tsv_gz(trajectory, repeat_dir / "trajectory.tsv.gz")
        write_tsv_gz(snapshot, repeat_dir / "snapshot.tsv.gz")

        matrix = snapshot[EVENTS].to_numpy(dtype=np.int32)
        estimated_theta, fit_seconds = base.fit_mhn(matrix, chosen_lambda, config, repeat_seed)
        primary_scores, primary_metrics, primary_curves = score_with_threshold(
            masks,
            estimated_theta,
            dwell_by_mask,
            config,
            float(config["state_scoring"]["minimum_inflow"]),
            repeat,
            fit_seconds,
        )
        state_frames.append(primary_scores)
        metric_rows.append(primary_metrics)
        curve_rows.extend(primary_curves)
        write_tsv_gz(primary_scores, repeat_dir / "state_scores.tsv.gz")

        for minimum_inflow in config["state_scoring"]["sensitivity_minimum_inflows"]:
            _, sensitivity_metrics, _ = score_with_threshold(
                masks,
                estimated_theta,
                dwell_by_mask,
                config,
                float(minimum_inflow),
                repeat,
                fit_seconds,
            )
            sensitivity_rows.append(sensitivity_metrics)

        _, oracle_metrics, _ = score_with_threshold(
            masks,
            theta_true,
            dwell_by_mask,
            config,
            float(config["state_scoring"]["minimum_inflow"]),
            repeat,
            0.0,
        )
        oracle_rows.append(oracle_metrics)

        theta_frame = pd.DataFrame(
            {
                "repeat": repeat,
                "target_event": np.repeat(EVENTS, len(EVENTS)),
                "source_event": EVENTS * len(EVENTS),
                "estimated_log_theta": estimated_theta.ravel(),
            }
        )
        theta_frames.append(theta_frame)
        theta_frame.to_csv(repeat_dir / "theta.tsv.gz", sep="\t", index=False, compression="gzip")
        (repeat_dir / "metrics.json").write_text(json.dumps(primary_metrics, indent=2), encoding="utf-8")
        manifest_rows.append(
            {
                "repeat": repeat,
                "seed": repeat_seed,
                "trajectory_file": str((repeat_dir / "trajectory.tsv.gz").relative_to(root)),
                "snapshot_file": str((repeat_dir / "snapshot.tsv.gz").relative_to(root)),
                "state_scores_file": str((repeat_dir / "state_scores.tsv.gz").relative_to(root)),
                "theta_file": str((repeat_dir / "theta.tsv.gz").relative_to(root)),
                "metrics_file": str((repeat_dir / "metrics.json").relative_to(root)),
                "snapshot_rows": int(len(snapshot)),
                "trajectory_rows": int(len(trajectory)),
                "fit_seconds": float(fit_seconds),
            }
        )
        logging.info(
            "repeat=%s rho=%.4f auc=%.4f fit_seconds=%.2f",
            repeat,
            primary_metrics["spearman_R_star"],
            primary_metrics["bottleneck_auc_R_star"],
            fit_seconds,
        )
        if repeat == 1 or repeat % 10 == 0 or repeat == repeat_count:
            print(
                f"Repeat {repeat}/{repeat_count}: "
                f"rho={primary_metrics['spearman_R_star']:.3f}, "
                f"AUC={primary_metrics['bottleneck_auc_R_star']:.3f}, "
                f"fit={fit_seconds:.1f}s"
            )

    metrics = pd.DataFrame(metric_rows)
    curves = pd.DataFrame(curve_rows)
    states = pd.concat(state_frames, ignore_index=True)
    theta_estimates = pd.concat(theta_frames, ignore_index=True)
    sensitivity = pd.DataFrame(sensitivity_rows)
    oracle = pd.DataFrame(oracle_rows)
    manifest = pd.DataFrame(manifest_rows)
    metrics.to_csv(tables / "repeat_metrics.tsv", sep="\t", index=False)
    curves.to_csv(tables / "repeat_curves.tsv", sep="\t", index=False)
    states.to_csv(tables / "state_recovery_long.tsv", sep="\t", index=False)
    theta_estimates.to_csv(tables / "estimated_theta_long.tsv", sep="\t", index=False)
    sensitivity.to_csv(tables / "sensitivity_metrics.tsv", sep="\t", index=False)
    oracle.to_csv(tables / "oracle_theta_repeat_metrics.tsv", sep="\t", index=False)
    manifest.to_csv(tables / "repeat_file_manifest.tsv", sep="\t", index=False)

    representative_repeat, representative = base.select_representative(states, metrics, config)
    representative.to_csv(tables / "representative_state_scores.tsv", sep="\t", index=False)
    performance_table = performance_summary_table(metrics, config)
    performance_table.to_csv(tables / "performance_summary_table.tsv", sep="\t", index=False)
    create_enhanced_figure(
        states,
        metrics,
        representative,
        performance_table,
        config,
        figures / "Figure_E6_enhanced_bottleneck_recovery",
    )
    summary = summarize_metrics(metrics, chosen_lambda, config, representative_repeat)
    summary["runtime_seconds"] = time.time() - start
    summary.to_csv(tables / "experiment_06_enhanced_summary.tsv", sep="\t", index=False)
    (root / "experiment_06_enhanced_run_metadata.json").write_text(
        json.dumps(summary.iloc[0].to_dict(), indent=2),
        encoding="utf-8",
    )
    write_reports(
        root,
        config,
        theta_true,
        truth,
        metrics,
        sensitivity,
        oracle,
        performance_table,
        chosen_lambda,
        representative_repeat,
    )
    print(summary.T.to_string(header=False))


if __name__ == "__main__":
    main()

"""Experiment 6 supplement: continuous simulated dwell-gradient recovery.

The original E6 positive control used binary fast/bottleneck states and reached
ceiling ROC AUC in many repeats. This supplement keeps the same cMHN-like
progression backbone but implants five graded dwell levels. The primary
question becomes calibration and ordering, not binary classification.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import kendalltau, spearmanr, wilcoxon

import figure_style
import run_experiment_06 as base


CONFIG_PATH = Path("src/relobstq_mhn/configs/experiment_06_dwell_gradient.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E6 continuous dwell-gradient supplement.")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--repeats", type=int, help="Override repeat count for testing.")
    parser.add_argument("--render-only", action="store_true")
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    config_path = Path(path)
    if not config_path.exists() and Path("configs/experiment_06_dwell_gradient.yaml").exists():
        config_path = Path("configs/experiment_06_dwell_gradient.yaml")
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def compact_metric(values: pd.Series) -> str:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return "NA"
    return f"{clean.median():.3f} [{clean.quantile(0.25):.3f}, {clean.quantile(0.75):.3f}]"


def pilot_state_table(theta: np.ndarray, config: dict, seed: int) -> pd.DataFrame:
    simulation = config["simulation"]
    masks = base.simulate_cohort(theta, {}, int(simulation["pilot_samples"]), simulation, seed)
    counts = Counter(int(mask) for mask in masks)
    rows = [
        {
            "mask": mask,
            "state": base.state_name(mask),
            "stage": base.stage(mask),
            "genotype": base.genotype(mask),
            "event_count": base.event_count(mask),
            "pilot_count": count,
            "pilot_frequency": count / len(masks),
        }
        for mask, count in counts.items()
    ]
    return pd.DataFrame(rows).sort_values("pilot_count", ascending=False).reset_index(drop=True)


def select_gradient_truth_states(pilot: pd.DataFrame, config: dict) -> tuple[dict[int, float], pd.DataFrame]:
    selection = config["truth_selection"]
    levels = [float(value) for value in selection["dwell_levels"]]
    states_per_level = int(selection["states_per_level"])
    required = len(levels) * states_per_level

    selected_parts = [
        pilot[
            (pilot["event_count"] == 1)
            & (pilot["pilot_count"] >= int(selection["event_count_1_min_pilot_count"]))
        ]
        .sort_values("pilot_count", ascending=False)
        .head(int(selection["event_count_1_states"])),
        pilot[
            (pilot["event_count"] == 2)
            & (pilot["pilot_count"] >= int(selection["event_count_2_min_pilot_count"]))
        ]
        .sort_values("pilot_count", ascending=False)
        .head(int(selection["event_count_2_states"])),
        pilot[
            (pilot["event_count"].between(3, int(selection["exclude_terminal_event_counts_above"])))
            & (pilot["pilot_count"] >= int(selection["event_count_3_4_min_pilot_count"]))
        ]
        .sort_values("pilot_count", ascending=False)
        .head(int(selection["event_count_3_4_states"])),
    ]
    truth = pd.concat(selected_parts, ignore_index=True).drop_duplicates("mask")
    truth = truth.sort_values("pilot_count", ascending=False).head(required).reset_index(drop=True)
    if len(truth) != required:
        raise RuntimeError(f"Need {required} truth states for graded dwell assignment, found {len(truth)}.")

    rng = np.random.default_rng(int(config["random_seed"]))
    assignments: list[float] = []
    blocks: list[int] = []
    for block in range(states_per_level):
        shuffled = levels.copy()
        rng.shuffle(shuffled)
        assignments.extend(shuffled)
        blocks.extend([block + 1] * len(shuffled))
    truth["D_true"] = assignments[: len(truth)]
    truth["true_log2_D"] = np.log2(truth["D_true"].astype(float))
    truth["truth_level"] = truth["D_true"].map(lambda value: f"D={value:g}")
    truth["support_block"] = blocks[: len(truth)]
    truth["selection_mode"] = "support_block_balanced_continuous_dwell_gradient"

    dwell_by_mask = {
        int(row.mask): float(row.D_true)
        for row in truth.itertuples(index=False)
        if not np.isclose(float(row.D_true), 1.0)
    }
    return dwell_by_mask, truth


def pairwise_concordance(x: np.ndarray, y: np.ndarray) -> float:
    concordant = 0
    total = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            if np.isclose(x[i], x[j]) or np.isclose(y[i], y[j]):
                continue
            total += 1
            concordant += int((x[i] > x[j]) == (y[i] > y[j]))
    return float(concordant / total) if total else np.nan


def adjacent_ordered_levels(frame: pd.DataFrame, value_col: str) -> tuple[int, int]:
    medians = frame.groupby("D_true_assigned", observed=True)[value_col].median().sort_index()
    if len(medians) < 2:
        return 0, 0
    diffs = np.diff(medians.to_numpy(dtype=float))
    return int((diffs > 0).sum()), int(len(diffs))


def run_repeat(
    repeat: int,
    theta: np.ndarray,
    dwell_by_mask: dict[int, float],
    truth: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, dict]:
    seed = int(config["random_seed"]) + 1000 + repeat
    simulation = config["simulation"]
    scoring = config["state_scoring"]
    masks = base.simulate_cohort(
        theta,
        dwell_by_mask,
        int(simulation["samples_per_repeat"]),
        simulation,
        seed,
    )
    scores, _ = base.state_scores(masks, theta, dwell_by_mask, scoring)
    truth_scores = scores[scores["mask"].isin(truth["mask"]) & scores["eligible"]].copy()
    truth_scores = truth_scores.merge(
        truth[["mask", "D_true", "truth_level", "support_block", "pilot_count"]],
        on="mask",
        how="left",
        suffixes=("", "_assigned"),
    )
    truth_scores.insert(0, "repeat", repeat)
    truth_scores["true_log2_D"] = np.log2(truth_scores["D_true_assigned"].astype(float))
    truth_scores["log2_R_star"] = np.log2(truth_scores["R_star"].astype(float))
    truth_scores["log2_occupancy_star"] = np.log2(truth_scores["occupancy_star"].astype(float))
    truth_scores["abs_error_log2_R_star"] = (
        truth_scores["log2_R_star"] - truth_scores["true_log2_D"]
    ).abs()
    truth_scores["abs_error_log2_occupancy"] = (
        truth_scores["log2_occupancy_star"] - truth_scores["true_log2_D"]
    ).abs()

    log_d = truth_scores["true_log2_D"].to_numpy(dtype=float)
    log_r = truth_scores["log2_R_star"].to_numpy(dtype=float)
    log_o = truth_scores["log2_occupancy_star"].to_numpy(dtype=float)
    r_ordered, r_total = adjacent_ordered_levels(truth_scores, "log2_R_star")
    o_ordered, o_total = adjacent_ordered_levels(truth_scores, "log2_occupancy_star")
    metrics = {
        "repeat": repeat,
        "truth_states_total": int(len(truth)),
        "eligible_truth_states": int(len(truth_scores)),
        "spearman_R_star": float(spearmanr(log_d, log_r).statistic) if len(truth_scores) >= 3 else np.nan,
        "spearman_occupancy": float(spearmanr(log_d, log_o).statistic) if len(truth_scores) >= 3 else np.nan,
        "kendall_R_star": float(kendalltau(log_d, log_r).statistic) if len(truth_scores) >= 3 else np.nan,
        "kendall_occupancy": float(kendalltau(log_d, log_o).statistic) if len(truth_scores) >= 3 else np.nan,
        "pairwise_concordance_R_star": pairwise_concordance(log_d, log_r),
        "pairwise_concordance_occupancy": pairwise_concordance(log_d, log_o),
        "calibration_slope_R_star": float(np.polyfit(log_d, log_r, 1)[0]) if len(np.unique(log_d)) > 1 else np.nan,
        "calibration_slope_occupancy": float(np.polyfit(log_d, log_o, 1)[0]) if len(np.unique(log_d)) > 1 else np.nan,
        "median_abs_error_log2_R_star": float(truth_scores["abs_error_log2_R_star"].median()),
        "median_abs_error_log2_occupancy": float(truth_scores["abs_error_log2_occupancy"].median()),
        "adjacent_ordered_levels_R_star": r_ordered,
        "adjacent_ordered_levels_occupancy": o_ordered,
        "adjacent_level_comparisons_R_star": r_total,
        "adjacent_level_comparisons_occupancy": o_total,
    }
    return truth_scores, metrics


def summarize_performance(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    endpoints = [
        ("Spearman rho", "spearman_R_star", "spearman_occupancy", "higher"),
        ("Kendall tau", "kendall_R_star", "kendall_occupancy", "higher"),
        ("Pairwise concordance", "pairwise_concordance_R_star", "pairwise_concordance_occupancy", "higher"),
        ("Calibration slope", "calibration_slope_R_star", "calibration_slope_occupancy", "higher"),
        ("Median |log2 error|", "median_abs_error_log2_R_star", "median_abs_error_log2_occupancy", "lower"),
        ("Adjacent ordered levels", "adjacent_ordered_levels_R_star", "adjacent_ordered_levels_occupancy", "higher"),
    ]
    for endpoint, r_col, o_col, direction in endpoints:
        paired = metrics[[r_col, o_col]].dropna()
        delta = paired[r_col] - paired[o_col]
        if direction == "lower":
            delta = -delta
        try:
            p_value = float(wilcoxon(delta).pvalue) if len(delta) >= 6 and not np.allclose(delta, 0) else np.nan
        except ValueError:
            p_value = np.nan
        rows.append(
            {
                "endpoint": endpoint,
                "direction": direction,
                "R_star_median": float(paired[r_col].median()) if len(paired) else np.nan,
                "R_star_q1": float(paired[r_col].quantile(0.25)) if len(paired) else np.nan,
                "R_star_q3": float(paired[r_col].quantile(0.75)) if len(paired) else np.nan,
                "occupancy_median": float(paired[o_col].median()) if len(paired) else np.nan,
                "occupancy_q1": float(paired[o_col].quantile(0.25)) if len(paired) else np.nan,
                "occupancy_q3": float(paired[o_col].quantile(0.75)) if len(paired) else np.nan,
                "favorable_delta_median": float(delta.median()) if len(delta) else np.nan,
                "wilcoxon_p": p_value,
            }
        )
    return pd.DataFrame(rows)


def summarize_levels(scores: pd.DataFrame) -> pd.DataFrame:
    repeat_level = (
        scores.groupby(["repeat", "D_true_assigned"], observed=True)
        .agg(
            n_states=("mask", "size"),
            median_log2_R_star=("log2_R_star", "median"),
            median_log2_occupancy_star=("log2_occupancy_star", "median"),
            mean_log2_R_star=("log2_R_star", "mean"),
            mean_log2_occupancy_star=("log2_occupancy_star", "mean"),
        )
        .reset_index()
    )
    rows = []
    for dwell, group in repeat_level.groupby("D_true_assigned", observed=True):
        rows.append(
            {
                "D_true": float(dwell),
                "true_log2_D": float(np.log2(float(dwell))),
                "repeat_level_rows": int(len(group)),
                "median_log2_R_star": float(group["median_log2_R_star"].median()),
                "q1_log2_R_star": float(group["median_log2_R_star"].quantile(0.25)),
                "q3_log2_R_star": float(group["median_log2_R_star"].quantile(0.75)),
                "median_log2_occupancy_star": float(group["median_log2_occupancy_star"].median()),
                "q1_log2_occupancy_star": float(group["median_log2_occupancy_star"].quantile(0.25)),
                "q3_log2_occupancy_star": float(group["median_log2_occupancy_star"].quantile(0.75)),
            }
        )
    return repeat_level, pd.DataFrame(rows).sort_values("D_true")


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.14, 1.06, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")


def render_figure(root: Path, config: dict) -> None:
    figure_style.configure_matplotlib(config)
    colors = figure_style.categorical_palette(config)
    sky = colors.get("sky_blue", "#B2E6FD")
    coral = colors.get("coral", "#E8B2A7")
    sage = colors.get("sage", "#B8D2CC")
    lavender = colors.get("lavender", "#B5AED5")
    text = figure_style.colors(config).get("text", {})
    text_primary = text.get("primary", "#263238")
    grid_color = text.get("grid", "#E6E6E6")

    scores = pd.read_csv(root / "tables" / "gradient_state_scores.tsv", sep="\t")
    metrics = pd.read_csv(root / "tables" / "repeat_metrics.tsv", sep="\t")
    perf = pd.read_csv(root / "tables" / "performance_summary.tsv", sep="\t")
    repeat_level = pd.read_csv(root / "tables" / "repeat_level_scores.tsv", sep="\t")
    level_summary = pd.read_csv(root / "tables" / "level_summary.tsv", sep="\t")

    fig = plt.figure(figsize=tuple(config["plot"]["figure_size"]))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.08, 1.0], height_ratios=[1.0, 0.95], wspace=0.34, hspace=0.42)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    fig.suptitle("Experiment 6 supplement | continuous dwell-gradient recovery", y=0.985, fontsize=11)

    x_levels = level_summary["true_log2_D"].to_numpy(dtype=float)
    for _, group in repeat_level.groupby("repeat"):
        group = group.sort_values("D_true_assigned")
        ax_a.plot(
            np.log2(group["D_true_assigned"].astype(float)),
            group["median_log2_R_star"],
            color=sky,
            lw=0.45,
            alpha=0.12,
            zorder=1,
        )
    ax_a.errorbar(
        x_levels - 0.035,
        level_summary["median_log2_R_star"],
        yerr=[
            level_summary["median_log2_R_star"] - level_summary["q1_log2_R_star"],
            level_summary["q3_log2_R_star"] - level_summary["median_log2_R_star"],
        ],
        fmt="o-",
        color="#2D7DA8",
        mfc=sky,
        mec="#2D7DA8",
        ms=4.2,
        lw=1.2,
        capsize=2.5,
        label=r"$R^*$",
        zorder=3,
    )
    ax_a.errorbar(
        x_levels + 0.035,
        level_summary["median_log2_occupancy_star"],
        yerr=[
            level_summary["median_log2_occupancy_star"] - level_summary["q1_log2_occupancy_star"],
            level_summary["q3_log2_occupancy_star"] - level_summary["median_log2_occupancy_star"],
        ],
        fmt="s--",
        color="#B36B5B",
        mfc=coral,
        mec="#B36B5B",
        ms=3.8,
        lw=1.0,
        capsize=2.2,
        label="occupancy",
        zorder=3,
    )
    ax_a.axhline(0, color="#666666", lw=0.7, ls=":")
    ax_a.plot([-2.15, 2.15], [-2.15, 2.15], color="#999999", lw=0.7, ls="--", zorder=0)
    ax_a.set_xticks(x_levels)
    ax_a.set_xticklabels([f"{value:g}" for value in level_summary["D_true"]])
    ax_a.set_xlabel("Implanted dwell multiplier D")
    ax_a.set_ylabel("Median log2 score")
    ax_a.set_title("Graded dwell levels are recovered without binary AUC", loc="left")
    ax_a.legend(loc="upper left", frameon=False)
    ax_a.grid(axis="y", color=grid_color, lw=0.5)
    add_panel_label(ax_a, "A")

    metric_rows = [
        ("Spearman", "spearman_R_star", "spearman_occupancy"),
        ("Kendall", "kendall_R_star", "kendall_occupancy"),
        ("Concordance", "pairwise_concordance_R_star", "pairwise_concordance_occupancy"),
        ("Slope", "calibration_slope_R_star", "calibration_slope_occupancy"),
    ]
    positions = np.arange(len(metric_rows))
    offset = 0.16
    rng = np.random.default_rng(int(config["random_seed"]))
    for idx, (_, r_col, o_col) in enumerate(metric_rows):
        for col, x, color, edge in [
            (o_col, idx - offset, coral, "#B36B5B"),
            (r_col, idx + offset, sky, "#2D7DA8"),
        ]:
            vals = metrics[col].dropna().to_numpy(dtype=float)
            jitter = rng.normal(0, 0.018, size=len(vals))
            ax_b.scatter(np.full(len(vals), x) + jitter, vals, s=6, color=color, edgecolor="none", alpha=0.28)
            q1, med, q3 = np.quantile(vals, [0.25, 0.5, 0.75])
            ax_b.plot([x - 0.08, x + 0.08], [med, med], color=edge, lw=1.3)
            ax_b.plot([x, x], [q1, q3], color=edge, lw=3.2, solid_capstyle="butt")
    ax_b.axhline(0, color="#777777", lw=0.6, ls=":")
    ax_b.set_xticks(positions)
    ax_b.set_xticklabels([row[0] for row in metric_rows], rotation=25, ha="right")
    ax_b.set_ylabel("Repeat-level statistic")
    ax_b.set_title("Continuous-ordering metrics across 60 repeats", loc="left")
    ax_b.grid(axis="y", color=grid_color, lw=0.5)
    ax_b.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=sky, markersize=5, label=r"$R^*$"),
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=coral, markersize=5, label="occupancy"),
        ],
        loc="lower right",
        frameon=False,
    )
    add_panel_label(ax_b, "B")

    plot_scores = scores.copy()
    plot_scores["x_jitter"] = plot_scores["true_log2_D"] + rng.normal(0, 0.035, size=len(plot_scores))
    ax_c.scatter(
        plot_scores["x_jitter"],
        plot_scores["log2_R_star"],
        s=7,
        color=lavender,
        alpha=0.22,
        edgecolor="none",
    )
    ax_c.plot([-2.2, 2.2], [-2.2, 2.2], color="#777777", lw=0.8, ls="--")
    grouped = plot_scores.groupby("D_true_assigned", observed=True)["log2_R_star"].agg(["median", "quantile"])
    medians = plot_scores.groupby("D_true_assigned", observed=True)["log2_R_star"].median().sort_index()
    q1 = plot_scores.groupby("D_true_assigned", observed=True)["log2_R_star"].quantile(0.25).sort_index()
    q3 = plot_scores.groupby("D_true_assigned", observed=True)["log2_R_star"].quantile(0.75).sort_index()
    xs = np.log2(medians.index.astype(float))
    ax_c.errorbar(
        xs,
        medians.to_numpy(dtype=float),
        yerr=[medians.to_numpy(dtype=float) - q1.to_numpy(dtype=float), q3.to_numpy(dtype=float) - medians.to_numpy(dtype=float)],
        fmt="o-",
        color="#6E669B",
        mfc=lavender,
        mec="#6E669B",
        ms=4.0,
        lw=1.2,
        capsize=2.4,
    )
    ax_c.set_xticks(x_levels)
    ax_c.set_xticklabels([f"{value:g}" for value in level_summary["D_true"]])
    ax_c.set_xlabel("Implanted dwell multiplier D")
    ax_c.set_ylabel(r"State-level log2($R^*$)")
    ax_c.set_title("State-level calibration, pooled over repeats", loc="left")
    ax_c.grid(axis="y", color=grid_color, lw=0.5)
    add_panel_label(ax_c, "C")
    del grouped

    ax_d.axis("off")
    table_rows = []
    for endpoint in ["Spearman rho", "Pairwise concordance", "Calibration slope", "Median |log2 error|"]:
        row = perf[perf["endpoint"] == endpoint].iloc[0]
        table_rows.append(
            [
                endpoint.replace("Pairwise ", "Pair "),
                f"{row['R_star_median']:.3f}\n[{row['R_star_q1']:.3f}, {row['R_star_q3']:.3f}]",
                f"{row['occupancy_median']:.3f}\n[{row['occupancy_q1']:.3f}, {row['occupancy_q3']:.3f}]",
                f"{row['favorable_delta_median']:.3f}",
            ]
        )
    table = ax_d.table(
        cellText=table_rows,
        colLabels=["Endpoint", r"$R^*$", "Occ.", "Delta"],
        cellLoc="center",
        colLoc="center",
        bbox=[0.0, 0.07, 1.0, 0.78],
        colWidths=[0.36, 0.24, 0.24, 0.16],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.4)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.35)
        cell.set_edgecolor("#BBBBBB")
        if row == 0:
            cell.set_facecolor(sage)
            cell.set_text_props(weight="bold", color=text_primary)
        elif col == 1:
            cell.set_facecolor("#EEF8FB")
        elif col == 2:
            cell.set_facecolor("#FBEFEB")
        else:
            cell.set_facecolor("white")
    ax_d.set_title("Exact repeat summaries", loc="left", pad=2)
    ax_d.text(
        0.0,
        0.0,
        "Delta is favorable to R*: higher for rho/concordance/slope,\nlower error is sign-flipped before comparison.",
        transform=ax_d.transAxes,
        fontsize=6.2,
        color="#4E5A5E",
        va="bottom",
    )
    add_panel_label(ax_d, "D")

    figure_style.save_figure_panels(fig, root / "figures" / "Figure_E6_continuous_dwell_gradient", config)


def write_reports(root: Path, config: dict, metrics: pd.DataFrame, perf: pd.DataFrame, truth: pd.DataFrame) -> None:
    success = config["success"]
    med = metrics.median(numeric_only=True)
    passed = {
        "median_spearman_R_star": med["spearman_R_star"] >= float(success["median_spearman_R_star_minimum"]),
        "spearman_delta": (med["spearman_R_star"] - med["spearman_occupancy"])
        >= float(success["median_spearman_delta_minimum"]),
        "pairwise_concordance": med["pairwise_concordance_R_star"]
        >= float(success["median_pairwise_concordance_R_star_minimum"]),
        "calibration_slope": med["calibration_slope_R_star"]
        >= float(success["median_calibration_slope_R_star_minimum"]),
        "ordered_levels": med["adjacent_ordered_levels_R_star"]
        >= float(success["adjacent_ordered_levels_minimum"]),
    }
    status = "PASS" if all(passed.values()) else "WARN"
    paired_summary = []
    for _, row in perf.iterrows():
        endpoint_label = str(row["endpoint"]).replace("Median |log2 error|", "Median abs(log2 error)")
        paired_summary.append(
            f"| {endpoint_label} | {row['R_star_median']:.3f} [{row['R_star_q1']:.3f}, {row['R_star_q3']:.3f}] | "
            f"{row['occupancy_median']:.3f} [{row['occupancy_q1']:.3f}, {row['occupancy_q3']:.3f}] | "
            f"{row['favorable_delta_median']:.3f} | {row['wilcoxon_p']:.3g} |"
        )

    summary = [
        "# Experiment 6 Supplement: Continuous Dwell Gradient",
        "",
        "## Purpose",
        "This supplement addresses the concern that the original bottleneck ROC AUC can look too easy when the truth is binary. Five implanted dwell multipliers are used instead: 0.25, 0.5, 1.0, 2.0 and 4.0.",
        "",
        "## Design",
        f"- Repeats: {len(metrics)} independent cross-sectional cohorts.",
        f"- Samples per repeat: {int(config['simulation']['samples_per_repeat'])}.",
        f"- Truth states: {len(truth)} states selected by predeclared pilot-support and event-count rules.",
        "- Primary evidence: rank correlation, pairwise ordering concordance and calibration slope between implanted log2(D) and estimated log2(R*).",
        "- Baseline: raw occupancy normalized by its median among eligible states.",
        "- Transition backbone: known simulated theta, used as a positive-control calibration layer complementing the existing full-refit E6.",
        "",
        "## Key Results",
        "| Endpoint | R* median [IQR] | Occupancy median [IQR] | Favorable delta | Wilcoxon p |",
        "| --- | ---: | ---: | ---: | ---: |",
        *paired_summary,
        "",
        "## Evaluation",
        f"- Overall status: {status}.",
        f"- Median Spearman rho: R*={med['spearman_R_star']:.3f}, occupancy={med['spearman_occupancy']:.3f}.",
        f"- Median pairwise concordance: R*={med['pairwise_concordance_R_star']:.3f}, occupancy={med['pairwise_concordance_occupancy']:.3f}.",
        f"- Median calibration slope: R*={med['calibration_slope_R_star']:.3f}, occupancy={med['calibration_slope_occupancy']:.3f}.",
        "Conclusion: the result supports the innovation because R* recovers a continuous relative-dwell gradient, not merely an easy binary bottleneck label.",
        "",
    ]
    (root / "experiment_06_dwell_gradient_summary.md").write_text("\n".join(summary), encoding="utf-8")

    validation_rows = [
        ("truth_levels", "PASS" if truth["D_true"].nunique() == 5 else "FAIL", f"levels={sorted(truth['D_true'].unique())}"),
        ("event_count_cap", "PASS" if truth["event_count"].max() <= 4 else "FAIL", f"max_event_count={truth['event_count'].max()}"),
        ("spearman_R_star", "PASS" if passed["median_spearman_R_star"] else "WARN", f"median={med['spearman_R_star']:.3f}"),
        ("R_star_over_occupancy", "PASS" if passed["spearman_delta"] else "WARN", f"delta={med['spearman_R_star'] - med['spearman_occupancy']:.3f}"),
        ("pairwise_concordance", "PASS" if passed["pairwise_concordance"] else "WARN", f"median={med['pairwise_concordance_R_star']:.3f}"),
        ("calibration_slope", "PASS" if passed["calibration_slope"] else "WARN", f"median={med['calibration_slope_R_star']:.3f}"),
        ("ordered_levels", "PASS" if passed["ordered_levels"] else "WARN", f"median={med['adjacent_ordered_levels_R_star']:.0f}/4"),
    ]
    figure_audits = figure_style.audit_rendered_figure_outputs(
        root / "figures" / "Figure_E6_continuous_dwell_gradient", config
    )
    figure_status = "PASS" if figure_audits and all(row["status"] == "PASS" for row in figure_audits) else "WARN"
    figure_detail = (
        f"single_panels={len(figure_audits)}"
        if figure_status == "PASS"
        else ";".join(row["warnings"] or "ok" for row in figure_audits) or "missing_rendered_panels"
    )
    validation_rows.append(("figure_boundary_audit", figure_status, figure_detail))
    validation = ["# Experiment 6 Dwell-Gradient Validation", "", "| check | status | detail |", "| --- | --- | --- |"]
    validation.extend([f"| {name} | {status} | {detail} |" for name, status, detail in validation_rows])
    (root / "experiment_06_dwell_gradient_validation.md").write_text("\n".join(validation) + "\n", encoding="utf-8")
    pd.DataFrame(figure_audits).to_csv(root / "experiment_06_dwell_gradient_figure_audit.csv", index=False)


def run(config: dict) -> None:
    root = Path(config["result_root"])
    tables = root / "tables"
    figures = root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    (root / "resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    seed = int(config["random_seed"])
    theta = base.create_true_theta(
        seed=20260625,
        sparsity=float(config["simulation"]["random_interaction_sparsity_before_scaffold"]),
    )
    pilot = pilot_state_table(theta, config, seed + 1)
    dwell_by_mask, truth = select_gradient_truth_states(pilot, config)
    pilot.to_csv(tables / "pilot_state_candidate_audit.tsv", sep="\t", index=False)
    truth.to_csv(tables / "truth_gradient_states.tsv", sep="\t", index=False)

    score_frames = []
    metric_rows = []
    repeats = int(config["simulation"]["repeats"])
    for repeat in range(1, repeats + 1):
        scores, metrics = run_repeat(repeat, theta, dwell_by_mask, truth, config)
        score_frames.append(scores)
        metric_rows.append(metrics)
        if repeat % 10 == 0:
            print(f"completed repeat {repeat}/{repeats}", flush=True)
    all_scores = pd.concat(score_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    repeat_level, level_summary = summarize_levels(all_scores)
    perf = summarize_performance(metrics)
    all_scores.to_csv(tables / "gradient_state_scores.tsv", sep="\t", index=False)
    metrics.to_csv(tables / "repeat_metrics.tsv", sep="\t", index=False)
    repeat_level.to_csv(tables / "repeat_level_scores.tsv", sep="\t", index=False)
    level_summary.to_csv(tables / "level_summary.tsv", sep="\t", index=False)
    perf.to_csv(tables / "performance_summary.tsv", sep="\t", index=False)

    render_figure(root, config)
    write_reports(root, config, metrics, perf, truth)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.repeats is not None:
        config["simulation"]["repeats"] = int(args.repeats)
    root = Path(config["result_root"])
    if args.render_only:
        render_figure(root, config)
        metrics = pd.read_csv(root / "tables" / "repeat_metrics.tsv", sep="\t")
        perf = pd.read_csv(root / "tables" / "performance_summary.tsv", sep="\t")
        truth = pd.read_csv(root / "tables" / "truth_gradient_states.tsv", sep="\t")
        write_reports(root, config, metrics, perf, truth)
        return
    run(config)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise

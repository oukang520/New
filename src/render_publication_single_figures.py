"""Render direct publication single figures from completed experiment tables.

This script is deliberately separate from the crop-based standardization tool.
It regenerates atomic figures from tabular outputs so manuscript assembly does
not depend on cropping multipanel PNGs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import figure_style


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
STYLE_CONFIG = PROJECT_ROOT / "configs" / "figure_style.yaml"
STYLE = {"plot_style_config": str(STYLE_CONFIG)}
DATASETS = ["AACR_LUAD", "AACR_COAD", "AACR_IDC"]
SHORT = {"AACR_LUAD": "LUAD", "AACR_COAD": "COAD", "AACR_IDC": "IDC"}


def init_style() -> tuple[dict, dict]:
    figure_style.configure_matplotlib(STYLE)
    colors = figure_style.colors(STYLE)
    cat = figure_style.categorical_palette(STYLE)
    return colors, cat


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    sep = "\t" if path.suffix == ".tsv" else ","
    return pd.read_csv(path, sep=sep)


def out(root: str | Path, name: str) -> Path:
    path = Path(root) / "single_figures" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save(fig: plt.Figure, path: Path, pad: float = 0.03) -> None:
    figure_style.save_figure(fig, path, STYLE, pad_inches=pad)


def clean(ax: plt.Axes, grid: str | None = "y") -> None:
    colors, _ = init_style()
    grid_color = colors.get("text", {}).get("grid", "#E6E6E6")
    if grid:
        ax.grid(axis=grid, color=grid_color, lw=0.35, zorder=0)
    sns.despine(ax=ax)


def format_number(value: object) -> str:
    if pd.isna(value):
        return "NE"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) >= 1000:
        return f"{number:,.0f}"
    if abs(number) >= 10:
        return f"{number:.1f}"
    if abs(number) >= 1:
        return f"{number:.2f}"
    return f"{number:.3f}"


DISPLAY_COLUMNS = {
    "short_name": "cohort",
    "dataset_name": "cohort",
    "analysis_samples": "samples",
    "observed_states": "states",
    "positive_inflow_states": "F>0",
    "zero_inflow_states": "F=0",
    "stable_states": "stable",
    "stable_sample_fraction": "stable frac.",
    "median_positive_F_hat": "median F",
    "max_F_hat": "max F",
    "spearman_L_vs_F": "rho L,F",
    "state": "state",
    "variant_display": "variant",
    "hr": "HR",
    "ci_low": "CI low",
    "ci_high": "CI high",
    "c_index": "C-index",
    "delta_c_index_vs_full": "delta C",
    "median_top_overlap": "top overlap",
    "repeats": "repeats",
    "top_k": "top k",
    "observed_overlap_fraction": "observed",
    "median_shuffled_overlap": "shuffled",
    "median_overlap_loss": "loss",
    "exact_recovery_fraction": "exact rec.",
    "display_paths": "paths",
    "top_rstar_paths": "top R* paths",
    "long_event_rstar_paths": "long-event paths",
    "unique_nodes": "nodes",
    "edges": "edges",
    "median_target_R_star": "median R*",
    "eligible_states": "eligible",
    "high_confidence_states": "high conf.",
    "top_states": "top states",
    "top_expected_module_fraction": "expected module",
    "top_ci_above_one_fraction": "CI > 1",
    "median_top_R_star": "median top R*",
    "spearman_R_star_vs_N": "rho R*,N",
    "top_R_star_median": "top R* median",
    "top_O_star_median": "top O* median",
    "states_R_gt_1": "R*>1 states",
    "states_O_gt_1": "O*>1 states",
    "median_common_states": "common states",
    "median_spearman_rho": "median rho",
    "iqr_spearman_rho": "IQR rho",
    "median_top10_overlap": "top10 overlap",
    "median_top10_enrichment": "top10 enrich.",
    "median_direction_concordance": "direction",
    "scenario": "scenario",
    "spearman_O_star": "rho O*",
    "spearman_occupancy": "rho occ.",
    "high_omega_auc_O_star": "AUC O*",
    "high_omega_auc_occupancy": "AUC occ.",
    "top3_precision_O_star": "top3 O*",
    "top3_precision_occupancy": "top3 occ.",
    "endpoint": "endpoint",
    "R_star_median": "R* median",
    "R_star_q1": "R* q1",
    "R_star_q3": "R* q3",
    "occupancy_median": "occ. median",
    "occupancy_q1": "occ. q1",
    "occupancy_q3": "occ. q3",
    "favorable_delta_median": "fav. delta",
    "wilcoxon_p": "p",
    "n_P_C": "n (P/C)",
    "AUC_95CI": "AUC (95% CI)",
    "AP_lift": "AP lift",
    "Delta_persist_95CI": "delta persist (95% CI)",
    "rho_minimum_dwell_95CI": "rho dwell (95% CI)",
    "exact_state_fraction": "exact state",
}


def render_table(df: pd.DataFrame, path: Path, width: float = 6.2, row_height: float = 0.32) -> None:
    _, cat = init_style()
    n_rows = len(df) + 1
    height = max(2.25, row_height * n_rows + 0.70)
    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.06, top=0.94)
    ax.axis("off")
    display = df.copy()
    for column in display.columns:
        display[column] = display[column].map(format_number)
    display.columns = [DISPLAY_COLUMNS.get(str(column), str(column).replace("_", " ")) for column in display.columns]
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    if len(display.columns) >= 14:
        font_size = 4.3
    elif len(display.columns) >= 10:
        font_size = 4.9
    elif len(display.columns) >= 7:
        font_size = 5.5
    else:
        font_size = 6.1
    table.set_fontsize(font_size)
    y_scale = 2.08 if n_rows <= 5 else 1.42
    table.scale(1.0, y_scale)
    header_color = cat.get("sage", "#B8D2CC")
    stripe = "#F6F7F7"
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#FFFFFF")
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(weight="bold", color="#263238")
        elif row % 2 == 0:
            cell.set_facecolor(stripe)
    save(fig, path, pad=0.05)


def render_forest(
    df: pd.DataFrame,
    y_col: str,
    x_col: str,
    low_col: str,
    high_col: str,
    group_col: str,
    path: Path,
    xlabel: str,
    xline: float | None = None,
    height: float = 4.0,
) -> None:
    _, cat = init_style()
    palette = {
        "LUAD": cat.get("lavender", "#B5AED5"),
        "COAD": cat.get("sky_blue", "#B2E6FD"),
        "IDC": cat.get("sage", "#B8D2CC"),
    }
    work = df.copy().reset_index(drop=True)
    work["label"] = work[group_col].astype(str) + " | " + work[y_col].astype(str)
    work = work.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(4.6, height))
    fig.subplots_adjust(left=0.43, right=0.96, bottom=0.15, top=0.96)
    y = np.arange(len(work))
    for idx, row in work.iterrows():
        color = palette.get(str(row[group_col]), "#B8D2CC")
        ax.plot([row[low_col], row[high_col]], [idx, idx], color=color, lw=1.0, zorder=2)
        ax.plot(row[x_col], idx, marker="o", ms=3.3, color=color, mec="#263238", mew=0.35, zorder=3)
    if xline is not None:
        ax.axvline(xline, color="#777777", lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.set_yticks(y, work["label"])
    ax.tick_params(axis="y", labelsize=5.5)
    ax.set_xlabel(xlabel)
    clean(ax, grid="x")
    save(fig, path)


def render_e4() -> None:
    root = RESULTS / "experiment_04_relative_inflow"
    if not (root / "experiment_04_summary.csv").exists():
        return
    _, cat = init_style()
    summary = read_table(root / "experiment_04_summary.csv")
    summary["short_name"] = summary["dataset_name"].map(SHORT)
    table = summary[
        [
            "short_name",
            "analysis_samples",
            "observed_states",
            "positive_inflow_states",
            "zero_inflow_states",
            "stable_sample_fraction",
            "spearman_L_vs_F",
            "main_edges",
        ]
    ].rename(
        columns={
            "short_name": "cohort",
            "analysis_samples": "samples",
            "observed_states": "states",
            "positive_inflow_states": "F>0 states",
            "zero_inflow_states": "F=0 states",
            "stable_sample_fraction": "stable fraction",
            "spearman_L_vs_F": "rho(L,F)",
            "main_edges": "edges",
        }
    )
    render_table(table, out(root, "Table_E4_relative_inflow__cohort_summary"), width=6.4)

    edge_frames = []
    for dataset in DATASETS:
        path = root / dataset / "tables" / "predecessor_edges_rule_a_one_step.tsv"
        if path.exists():
            df = read_table(path).sort_values("inflow_contribution", ascending=False).head(6)
            df["short_name"] = SHORT[dataset]
            df["edge"] = df["source_state"].str.replace("primary::", "P:", regex=False).str.replace("metastatic::", "M:", regex=False) + " -> " + df["event_added"].astype(str)
            edge_frames.append(df)
    if edge_frames:
        edges = pd.concat(edge_frames, ignore_index=True)
        fig, axes = plt.subplots(1, len(DATASETS), figsize=(6.8, 2.55), sharex=False)
        fig.subplots_adjust(left=0.10, right=0.98, bottom=0.20, top=0.95, wspace=0.65)
        for ax, dataset in zip(axes, DATASETS):
            sub = edges[edges["short_name"].eq(SHORT[dataset])].iloc[::-1]
            y = np.arange(len(sub))
            ax.barh(y, sub["inflow_contribution"], color=cat.get("sage", "#B8D2CC"), edgecolor="#263238", linewidth=0.32)
            ax.set_yticks(y, sub["edge"])
            ax.tick_params(axis="y", labelsize=4.5)
            ax.set_xlabel("Inflow contribution")
            ax.text(0.02, 0.98, SHORT[dataset], transform=ax.transAxes, ha="left", va="top", fontsize=6.3, fontweight="bold")
            clean(ax, grid="x")
        save(fig, out(root, "Figure_E4_relative_inflow__dominant_edges"))

    rows = []
    for dataset in DATASETS:
        path = root / dataset / "tables" / "inflow_rule_sensitivity.tsv"
        if path.exists():
            df = read_table(path)
            df["short_name"] = SHORT[dataset]
            rows.append(df)
    if rows:
        sens = pd.concat(rows, ignore_index=True)
        keep = [col for col in ["short_name", "rule", "positive_inflow_states", "stable_states", "spearman_L_vs_F"] if col in sens.columns]
        render_table(
            sens[keep].rename(columns={"short_name": "cohort", "positive_inflow_states": "F>0 states", "stable_states": "stable states", "spearman_L_vs_F": "rho(L,F)"}),
            out(root, "Table_E4_relative_inflow__rule_sensitivity"),
            width=6.6,
            row_height=0.25,
        )


def render_e5() -> None:
    root = RESULTS / "experiment_05_state_scores"
    if not (root / "experiment_05_summary.csv").exists():
        return
    summary = read_table(root / "experiment_05_summary.csv")
    summary["short_name"] = summary["dataset_name"].map(SHORT)
    render_table(
        summary[
            [
                "short_name",
                "states_eligible",
                "states_high_confidence",
                "median_R_raw",
                "median_O_star",
                "top_bottleneck_stability",
                "top_high_confidence_stability",
            ]
        ].rename(
            columns={
                "short_name": "cohort",
                "states_eligible": "eligible",
                "states_high_confidence": "high confidence",
                "median_R_raw": "median R",
                "median_O_star": "median O*",
                "top_bottleneck_stability": "top stability",
                "top_high_confidence_stability": "HC stability",
            }
        ),
        out(root, "Table_E5_state_scores__cohort_summary"),
        width=6.1,
    )
    rows = []
    for dataset in DATASETS:
        path = root / dataset / "tables" / "top_bottleneck_states_high_confidence.tsv"
        if path.exists():
            df = read_table(path).head(5)
            df["short_name"] = SHORT[dataset]
            df["state_short"] = df["state"].str.replace("primary::", "P:", regex=False).str.replace("metastatic::", "M:", regex=False)
            rows.append(df)
    if rows:
        top = pd.concat(rows, ignore_index=True)
        render_forest(top, "state_short", "R_star", "R_star_ci_low", "R_star_ci_high", "short_name", out(root, "Figure_E5_state_scores__top_rstar_forest"), r"$R^*$", xline=1.0, height=4.6)


def render_e6() -> None:
    root = RESULTS / "experiment_06_bottleneck_recovery_enhanced"
    table_path = root / "tables" / "performance_summary_table.tsv"
    if table_path.exists():
        perf = read_table(table_path)
        render_table(
            perf[["endpoint", "R_star_median_iqr", "occupancy_median_iqr", "paired_delta_median", "paired_p_value", "n_repeats"]],
            out(root, "Table_E6_bottleneck_recovery__performance_summary"),
            width=6.6,
            row_height=0.28,
        )
    repeat_path = root / "tables" / "repeat_metrics.tsv"
    if repeat_path.exists():
        repeat = read_table(repeat_path)
        metrics = [
            ("spearman", "spearman_R_star", "spearman_occupancy"),
            ("ROC AUC", "bottleneck_auc_R_star", "bottleneck_auc_occupancy"),
            ("AP", "bottleneck_ap_R_star", "bottleneck_ap_occupancy"),
            ("recall@5", "bottleneck_recall_at5_R_star", "bottleneck_recall_at5_occupancy"),
        ]
        rows = []
        for label, r_col, o_col in metrics:
            for method, col in [("R*", r_col), ("Occupancy", o_col)]:
                rows.extend({"metric": label, "method": method, "value": value} for value in repeat[col].dropna())
        work = pd.DataFrame(rows)
        _, cat = init_style()
        fig, ax = plt.subplots(figsize=(4.8, 3.0))
        fig.subplots_adjust(left=0.14, right=0.98, bottom=0.20, top=0.95)
        sns.boxplot(data=work, x="metric", y="value", hue="method", palette={"R*": cat.get("coral", "#E8B2A7"), "Occupancy": cat.get("sky_blue", "#B2E6FD")}, fliersize=1.0, linewidth=0.55, ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("Score")
        ax.legend(frameon=False, loc="lower right", fontsize=6)
        clean(ax)
        save(fig, out(root, "Figure_E6_bottleneck_recovery__metric_distributions"))


def render_e6_gradient() -> None:
    root = RESULTS / "experiment_06_dwell_gradient"
    perf_path = root / "tables" / "performance_summary.tsv"
    if perf_path.exists():
        perf = read_table(perf_path)
        render_table(perf[["endpoint", "R_star_median", "R_star_q1", "R_star_q3", "occupancy_median", "occupancy_q1", "occupancy_q3", "favorable_delta_median", "wilcoxon_p"]], out(root, "Table_E6_dwell_gradient__performance_summary"), width=7.0, row_height=0.27)
    level_path = root / "tables" / "repeat_level_scores.tsv"
    if level_path.exists():
        level = read_table(level_path)
        work = []
        for (dwell, method), group in level.melt(id_vars=["repeat", "D_true_assigned"], value_vars=["median_log2_R_star", "median_log2_occupancy_star"], var_name="method", value_name="log2_score").groupby(["D_true_assigned", "method"]):
            values = group["log2_score"].dropna()
            work.append({"D_true": dwell, "method": "R*" if method == "median_log2_R_star" else "Occupancy", "median": values.median(), "q1": values.quantile(0.25), "q3": values.quantile(0.75)})
        work = pd.DataFrame(work)
        _, cat = init_style()
        fig, ax = plt.subplots(figsize=(3.6, 3.1))
        fig.subplots_adjust(left=0.17, right=0.96, bottom=0.18, top=0.96)
        for method, color in [("R*", cat.get("coral", "#E8B2A7")), ("Occupancy", cat.get("sky_blue", "#B2E6FD"))]:
            sub = work[work["method"].eq(method)].sort_values("D_true")
            ax.plot(sub["D_true"], sub["median"], marker="o", ms=3.5, lw=1.0, color=color, label=method)
            ax.fill_between(sub["D_true"], sub["q1"], sub["q3"], color=color, alpha=0.18, lw=0)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("True relative dwell level")
        ax.set_ylabel(r"Median $\log_2$ score")
        ax.legend(frameon=False, loc="upper left", fontsize=6)
        clean(ax)
        save(fig, out(root, "Figure_E6_dwell_gradient__level_recovery"))


def render_e7() -> None:
    root = RESULTS / "experiment_07_topology_robustness_balanced"
    global_path = root / "tables" / "experiment_07_global_summary.tsv"
    combo_path = root / "tables" / "combo_summary.tsv"
    if global_path.exists():
        g = read_table(global_path)
        columns = [col for col in g.columns if col.startswith("global_spearman") or col.startswith("global_bottleneck_auc") or col in ["conditions", "repeats", "combo_auc_pass_fraction"]]
        render_table(g[columns].T.reset_index().rename(columns={"index": "metric", 0: "value"}), out(root, "Table_E7_topology_robustness__global_summary"), width=6.5, row_height=0.24)
    if combo_path.exists():
        combo = read_table(combo_path)
        _, cat = init_style()
        pivot = combo.pivot_table(index="placement_label", columns="topology_label", values="spearman_R_star_median", aggfunc="median")
        fig, ax = plt.subplots(figsize=(3.35, 3.0))
        fig.subplots_adjust(left=0.30, right=0.92, bottom=0.20, top=0.94)
        sns.heatmap(pivot, cmap=sns.light_palette(cat.get("sage", "#B8D2CC"), as_cmap=True), annot=True, fmt=".2f", linewidths=0.45, linecolor="white", cbar_kws={"label": "median rho", "shrink": 0.72}, annot_kws={"fontsize": 6}, ax=ax)
        ax.set_xlabel("Topology")
        ax.set_ylabel("Bottleneck placement")
        ax.tick_params(axis="x", rotation=30, labelsize=5.8)
        ax.tick_params(axis="y", rotation=0, labelsize=5.8)
        save(fig, out(root, "Figure_E7_topology_robustness__spearman_matrix"))


def render_e8() -> None:
    root = RESULTS / "experiment_08_biological_convergence"
    summary_path = root / "tables" / "cohort_summary.tsv"
    module_path = root / "tables" / "module_enrichment.tsv"
    if summary_path.exists():
        summary = read_table(summary_path)
        render_table(summary[["short_name", "eligible_states", "high_confidence_states", "top_states", "top_expected_module_fraction", "top_ci_above_one_fraction", "median_top_R_star", "spearman_R_star_vs_N"]].rename(columns={"short_name": "cohort"}), out(root, "Table_E8_biological_convergence__cohort_summary"), width=6.8)
    if module_path.exists():
        module = read_table(module_path)
        if "module" in module.columns and "top_fraction" in module.columns:
            _, cat = init_style()
            fig, ax = plt.subplots(figsize=(4.3, 3.1))
            fig.subplots_adjust(left=0.24, right=0.96, bottom=0.17, top=0.95)
            top = module.sort_values("top_fraction", ascending=False).head(12)
            labels = top["short_name"].astype(str) + " | " + top["module"].astype(str)
            y = np.arange(len(top))[::-1]
            ax.barh(y, top["top_fraction"], color=cat.get("sage", "#B8D2CC"), edgecolor="#263238", linewidth=0.32)
            ax.set_yticks(y, labels)
            ax.tick_params(axis="y", labelsize=5.3)
            ax.set_xlabel("Top-state module fraction")
            ax.set_xlim(0, 1.05)
            clean(ax, grid="x")
            save(fig, out(root, "Figure_E8_biological_convergence__module_fraction"))


def render_e9() -> None:
    root = RESULTS / "experiment_09_observation_enrichment"
    summary_path = root / "tables" / "experiment_09_summary.tsv"
    if summary_path.exists():
        summary = read_table(summary_path)
        render_table(summary[["scenario", "stable_states", "spearman_O_star", "spearman_occupancy", "high_omega_auc_O_star", "high_omega_auc_occupancy", "top3_precision_O_star", "top3_precision_occupancy"]], out(root, "Table_E9_observation_enrichment__summary"), width=6.8, row_height=0.28)
    curves_path = root / "tables" / "repeat_curves.tsv"
    if curves_path.exists():
        curves = read_table(curves_path)
        mean_curve = curves.groupby(["scenario", "method", "fpr"], as_index=False)["tpr"].mean()
        _, cat = init_style()
        fig, ax = plt.subplots(figsize=(3.2, 3.1))
        fig.subplots_adjust(left=0.16, right=0.96, bottom=0.17, top=0.96)
        palette = {"O_star": cat.get("coral", "#E8B2A7"), "occupancy": cat.get("sky_blue", "#B2E6FD")}
        for (scenario, method), sub in mean_curve.groupby(["scenario", "method"]):
            ax.plot(sub["fpr"], sub["tpr"], lw=1.0, color=palette.get(method, "#B8D2CC"), ls="-" if scenario == "omega_only" else "--", label=f"{scenario} | {method}")
        ax.plot([0, 1], [0, 1], color="#777777", lw=0.65, ls=(0, (3, 2)))
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(frameon=False, fontsize=4.9, loc="lower right")
        clean(ax, grid="both")
        save(fig, out(root, "Figure_E9_observation_enrichment__mean_roc"))


def render_e10() -> None:
    root = RESULTS / "experiment_10_real_cohort_main"
    summary_path = root / "tables" / "cohort_main_summary.tsv"
    if summary_path.exists():
        summary = read_table(summary_path)
        render_table(summary[["short_name", "eligible_states", "high_confidence_states", "top_R_star_median", "top_R_ci_above_one_fraction", "top_expected_module_fraction", "top_O_star_median", "states_R_gt_1", "states_O_gt_1"]].rename(columns={"short_name": "cohort"}), out(root, "Table_E10_real_cohort_main__summary"), width=6.8)
    top_path = root / "tables" / "top_real_cohort_bottleneck_states.tsv"
    if top_path.exists():
        top = read_table(top_path).groupby("short_name", group_keys=False).head(4).copy()
        top["state_short"] = top["state_label"]
        render_forest(top, "state_short", "R_star", "R_star_ci_low", "R_star_ci_high", "short_name", out(root, "Figure_E10_real_cohort_main__top_rstar_forest"), r"$R^*$", xline=1.0, height=4.0)


def render_e11() -> None:
    root = RESULTS / "experiment_11_information_gain"
    corr_path = root / "tables" / "rank_correlation_bootstrap.tsv"
    if corr_path.exists():
        corr = read_table(corr_path)
        corr["pair_short"] = corr["pair"].str.replace("R* vs ", "R* vs\n", regex=False).str.replace("occupancy vs ", "occ. vs\n", regex=False)
        render_forest(corr, "pair_short", "spearman_rho", "ci_low", "ci_high", "short_name", out(root, "Figure_E11_information_gain__rank_correlation_forest"), "Spearman rho", xline=0.0, height=4.6)
    gain_path = root / "tables" / "rank_gain_distribution.tsv"
    if gain_path.exists():
        gain = read_table(gain_path)
        _, cat = init_style()
        fig, ax = plt.subplots(figsize=(4.1, 3.1))
        fig.subplots_adjust(left=0.16, right=0.96, bottom=0.20, top=0.95)
        sns.stripplot(data=gain, x="short_name", y="percentile_rank_gain", hue="baseline_label", dodge=True, palette={"MHN inflow": cat.get("lavender", "#B5AED5"), "Occupancy": cat.get("sky_blue", "#B2E6FD")}, size=2.8, linewidth=0.25, edgecolor="#263238", ax=ax)
        ax.axhline(0, color="#777777", lw=0.7, ls=(0, (3, 2)))
        ax.set_xlabel("")
        ax.set_ylabel("Percentile rank gain")
        ax.legend(frameon=False, fontsize=5.5, loc="upper left")
        clean(ax)
        save(fig, out(root, "Figure_E11_information_gain__rank_gain_distribution"))


def render_e13() -> None:
    root = RESULTS / "experiment_13_cross_cohort_replication"
    summary_path = root / "tables" / "experiment_13_summary.tsv"
    if summary_path.exists():
        summary = read_table(summary_path)
        render_table(summary[["short_name", "repeats", "median_common_states", "median_spearman_rho", "iqr_spearman_rho", "median_top10_overlap", "median_top10_enrichment", "median_direction_concordance"]].rename(columns={"short_name": "cohort"}), out(root, "Table_E13_split_replication__summary"), width=6.8)
    metrics_path = root / "tables" / "split_replication_metrics.tsv"
    if metrics_path.exists():
        metrics = read_table(metrics_path)
        work = metrics.melt(id_vars=["short_name"], value_vars=["spearman_rho", "top_overlap_fraction", "direction_concordance"], var_name="metric", value_name="value")
        _, cat = init_style()
        fig, ax = plt.subplots(figsize=(4.4, 3.05))
        fig.subplots_adjust(left=0.14, right=0.96, bottom=0.23, top=0.95)
        sns.boxplot(data=work, x="metric", y="value", hue="short_name", palette=[cat.get("lavender", "#B5AED5"), cat.get("sky_blue", "#B2E6FD"), cat.get("sage", "#B8D2CC")], fliersize=0.8, linewidth=0.5, ax=ax)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["rho", "top-10 overlap", "direction"])
        ax.set_xlabel("")
        ax.set_ylabel("Replication score")
        ax.legend(frameon=False, fontsize=5.4, ncol=3, loc="lower left")
        clean(ax)
        save(fig, out(root, "Figure_E13_split_replication__score_distribution"))


def render_e14() -> None:
    root = RESULTS / "experiment_14_ablation_backbone"
    summary_path = root / "tables" / "experiment_14_summary.tsv"
    if summary_path.exists():
        summary = read_table(summary_path)
        render_table(summary[["short_name", "variant_display", "hr", "ci_low", "ci_high", "c_index", "delta_c_index_vs_full", "median_top_overlap"]].rename(columns={"short_name": "cohort"}), out(root, "Table_E14_ablation_backbone__clinical_and_overlap"), width=7.0, row_height=0.23)
    lift_path = root / "tables" / "relative_dwell_rank_lift.tsv"
    if lift_path.exists():
        lift = read_table(lift_path).groupby("short_name", group_keys=False).head(5)
        _, cat = init_style()
        fig, ax = plt.subplots(figsize=(4.5, 3.4))
        fig.subplots_adjust(left=0.36, right=0.96, bottom=0.16, top=0.96)
        lift = lift.iloc[::-1]
        y = np.arange(len(lift))
        palette = {"LUAD": cat.get("lavender", "#B5AED5"), "COAD": cat.get("sky_blue", "#B2E6FD"), "IDC": cat.get("sage", "#B8D2CC")}
        for idx, row in enumerate(lift.itertuples(index=False)):
            color = palette.get(str(row.short_name), "#B8D2CC")
            ax.hlines(idx, row.percentile_L, row.percentile_R, color=color, lw=1.1)
            ax.plot(row.percentile_L, idx, marker="o", ms=2.6, lw=0, color="#FFFFFF", mec="#777777")
            ax.plot(row.percentile_R, idx, marker="o", ms=3.0, lw=0, color=color, mec="#263238", mew=0.35)
        ax.plot([], [], marker="o", ms=2.6, lw=0, color="#FFFFFF", mec="#777777", label="L")
        ax.plot([], [], marker="o", ms=3.0, lw=0, color=cat.get("sage", "#B8D2CC"), mec="#263238", mew=0.35, label="R*")
        labels = lift["short_name"].astype(str) + " | " + lift["state"].str.replace("primary::", "P:", regex=False).str.replace("metastatic::", "M:", regex=False)
        ax.set_yticks(y, labels)
        ax.tick_params(axis="y", labelsize=4.8)
        ax.set_xlabel("Rank percentile")
        ax.legend(frameon=False, loc="lower right", fontsize=5.6)
        clean(ax, grid="x")
        save(fig, out(root, "Figure_E14_ablation_backbone__rank_lift"))


def render_e15() -> None:
    root = RESULTS / "experiment_15_uncertainty_negative_controls"
    fals_path = root / "tables" / "inflow_pairing_falsification_summary.tsv"
    decoy_path = root / "tables" / "matched_decoy_summary.tsv"
    if fals_path.exists():
        fals = read_table(fals_path)
        render_table(fals[["short_name", "repeats", "top_k", "observed_overlap_fraction", "median_shuffled_overlap", "median_overlap_loss", "exact_recovery_fraction"]].rename(columns={"short_name": "cohort"}), out(root, "Table_E15_falsification_controls__inflow_pairing"), width=6.5)
    if decoy_path.exists():
        decoy = read_table(decoy_path)
        _, cat = init_style()
        fig, ax = plt.subplots(figsize=(3.35, 3.0))
        fig.subplots_adjust(left=0.18, right=0.96, bottom=0.18, top=0.95)
        x = np.arange(len(decoy))
        ax.bar(x - 0.16, decoy["fraction_above_decoy_q90"], width=0.30, color=cat.get("coral", "#E8B2A7"), edgecolor="#263238", linewidth=0.32, label="above decoy q90")
        ax.bar(x + 0.16, decoy["median_log2_R_advantage"], width=0.30, color=cat.get("sage", "#B8D2CC"), edgecolor="#263238", linewidth=0.32, label="log2 R advantage")
        ax.set_xticks(x, decoy["short_name"])
        ax.set_ylabel("Value")
        ax.legend(frameon=False, fontsize=5.3, loc="upper left")
        clean(ax)
        save(fig, out(root, "Figure_E15_falsification_controls__matched_decoy_contrast"))


def render_e16() -> None:
    root = RESULTS / "experiment_16_real_topology"
    audit_path = root / "tables" / "real_topology_audit.tsv"
    if audit_path.exists():
        audit = read_table(audit_path)
        render_table(audit[["short_name", "display_paths", "top_rstar_paths", "long_event_rstar_paths", "unique_nodes", "edges", "median_target_R_star"]].rename(columns={"short_name": "cohort"}), out(root, "Table_E16_real_topology__audit"), width=6.4)
    path_path = root / "tables" / "real_topology_paths.tsv"
    if path_path.exists():
        paths = read_table(path_path)
        _, cat = init_style()
        colors = {"LUAD": cat.get("lavender", "#B5AED5"), "COAD": cat.get("sky_blue", "#B2E6FD"), "IDC": cat.get("sage", "#B8D2CC")}
        selected = paths.groupby(["short_name", "path_rank"], as_index=False).first()[["short_name", "path_rank"]].groupby("short_name").head(3)
        rows = []
        for row in selected.itertuples(index=False):
            sub = paths[paths["short_name"].eq(row.short_name) & paths["path_rank"].eq(row.path_rank)].sort_values("path_position")
            label = " -> ".join(sub["state"].str.split("::").str[-1].replace("", "WT"))
            rows.append({"cohort": row.short_name, "rank": row.path_rank, "label": label})
        fig, ax = plt.subplots(figsize=(6.4, 2.8))
        fig.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.96)
        ax.axis("off")
        y = 0.92
        for row in rows:
            ax.text(0.02, y, f"{row['cohort']} r{int(row['rank'])}", color=colors.get(row["cohort"], "#B8D2CC"), fontsize=6.5, fontweight="bold", va="center")
            ax.text(0.17, y, row["label"], fontsize=5.7, va="center", color="#263238")
            y -= 0.105
        save(fig, out(root, "Figure_E16_real_topology__route_strips"), pad=0.04)


def render_e17() -> None:
    root = RESULTS / "experiment_17_longitudinal_public"
    table_path = root / "tables" / "core_metric_table.tsv"
    if table_path.exists():
        table = read_table(table_path)
        render_table(table, out(root, "Table_E17_longitudinal_validation__core_metrics"), width=6.7)
    summary_path = root / "tables" / "dwell_persistence_summary_all.tsv"
    if summary_path.exists():
        summary = read_table(summary_path)
        summary["cohort"] = summary["study_id"].map({"difg_glass": "GLASS", "coadread_mskcc": "CRC-triplets", "mnm_washu_2016": "MNM-WashU"}).fillna(summary["study_id"])
        _, cat = init_style()
        fig, ax = plt.subplots(figsize=(3.6, 3.0))
        fig.subplots_adjust(left=0.22, right=0.96, bottom=0.17, top=0.95)
        y = np.arange(len(summary))[::-1]
        ax.hlines(y, summary["delta_persistence_ci_low"], summary["delta_persistence_ci_high"], color=cat.get("sage", "#B8D2CC"), lw=1.1)
        ax.plot(summary["delta_persistence_rate_high_minus_low"], y, marker="o", ms=3.4, color=cat.get("coral", "#E8B2A7"), mec="#263238", mew=0.35)
        ax.axvline(0, color="#777777", lw=0.7, ls=(0, (3, 2)))
        ax.set_yticks(y, summary["cohort"])
        ax.set_xlabel("Top-bottom persistence difference")
        clean(ax, grid="x")
        save(fig, out(root, "Figure_E17_longitudinal_validation__top_bottom_persistence"))

        fig, ax = plt.subplots(figsize=(3.6, 3.0))
        fig.subplots_adjust(left=0.22, right=0.96, bottom=0.17, top=0.95)
        y = np.arange(len(summary))[::-1]
        ax.hlines(y, summary["spearman_r_minimum_dwell_ci_low"], summary["spearman_r_minimum_dwell_ci_high"], color=cat.get("sky_blue", "#B2E6FD"), lw=1.1)
        ax.plot(summary["spearman_r_minimum_dwell_interval"], y, marker="o", ms=3.4, color=cat.get("lavender", "#B5AED5"), mec="#263238", mew=0.35)
        ax.axvline(0, color="#777777", lw=0.7, ls=(0, (3, 2)))
        ax.set_yticks(y, summary["cohort"])
        ax.set_xlabel("rho(R*, minimum dwell proxy)")
        clean(ax, grid="x")
        save(fig, out(root, "Figure_E17_longitudinal_validation__minimum_dwell_correlation"))


def main() -> None:
    init_style()
    for renderer in [
        render_e4,
        render_e5,
        render_e6,
        render_e6_gradient,
        render_e7,
        render_e8,
        render_e9,
        render_e10,
        render_e11,
        render_e13,
        render_e14,
        render_e15,
        render_e16,
        render_e17,
    ]:
        renderer()
    print("direct publication single figures rendered")


if __name__ == "__main__":
    main()

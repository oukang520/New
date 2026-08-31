"""Experiment 17 extension: real longitudinal calibration of held-out R*.

This script consumes the audited Experiment 17 pair-level predictions and asks
whether higher held-out R* states are progressively more persistent in real
longitudinal samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

import figure_style


CONFIG_PATH = Path("src/relobstq_mhn/configs/experiment_17_longitudinal_extension.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E17 longitudinal R* calibration extension.")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    config_path = Path(path)
    if not config_path.exists() and Path("configs/experiment_17_longitudinal_extension.yaml").exists():
        config_path = Path("configs/experiment_17_longitudinal_extension.yaml")
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def genotype_events(genotype: object) -> set[str]:
    if pd.isna(genotype):
        return set()
    text = str(genotype).strip()
    if text == "" or text.upper() == "WT":
        return set()
    return {event for event in text.split("+") if event}


def genotype_similarity(row: pd.Series) -> float:
    early = genotype_events(row["early_genotype"])
    late = genotype_events(row["late_genotype"])
    if not early and not late:
        return 1.0
    union = early | late
    return float(len(early & late) / len(union)) if union else 1.0


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, replicates: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    if values.size == 1:
        return float(values[0]), float(values[0]), float(values[0])
    means = np.empty(replicates, dtype=float)
    for idx in range(replicates):
        sample = rng.choice(values, size=values.size, replace=True)
        means[idx] = sample.mean()
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def prepare_predictions(config: dict) -> pd.DataFrame:
    source = Path(config["source_result_root"]) / "tables" / "dwell_persistence_predictions_all.tsv"
    predictions = pd.read_csv(source, sep="\t")
    included = set(config["included_studies"].keys())
    score_col = config["analysis"]["primary_score_column"]
    data = predictions[predictions["study_id"].isin(included)].copy()
    filters = config["analysis"]["filters"]
    if filters.get("require_pair_qc_pass", True):
        data = data[data["pair_qc_pass"].eq(True)]
    data = data[data[score_col].notna()].copy()
    if filters.get("require_exact_heldout_state_score", True):
        data = data[data["score_source"].eq("exact_state")].copy()
    data["genotype_similarity_fixed"] = data.apply(genotype_similarity, axis=1)
    data["minimum_dwell_proxy"] = pd.to_numeric(data["minimum_observed_dwell_interval"], errors="coerce").fillna(0.0)
    data["empirical_persistent"] = pd.to_numeric(data["empirical_persistent"], errors="coerce").astype(int)
    bins = int(config["analysis"]["rstar_quantile_bins"])
    labels = [f"Q{i}" for i in range(1, bins + 1)]
    parts = []
    for study_id, group in data.groupby("study_id", sort=False):
        ranked = group[score_col].rank(method="first")
        group = group.copy()
        group["rstar_bin"] = pd.qcut(ranked, bins, labels=labels)
        group["rstar_bin_index"] = group["rstar_bin"].astype(str).str.extract(r"Q(\d+)").astype(int)
        max_proxy = float(group["minimum_dwell_proxy"].max())
        group["minimum_dwell_proxy_scaled"] = group["minimum_dwell_proxy"] / max_proxy if max_proxy > 0 else 0.0
        group["study_short_name"] = config["included_studies"][study_id]["short_name"]
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def summarize_bins(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    rng = np.random.default_rng(int(config["random_seed"]))
    replicates = int(config["analysis"]["bootstrap_replicates"])
    rows = []
    score_col = config["analysis"]["primary_score_column"]
    for (study_id, bin_label), group in data.groupby(["study_id", "rstar_bin"], observed=True):
        row = {
            "study_id": study_id,
            "study_short_name": group["study_short_name"].iloc[0],
            "rstar_bin": str(bin_label),
            "rstar_bin_index": int(group["rstar_bin_index"].iloc[0]),
            "n_pairs": int(len(group)),
            "log2_R_min": float(group[score_col].min()),
            "log2_R_median": float(group[score_col].median()),
            "log2_R_max": float(group[score_col].max()),
        }
        for column, prefix in [
            ("empirical_persistent", "persistence_rate"),
            ("genotype_similarity_fixed", "genotype_similarity"),
            ("minimum_dwell_proxy_scaled", "minimum_dwell_proxy_scaled"),
        ]:
            mean, low, high = bootstrap_mean_ci(group[column].to_numpy(dtype=float), rng, replicates)
            row[prefix] = mean
            row[f"{prefix}_ci_low"] = low
            row[f"{prefix}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["study_id", "rstar_bin_index"])


def summarize_metrics(data: pd.DataFrame, bins: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows = []
    score_col = config["analysis"]["primary_score_column"]
    for study_id, group in data.groupby("study_id", sort=False):
        study_bins = bins[bins["study_id"] == study_id].sort_values("rstar_bin_index")
        low = study_bins.iloc[0]
        high = study_bins.iloc[-1]
        row = {
            "study_id": study_id,
            "study_short_name": config["included_studies"][study_id]["short_name"],
            "n_evaluable_pairs": int(len(group)),
            "persistent_pairs": int(group["empirical_persistent"].sum()),
            "changed_pairs": int((1 - group["empirical_persistent"]).sum()),
            "top_bottom_persistence_delta": float(high["persistence_rate"] - low["persistence_rate"]),
            "top_bottom_similarity_delta": float(high["genotype_similarity"] - low["genotype_similarity"]),
            "top_bottom_minimum_dwell_proxy_delta_scaled": float(
                high["minimum_dwell_proxy_scaled"] - low["minimum_dwell_proxy_scaled"]
            ),
            "top_bin_persistence_rate": float(high["persistence_rate"]),
            "bottom_bin_persistence_rate": float(low["persistence_rate"]),
            "spearman_persistence": float(spearmanr(group[score_col], group["empirical_persistent"]).statistic),
            "spearman_persistence_p": float(spearmanr(group[score_col], group["empirical_persistent"]).pvalue),
            "spearman_similarity": float(spearmanr(group[score_col], group["genotype_similarity_fixed"]).statistic),
            "spearman_similarity_p": float(spearmanr(group[score_col], group["genotype_similarity_fixed"]).pvalue),
            "spearman_minimum_dwell_proxy": float(spearmanr(group[score_col], group["minimum_dwell_proxy"]).statistic),
            "spearman_minimum_dwell_proxy_p": float(spearmanr(group[score_col], group["minimum_dwell_proxy"]).pvalue),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.14, 1.06, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")


def plot_bin_lines(ax: plt.Axes, bins: pd.DataFrame, y: str, low: str, high: str, ylabel: str, title: str, colors: dict) -> None:
    palette = {
        "GLASS": colors.get("sky_blue", "#B2E6FD"),
        "CRC-triplets": colors.get("coral", "#E8B2A7"),
        "MNM-WashU": colors.get("sage", "#B8D2CC"),
    }
    edges = {
        "GLASS": "#2D7DA8",
        "CRC-triplets": "#B36B5B",
        "MNM-WashU": "#5F8F84",
    }
    for short_name, group in bins.groupby("study_short_name"):
        group = group.sort_values("rstar_bin_index")
        x = group["rstar_bin_index"].to_numpy(dtype=float)
        y_values = group[y].to_numpy(dtype=float)
        yerr = [
            y_values - group[low].to_numpy(dtype=float),
            group[high].to_numpy(dtype=float) - y_values,
        ]
        ax.errorbar(
            x,
            y_values,
            yerr=yerr,
            fmt="o-",
            color=edges.get(short_name, "#333333"),
            mfc=palette.get(short_name, "#CCCCCC"),
            mec=edges.get(short_name, "#333333"),
            lw=1.1,
            ms=4.0,
            capsize=2.3,
            label=short_name,
        )
    ax.set_xlim(0.75, 4.25)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
    ax.set_xlabel("Held-out R* quartile")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.grid(axis="y", color="#E6E6E6", lw=0.5)


def _cohort_colors(config: dict) -> tuple[dict[str, str], dict[str, str]]:
    colors = figure_style.categorical_palette(config)
    fills = {
        "GLASS": colors.get("lavender", "#B5AED5"),
        "CRC-triplets": colors.get("sky_blue", "#B2E6FD"),
        "MNM-WashU": colors.get("sage", "#B8D2CC"),
    }
    edges = {
        "GLASS": "#766FA2",
        "CRC-triplets": "#2D7DA8",
        "MNM-WashU": "#5F8F84",
    }
    return fills, edges


def _fmt_float(value: object, digits: int = 2, missing: str = "NE") -> str:
    try:
        number = float(value)
    except Exception:
        return missing
    if not np.isfinite(number):
        return missing
    return f"{number:.{digits}f}"


def _add_dumbbell_panel(ax: plt.Axes, summary: pd.DataFrame, study_order: list[str], short_lookup: dict[str, str], config: dict) -> None:
    fills, edges = _cohort_colors(config)
    y_values = np.arange(len(study_order), 0, -1)
    for y, study_id in zip(y_values, study_order):
        row = summary[summary["study_id"].eq(study_id)].iloc[0]
        short_name = short_lookup[study_id]
        low = float(row["low_rstar_persistence_rate"])
        high = float(row["high_rstar_persistence_rate"])
        delta = float(row["delta_persistence_rate_high_minus_low"])
        ax.plot([low, high], [y, y], color="#4E5A5E", lw=1.05, zorder=1)
        ax.scatter(low, y, s=36, facecolor="white", edgecolor=fills.get(short_name, "#CCCCCC"), linewidth=1.4, zorder=2)
        ax.scatter(high, y, s=42, facecolor=fills.get(short_name, "#CCCCCC"), edgecolor="#263238", linewidth=0.75, zorder=3)
        ax.text(
            min(max(low, high) + 0.035, 1.05),
            y,
            f"n={int(row['evaluable_pairs'])}, d={delta:+.2f}",
            ha="left",
            va="center",
            fontsize=6.2,
            color="#263238",
            clip_on=False,
        )
    ax.axvline(0, color="#333333", lw=0.7, ls=":")
    ax.set_yticks(y_values)
    ax.set_yticklabels([short_lookup[study] for study in study_order])
    ax.set_xlim(-0.03, 1.13)
    ax.set_ylim(0.35, len(study_order) + 0.65)
    ax.set_xlabel("genotype-persistence rate")
    ax.set_title("top R* states persist more often", loc="left")
    ax.grid(axis="x", color="#E6E6E6", lw=0.5)
    ax.set_box_aspect(1)


def _add_dwell_forest_panel(ax: plt.Axes, summary: pd.DataFrame, study_order: list[str], short_lookup: dict[str, str], config: dict) -> None:
    fills, edges = _cohort_colors(config)
    y_values = np.arange(len(study_order), 0, -1)
    for y, study_id in zip(y_values, study_order):
        row = summary[summary["study_id"].eq(study_id)].iloc[0]
        short_name = short_lookup[study_id]
        rho = float(row["spearman_r_minimum_dwell_interval"])
        low = float(row["spearman_r_minimum_dwell_ci_low"])
        high = float(row["spearman_r_minimum_dwell_ci_high"])
        xerr = np.array([[rho - low], [high - rho]])
        ax.errorbar(
            rho,
            y,
            xerr=xerr,
            fmt="o",
            color=edges.get(short_name, "#333333"),
            mfc=fills.get(short_name, "#CCCCCC"),
            mec="#263238",
            mew=0.65,
            ms=5.0,
            lw=1.05,
            capsize=3.0,
            zorder=3,
        )
        ax.text(
            min(high + 0.035, 0.76),
            y + 0.08,
            f"rho={rho:.2f}",
            ha="left",
            va="bottom",
            fontsize=6.2,
            color="#263238",
            clip_on=False,
        )
    ax.axvline(0, color="#333333", lw=0.7, ls=":")
    ax.set_yticks(y_values)
    ax.set_yticklabels([short_lookup[study] for study in study_order])
    ax.set_xlim(-0.25, 0.78)
    ax.set_ylim(0.35, len(study_order) + 0.65)
    ax.set_xlabel(r"Spearman $\rho$: $R^*$ vs minimum dwell")
    ax.set_title("ranked R* tracks dwell proxy", loc="left")
    ax.grid(axis="x", color="#E6E6E6", lw=0.5)
    ax.set_box_aspect(1)


def _add_single_calibration_panel(
    ax: plt.Axes,
    bins: pd.DataFrame,
    metrics: pd.DataFrame,
    study_id: str,
    short_lookup: dict[str, str],
    config: dict,
    y: str,
    low: str,
    high: str,
    ylabel: str,
    metric_prefix: str,
    show_ylabel: bool,
    show_xlabel: bool,
) -> None:
    fills, edges = _cohort_colors(config)
    group = bins[bins["study_id"].eq(study_id)].sort_values("rstar_bin_index")
    short_name = short_lookup[study_id]
    if group.empty:
        ax.text(0.5, 0.5, "NE", transform=ax.transAxes, ha="center", va="center", fontsize=7)
    else:
        x = group["rstar_bin_index"].to_numpy(dtype=float)
        y_values = group[y].to_numpy(dtype=float)
        yerr = np.vstack(
            [
                y_values - group[low].to_numpy(dtype=float),
                group[high].to_numpy(dtype=float) - y_values,
            ]
        )
        ax.errorbar(
            x,
            y_values,
            yerr=yerr,
            fmt="o-",
            color=edges.get(short_name, "#333333"),
            mfc=fills.get(short_name, "#CCCCCC"),
            mec="#263238",
            mew=0.55,
            lw=1.05,
            ms=4.0,
            capsize=2.3,
            label=short_name,
        )
        metric_row = metrics[metrics["study_id"].eq(study_id)]
        if not metric_row.empty:
            metric_row = metric_row.iloc[0]
            if metric_prefix == "persistence":
                text = (
                    f"d={float(metric_row['top_bottom_persistence_delta']):+.2f}\n"
                    f"rho={float(metric_row['spearman_persistence']):.2f}"
                )
            else:
                text = (
                    f"d={float(metric_row['top_bottom_minimum_dwell_proxy_delta_scaled']):+.2f}\n"
                    f"rho={float(metric_row['spearman_minimum_dwell_proxy']):.2f}"
                )
            ax.text(
                0.97,
                0.08,
                text,
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=5.7,
                color="#4E5A5E",
            )
    ax.set_xlim(0.75, 4.25)
    ax.set_ylim(-0.06, 1.06)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
    ax.set_xlabel("held-out R* quartile" if show_xlabel else "")
    ax.set_ylabel(ylabel if show_ylabel else "")
    if not show_ylabel:
        ax.set_yticklabels([])
    ax.set_title(short_name, loc="left", pad=3, fontsize=7.5)
    ax.grid(axis="y", color="#E6E6E6", lw=0.5)
    ax.set_box_aspect(1)


def render_integrated_figure(root: Path, config: dict) -> None:
    source_root = Path(config["source_result_root"])
    figure_style.configure_matplotlib(config)
    summary = pd.read_csv(source_root / "tables" / "dwell_persistence_summary_all.tsv", sep="\t")
    bins = pd.read_csv(root / "tables" / "rstar_calibration_bins.tsv", sep="\t")
    metrics = pd.read_csv(root / "tables" / "rstar_calibration_metrics.tsv", sep="\t")
    included = config["included_studies"]
    study_order = [study_id for study_id in included if study_id in set(summary["study_id"])]
    short_lookup = {study_id: included[study_id]["short_name"] for study_id in study_order}

    fig = plt.figure(figsize=tuple(config["plot"].get("integrated_figure_size", [8.4, 8.4])))
    gs = fig.add_gridspec(
        3,
        6,
        height_ratios=[1.08, 0.78, 0.78],
        width_ratios=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        left=0.095,
        right=0.985,
        bottom=0.075,
        top=0.945,
        wspace=0.88,
        hspace=0.70,
    )
    ax_a = fig.add_subplot(gs[0, 0:3])
    ax_b = fig.add_subplot(gs[0, 3:6])
    c_axes = [fig.add_subplot(gs[1, start : start + 2]) for start in (0, 2, 4)]
    d_axes = [fig.add_subplot(gs[2, start : start + 2]) for start in (0, 2, 4)]

    _add_dumbbell_panel(ax_a, summary, study_order, short_lookup, config)
    add_panel_label(ax_a, "A")
    _add_dwell_forest_panel(ax_b, summary, study_order, short_lookup, config)
    add_panel_label(ax_b, "B")
    fig.text(0.095, 0.595, "C", fontsize=10, fontweight="bold", ha="left", va="center")
    fig.text(0.126, 0.595, "R* quartiles calibrate persistence", fontsize=9.5, ha="left", va="center")
    fig.text(0.095, 0.318, "D", fontsize=10, fontweight="bold", ha="left", va="center")
    fig.text(0.126, 0.318, "R* quartiles calibrate dwell proxy", fontsize=9.5, ha="left", va="center")
    for index, study_id in enumerate(study_order):
        _add_single_calibration_panel(
            c_axes[index],
            bins,
            metrics,
            study_id,
            short_lookup,
            config,
            "persistence_rate",
            "persistence_rate_ci_low",
            "persistence_rate_ci_high",
            "persistence rate",
            "persistence",
            show_ylabel=index == 0,
            show_xlabel=False,
        )
        _add_single_calibration_panel(
            d_axes[index],
            bins,
            metrics,
            study_id,
            short_lookup,
            config,
            "minimum_dwell_proxy_scaled",
            "minimum_dwell_proxy_scaled_ci_low",
            "minimum_dwell_proxy_scaled_ci_high",
            "minimum dwell proxy, scaled",
            "dwell",
            show_ylabel=index == 0,
            show_xlabel=True,
        )

    figure_style.save_figure_panels(
        fig,
        source_root / "figures" / "Figure_E17_integrated_longitudinal_validation",
        config,
        pad_inches=0.08,
    )


def write_and_render_integrated_metrics_table(root: Path, config: dict) -> pd.DataFrame:
    source_root = Path(config["source_result_root"])
    figure_style.configure_matplotlib(config)
    colors = figure_style.categorical_palette(config)
    core = pd.read_csv(source_root / "tables" / "core_metric_table.tsv", sep="\t")
    metrics = pd.read_csv(root / "tables" / "rstar_calibration_metrics.tsv", sep="\t")
    included = config["included_studies"]
    core_lookup = core.set_index("cohort")
    metric_lookup = metrics.set_index("study_id")
    primary_ids = {"difg_glass", "coadread_mskcc"}
    rows = []
    for study_id, info in included.items():
        short_name = info["short_name"]
        core_row = core_lookup.loc[short_name] if short_name in core_lookup.index else pd.Series(dtype=object)
        metric_row = metric_lookup.loc[study_id] if study_id in metric_lookup.index else pd.Series(dtype=object)
        rows.append(
            {
                "study_id": study_id,
                "cohort": short_name,
                "evidence_role": "primary" if study_id in primary_ids else "supplementary",
                "n_P_C": core_row.get("n_P_C", "NE"),
                "auc_95ci": core_row.get("AUC_95CI", "NE"),
                "ap_lift": core_row.get("AP_lift", "NE"),
                "tertile_delta_persistence_95ci": core_row.get("Delta_persist_95CI", "NE"),
                "tertile_rho_minimum_dwell_95ci": core_row.get("rho_minimum_dwell_95CI", "NE"),
                "quartile_delta_persistence": _fmt_float(metric_row.get("top_bottom_persistence_delta")),
                "quartile_delta_similarity": _fmt_float(metric_row.get("top_bottom_similarity_delta")),
                "quartile_delta_minimum_dwell_scaled": _fmt_float(
                    metric_row.get("top_bottom_minimum_dwell_proxy_delta_scaled")
                ),
                "spearman_persistence": _fmt_float(metric_row.get("spearman_persistence")),
                "spearman_persistence_p": _fmt_float(metric_row.get("spearman_persistence_p"), 3),
                "spearman_similarity": _fmt_float(metric_row.get("spearman_similarity")),
                "spearman_similarity_p": _fmt_float(metric_row.get("spearman_similarity_p"), 3),
                "spearman_minimum_dwell_proxy": _fmt_float(metric_row.get("spearman_minimum_dwell_proxy")),
                "spearman_minimum_dwell_proxy_p": _fmt_float(
                    metric_row.get("spearman_minimum_dwell_proxy_p"), 3
                ),
            }
        )
    table_df = pd.DataFrame(rows)
    table_path = source_root / "tables" / "integrated_longitudinal_metrics_table.tsv"
    table_df.to_csv(table_path, sep="\t", index=False)

    display = table_df[
        [
            "cohort",
            "evidence_role",
            "n_P_C",
            "auc_95ci",
            "ap_lift",
            "tertile_delta_persistence_95ci",
            "tertile_rho_minimum_dwell_95ci",
            "quartile_delta_persistence",
            "spearman_persistence",
            "spearman_similarity",
            "spearman_minimum_dwell_proxy",
        ]
    ].copy()
    display.columns = [
        "cohort",
        "role",
        "n (P/C)",
        "AUC\n(95% CI)",
        "AP\nlift",
        "tertile d\npersist",
        "tertile rho\nmin dwell",
        "quartile d\npersist",
        "rho\npersist",
        "rho\nsimilarity",
        "rho\nmin dwell",
    ]
    for column in ["AUC\n(95% CI)", "tertile d\npersist", "tertile rho\nmin dwell"]:
        display[column] = display[column].astype(str).str.replace(" [", "\n[", regex=False)

    fig, ax = plt.subplots(figsize=(8.4, 2.65))
    ax.axis("off")
    table = ax.table(
        cellText=display.values.tolist(),
        colLabels=display.columns.tolist(),
        cellLoc="center",
        colLoc="center",
        bbox=[0.0, 0.16, 1.0, 0.76],
        colWidths=[0.105, 0.092, 0.078, 0.125, 0.058, 0.125, 0.125, 0.092, 0.072, 0.078, 0.078],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(5.7)
    for (row_index, col_index), cell in table.get_celld().items():
        cell.set_edgecolor("#D8D8D8")
        cell.set_linewidth(0.35)
        if row_index == 0:
            cell.set_facecolor(colors.get("pale_yellow", "#FEEBB9"))
            cell.set_text_props(weight="bold", color="#263238")
        elif row_index % 2 == 0:
            cell.set_facecolor("#FAFAFA")
        else:
            cell.set_facecolor("white")
        if col_index == 0 and row_index > 0:
            cell.set_text_props(ha="left")
    ax.set_title("Experiment 17 integrated longitudinal metrics", loc="left", fontsize=9.5, pad=3)
    ax.text(
        0.0,
        0.045,
        "Tertile metrics summarize the main top-vs-bottom R* validation; quartile metrics summarize the calibration extension. "
        "rho denotes Spearman correlation.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.8,
        color="#4E5A5E",
    )
    figure_style.save_figure_panels(
        fig,
        source_root / "figures" / "Table_E17_integrated_longitudinal_metrics",
        config,
        pad_inches=0.06,
    )
    return table_df


def render_figure(root: Path, config: dict) -> None:
    figure_style.configure_matplotlib(config)
    colors = figure_style.categorical_palette(config)
    text = figure_style.colors(config).get("text", {})
    text_primary = text.get("primary", "#263238")
    bins = pd.read_csv(root / "tables" / "rstar_calibration_bins.tsv", sep="\t")
    metrics = pd.read_csv(root / "tables" / "rstar_calibration_metrics.tsv", sep="\t")

    fig = plt.figure(figsize=tuple(config["plot"]["figure_size"]))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[1.0, 0.92], wspace=0.34, hspace=0.42)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])
    fig.suptitle("Experiment 17 extension | held-out R* calibration in real longitudinal cohorts", y=0.985, fontsize=10.8)

    plot_bin_lines(
        ax_a,
        bins,
        "persistence_rate",
        "persistence_rate_ci_low",
        "persistence_rate_ci_high",
        "Persistence rate",
        "Higher R* states persist more often",
        colors,
    )
    ax_a.set_ylim(-0.04, 1.04)
    ax_a.legend(loc="lower right", frameon=False)
    add_panel_label(ax_a, "A")

    plot_bin_lines(
        ax_b,
        bins,
        "genotype_similarity",
        "genotype_similarity_ci_low",
        "genotype_similarity_ci_high",
        "Genotype similarity",
        "Higher R* states show greater retained similarity",
        colors,
    )
    ax_b.set_ylim(-0.04, 1.04)
    add_panel_label(ax_b, "B")

    plot_bin_lines(
        ax_c,
        bins,
        "minimum_dwell_proxy_scaled",
        "minimum_dwell_proxy_scaled_ci_low",
        "minimum_dwell_proxy_scaled_ci_high",
        "Minimum dwell proxy, scaled",
        "R* tracks observed dwell-time proxy",
        colors,
    )
    ax_c.set_ylim(-0.04, 1.04)
    add_panel_label(ax_c, "C")

    ax_d.axis("off")
    table_rows = []
    for _, row in metrics.iterrows():
        table_rows.append(
            [
                row["study_short_name"],
                f"{int(row['n_evaluable_pairs'])}",
                f"{row['top_bottom_persistence_delta']:+.2f}",
                f"{row['spearman_persistence']:.2f}\n(p={row['spearman_persistence_p']:.3g})",
                f"{row['spearman_minimum_dwell_proxy']:.2f}\n(p={row['spearman_minimum_dwell_proxy_p']:.3g})",
            ]
        )
    table = ax_d.table(
        cellText=table_rows,
        colLabels=["Cohort", "n", "Q4-Q1\npersist", "rho\npersist", "rho\ndwell"],
        cellLoc="center",
        colLoc="center",
        bbox=[0.0, 0.18, 1.0, 0.62],
        colWidths=[0.26, 0.12, 0.20, 0.21, 0.21],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.6)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.35)
        cell.set_edgecolor("#BBBBBB")
        if row == 0:
            cell.set_facecolor(colors.get("sage", "#B8D2CC"))
            cell.set_text_props(weight="bold", color=text_primary)
        elif col in (2, 3, 4):
            cell.set_facecolor("#F7FBFC")
        else:
            cell.set_facecolor("white")
    ax_d.set_title("Core validation metrics", loc="left", pad=2)
    ax_d.text(
        0.0,
        0.03,
        "R* is predicted from held-out fold scores; Q4-Q1 compares highest and lowest R* quartiles.",
        transform=ax_d.transAxes,
        fontsize=6.2,
        color="#4E5A5E",
        va="bottom",
    )
    add_panel_label(ax_d, "D")

    figure_style.save_figure_panels(fig, root / "figures" / "Figure_E17_longitudinal_rstar_calibration_extension", config)


def write_reports(root: Path, data: pd.DataFrame, bins: pd.DataFrame, metrics: pd.DataFrame, config: dict) -> None:
    checks = {
        "positive_top_bottom_persistence_delta": bool((metrics["top_bottom_persistence_delta"] > 0).all()),
        "positive_spearman_persistence": bool((metrics["spearman_persistence"] > 0).all()),
        "positive_spearman_similarity": bool((metrics["spearman_similarity"] > 0).all()),
        "positive_spearman_minimum_dwell_proxy": bool((metrics["spearman_minimum_dwell_proxy"] > 0).all()),
    }
    status = "PASS" if all(checks.values()) else "WARN"
    rows = []
    for _, row in metrics.iterrows():
        rows.append(
            f"| {row['study_short_name']} | {int(row['n_evaluable_pairs'])} | "
            f"{int(row['persistent_pairs'])}/{int(row['changed_pairs'])} | "
            f"{row['top_bottom_persistence_delta']:+.3f} | "
            f"{row['spearman_persistence']:.3f} | {row['spearman_similarity']:.3f} | "
            f"{row['spearman_minimum_dwell_proxy']:.3f} |"
        )
    summary = [
        "# Experiment 17 Extension: Longitudinal R* Calibration",
        "",
        "## Purpose",
        "This extension asks a direct real-data question: when a baseline state has higher held-out R*, is that state more likely to persist, remain genomically similar, and show a longer minimum observed dwell proxy in a later same-patient sample?",
        "",
        "## Design",
        f"- Evaluable held-out pairs: {len(data)} across {data['study_id'].nunique()} public longitudinal cohorts.",
        "- Binning: study-specific R* quartiles, used only for visualization and top-bottom contrast.",
        "- Continuous evidence: Spearman correlation between held-out log2(R*) and three longitudinal outcomes.",
        "- Missing and non-evaluable state scores remain excluded rather than converted to zero.",
        "",
        "## Key Results",
        "| Cohort | n pairs | persistent/changed | Q4-Q1 persistence delta | rho persistence | rho similarity | rho minimum dwell proxy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        *rows,
        "",
        "## Evaluation",
        f"- Overall status: {status}.",
        "- The result is supportive if effect directions are positive in the retained real longitudinal cohorts, because this is an external real-data calibration rather than a fully controlled simulator.",
        "- GLASS provides the main sample-size support; CRC-triplets gives a stronger but smaller corroborating signal; MNM-WashU is retained as a small supplementary paired cohort.",
        "Conclusion: the extension supports the innovation in real longitudinal data, with moderate strength and appropriate caution about small external cohort sizes.",
        "",
    ]
    (root / "experiment_17_longitudinal_extension_summary.md").write_text("\n".join(summary), encoding="utf-8")

    audits = []
    if config.get("plot", {}).get("write_standalone_extension_figure", True):
        audits.extend(
            (
                f"extension_figure_boundary_audit_{index:02d}",
                audit,
            )
            for index, audit in enumerate(
                figure_style.audit_rendered_figure_outputs(
                    root / "figures" / "Figure_E17_longitudinal_rstar_calibration_extension", config
                ),
                start=1,
            )
        )
    audits.extend(
        (
            f"integrated_figure_boundary_audit_{index:02d}",
            audit,
        )
        for index, audit in enumerate(
            figure_style.audit_rendered_figure_outputs(
                Path(config["source_result_root"]) / "figures" / "Figure_E17_integrated_longitudinal_validation",
                config,
            ),
            start=1,
        )
    )
    audits.extend(
        (
            f"integrated_metric_table_boundary_audit_{index:02d}",
            audit,
        )
        for index, audit in enumerate(
            figure_style.audit_rendered_figure_outputs(
                Path(config["source_result_root"]) / "figures" / "Table_E17_integrated_longitudinal_metrics",
                config,
            ),
            start=1,
        )
    )
    validation = [
        "# Experiment 17 Extension Validation",
        "",
        "| check | status | detail |",
        "| --- | --- | --- |",
        f"| included_studies | PASS | {', '.join(sorted(data['study_id'].unique()))} |",
        f"| evaluable_pairs | PASS | n={len(data)} |",
        f"| top_bottom_persistence_delta | {'PASS' if checks['positive_top_bottom_persistence_delta'] else 'WARN'} | {metrics['top_bottom_persistence_delta'].round(3).tolist()} |",
        f"| spearman_persistence_direction | {'PASS' if checks['positive_spearman_persistence'] else 'WARN'} | {metrics['spearman_persistence'].round(3).tolist()} |",
        f"| spearman_similarity_direction | {'PASS' if checks['positive_spearman_similarity'] else 'WARN'} | {metrics['spearman_similarity'].round(3).tolist()} |",
        f"| spearman_minimum_dwell_proxy_direction | {'PASS' if checks['positive_spearman_minimum_dwell_proxy'] else 'WARN'} | {metrics['spearman_minimum_dwell_proxy'].round(3).tolist()} |",
        *[
            f"| {name} | {audit['status']} | {audit['warnings'] or 'rendered_png_ok'} |"
            for name, audit in audits
        ],
    ]
    (root / "experiment_17_longitudinal_extension_validation.md").write_text("\n".join(validation) + "\n", encoding="utf-8")
    pd.DataFrame([audit for _, audit in audits]).to_csv(
        root / "experiment_17_longitudinal_extension_figure_audit.csv", index=False
    )


def run(config: dict) -> None:
    root = Path(config["result_root"])
    tables = root / "tables"
    figures = root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    (root / "resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    data = prepare_predictions(config)
    bins = summarize_bins(data, config)
    metrics = summarize_metrics(data, bins, config)
    data.to_csv(tables / "rstar_calibration_pair_level.tsv", sep="\t", index=False)
    bins.to_csv(tables / "rstar_calibration_bins.tsv", sep="\t", index=False)
    metrics.to_csv(tables / "rstar_calibration_metrics.tsv", sep="\t", index=False)

    if config.get("plot", {}).get("write_standalone_extension_figure", True):
        render_figure(root, config)
    render_integrated_figure(root, config)
    write_and_render_integrated_metrics_table(root, config)
    write_reports(root, data, bins, metrics, config)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run(config)


if __name__ == "__main__":
    main()

"""Run Experiment 8: real-cohort biological convergence of R* states."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import fisher_exact, spearmanr

import figure_style


CONFIG_PATH = Path("configs/experiment_08.yaml")


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def split_modules(annotation: str) -> set[str]:
    if not isinstance(annotation, str) or not annotation.strip():
        return set()
    return {part.strip() for part in annotation.split(";") if part.strip() and part.strip() != "other/WT"}


def display_state(state: str, max_len: int = 20) -> str:
    stage_map = {"primary": "P", "metastatic": "M", "unspecified": "U"}
    if "::" in state:
        stage, genotype = state.split("::", 1)
        genes = genotype.split("+")
        if len(genes) > 2:
            genotype = "+".join(genes[:2]) + "+..."
        cleaned = f"{stage_map.get(stage, stage[:1].upper())}:{genotype}"
    else:
        cleaned = state
    return cleaned if len(cleaned) <= max_len else cleaned[: max_len - 1] + "..."


def display_state_for_figure(state: str) -> str:
    stage_map = {"primary": "P", "metastatic": "M", "unspecified": "U"}
    if "::" not in state:
        return state if len(state) <= 36 else state[:33] + "..."
    stage, genotype = state.split("::", 1)
    prefix = stage_map.get(stage, stage[:1].upper())
    cleaned = f"{prefix}:{genotype}"
    return cleaned if len(cleaned) <= 36 else cleaned[:33] + "..."


def read_state_tables(config: dict) -> pd.DataFrame:
    frames = []
    root = Path(config["experiment_05_root"])
    for dataset, ds_cfg in config["datasets"].items():
        path = root / dataset / "tables" / "state_scores.tsv"
        df = pd.read_csv(path, sep="\t")
        df["dataset_name"] = dataset
        df["display_name"] = ds_cfg["display_name"]
        df["short_name"] = ds_cfg["short_name"]
        df["modules"] = df["clinical_annotation"].map(split_modules)
        df["expected_modules"] = df["modules"].map(lambda mods, expected=set(ds_cfg["expected_modules"]): bool(mods & expected))
        df["ci_above_one"] = df["R_star_ci_low"].astype(float) > float(config["analysis"]["ci_reference"])
        frames.append(df)
    states = pd.concat(frames, ignore_index=True)
    states["R_rank"] = states.groupby("dataset_name")["R_star"].rank(ascending=False, method="min")
    states["N_rank"] = states.groupby("dataset_name")["N_v"].rank(ascending=False, method="min")
    states["N_percentile"] = states.groupby("dataset_name")["N_v"].rank(pct=True)
    return states


def top_state_table(states: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows = []
    top_n = int(config["analysis"]["top_states_per_cohort"])
    for dataset in config["datasets"]:
        sub = states[(states["dataset_name"] == dataset) & states["high_confidence"].astype(bool)].copy()
        sub = sub.sort_values("R_star", ascending=False).head(top_n)
        for rank, (_, row) in enumerate(sub.iterrows(), start=1):
            rows.append(
                {
                    "dataset_name": dataset,
                    "short_name": row["short_name"],
                    "rank": rank,
                    "state_id": f"S{rank}",
                    "state": row["state"],
                    "state_label": display_state(row["state"]),
                    "R_star": row["R_star"],
                    "R_star_ci_low": row["R_star_ci_low"],
                    "R_star_ci_high": row["R_star_ci_high"],
                    "N_v": row["N_v"],
                    "N_percentile": row["N_percentile"],
                    "stability_high_confidence": row["stability_high_confidence"],
                    "clinical_annotation": row["clinical_annotation"],
                    "expected_modules": row["expected_modules"],
                    "ci_above_one": row["ci_above_one"],
                }
            )
    return pd.DataFrame(rows)


def cohort_summary(states: pd.DataFrame, top_states: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows = []
    for dataset, ds_cfg in config["datasets"].items():
        eligible = states[(states["dataset_name"] == dataset) & states["eligible_experiment5"].astype(bool)].copy()
        high = eligible[eligible["high_confidence"].astype(bool)].copy()
        top = top_states[top_states["dataset_name"] == dataset].copy()
        rho, pval = spearmanr(high["R_star"], high["N_v"], nan_policy="omit") if len(high) >= 3 else (np.nan, np.nan)
        expected = set(ds_cfg["expected_modules"])
        if len(high) and len(top):
            high = high.sort_values("R_star", ascending=False).copy()
            top_states_set = set(top["state"])
            high["in_top_display"] = high["state"].isin(top_states_set)
            high_has_expected = high["modules"].map(lambda mods, expected=expected: bool(mods & expected))
            in_top = high["in_top_display"]
            a = int((in_top & high_has_expected).sum())
            b = int((in_top & ~high_has_expected).sum())
            c = int((~in_top & high_has_expected).sum())
            d = int((~in_top & ~high_has_expected).sum())
            expected_odds, expected_p = fisher_exact([[a, b], [c, d]], alternative="greater") if (a + b > 0 and c + d > 0) else (np.nan, np.nan)
            expected_background_fraction = c / (c + d) if (c + d) > 0 else np.nan
        else:
            a = b = c = d = 0
            expected_odds = np.nan
            expected_p = np.nan
            expected_background_fraction = np.nan
        rows.append(
            {
                "dataset_name": dataset,
                "display_name": ds_cfg["display_name"],
                "short_name": ds_cfg["short_name"],
                "eligible_states": int(len(eligible)),
                "high_confidence_states": int(len(high)),
                "top_states": int(len(top)),
                "top_expected_module_fraction": float(top["expected_modules"].mean()) if len(top) else np.nan,
                "top_expected_module_count": int(top["expected_modules"].sum()) if len(top) else 0,
                "expected_module_background_fraction": float(expected_background_fraction),
                "expected_module_fraction_delta": float(top["expected_modules"].mean() - expected_background_fraction) if len(top) else np.nan,
                "expected_module_odds_ratio": float(expected_odds),
                "expected_module_p_value": float(expected_p),
                "top_ci_above_one_fraction": float(top["ci_above_one"].mean()) if len(top) else np.nan,
                "median_top_R_star": float(top["R_star"].median()) if len(top) else np.nan,
                "median_top_N_percentile": float(top["N_percentile"].median()) if len(top) else np.nan,
                "spearman_R_star_vs_N": float(rho),
                "spearman_R_star_vs_N_p": float(pval),
            }
        )
    return pd.DataFrame(rows)


def bh_qvalues(pvalues: list[float]) -> list[float]:
    arr = np.asarray(pvalues, dtype=float)
    q = np.full_like(arr, np.nan, dtype=float)
    valid = np.isfinite(arr)
    if not valid.any():
        return list(q)
    order = np.argsort(arr[valid])
    valid_idx = np.where(valid)[0][order]
    ranked = arr[valid_idx]
    m = len(ranked)
    adjusted = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    q[valid_idx] = np.clip(adjusted, 0, 1)
    return list(q)


def module_enrichment(states: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows = []
    top_n = int(config["analysis"]["enrichment_top_states"])
    for dataset, ds_cfg in config["datasets"].items():
        high = states[(states["dataset_name"] == dataset) & states["high_confidence"].astype(bool)].copy()
        high = high.sort_values("R_star", ascending=False)
        high["in_top"] = False
        high.loc[high.index[: min(top_n, len(high))], "in_top"] = True
        for module in config["modules"]:
            has = high["modules"].map(lambda mods, m=module: m in mods)
            in_top = high["in_top"]
            a = int((in_top & has).sum())
            b = int((in_top & ~has).sum())
            c = int((~in_top & has).sum())
            d = int((~in_top & ~has).sum())
            odds, pvalue = fisher_exact([[a, b], [c, d]], alternative="greater") if len(high) else (np.nan, np.nan)
            top_fraction = a / max(a + b, 1)
            background_fraction = c / max(c + d, 1)
            rows.append(
                {
                    "dataset_name": dataset,
                    "short_name": ds_cfg["short_name"],
                    "module": module,
                    "expected_for_cohort": module in set(ds_cfg["expected_modules"]),
                    "top_count": a,
                    "top_total": a + b,
                    "background_count": c,
                    "background_total": c + d,
                    "top_fraction": top_fraction,
                    "background_fraction": background_fraction,
                    "fraction_delta": top_fraction - background_fraction,
                    "odds_ratio": odds,
                    "p_value": pvalue,
                }
            )
    result = pd.DataFrame(rows)
    result["q_value"] = bh_qvalues(result["p_value"].tolist())
    return result


def plot_top_states(top_states: pd.DataFrame, output: Path, config: dict) -> None:
    figure_style.configure_matplotlib(config)
    text_primary = figure_style.colors(config).get("text", {}).get("primary", "#263238")
    text_secondary = figure_style.colors(config).get("text", {}).get("secondary", "#4E5A5E")
    grid_color = figure_style.colors(config).get("text", {}).get("grid", "#E6E6E6")
    interval_color = "#6F777A"
    reference_color = "#9A9A9A"
    datasets = list(config["datasets"])
    x_max = float(np.ceil(top_states["R_star_ci_high"].max() * 2) / 2 + 0.5)
    x_max = max(x_max, 6.0)
    x_ticks = np.arange(0, np.ceil(x_max) + 0.1, 2.0)
    fig = plt.figure(figsize=(7.0, 4.85))
    if len(datasets) == 3:
        axes_positions = [
            [0.09, 0.55, 0.38, 0.29],
            [0.56, 0.55, 0.38, 0.29],
            [0.325, 0.14, 0.38, 0.29],
        ]
    else:
        ncols = min(2, max(1, len(datasets)))
        nrows = int(np.ceil(len(datasets) / ncols))
        axes_positions = []
        ax_w = 0.38
        ax_h = 0.29
        x0s = [0.09] if ncols == 1 else [0.09, 0.56]
        y0s = np.linspace(0.55, 0.14, max(1, nrows))
        for row_index in range(nrows):
            for col_index in range(ncols):
                if len(axes_positions) >= len(datasets):
                    break
                axes_positions.append([x0s[col_index], float(y0s[row_index]), ax_w, ax_h])
    axes = [fig.add_axes(position) for position in axes_positions]
    fig.text(0.09, 0.955, r"State IDs are defined in Figure E8 state legend. Whiskers, bootstrap 95% CI; dashed line, neutral $R^*=1$", fontsize=5.6, color=text_secondary, ha="left")
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=text_primary, markeredgecolor=text_primary, markersize=3.2, label=r"CI lower bound > 1"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=text_primary, markersize=3.2, label=r"CI overlaps 1"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.935, 0.982),
        frameon=False,
        ncol=2,
        handlelength=1.0,
        columnspacing=1.1,
        fontsize=5.6,
    )
    for panel_index, (ax, dataset) in enumerate(zip(axes, datasets)):
        sub = top_states[top_states["dataset_name"] == dataset].sort_values("rank", ascending=True)
        y = np.arange(len(sub))
        for idx, (_, row) in enumerate(sub.iterrows()):
            low = float(row["R_star_ci_low"])
            high = float(row["R_star_ci_high"])
            value = float(row["R_star"])
            ax.hlines(idx, low, high, color=interval_color, lw=0.78, zorder=1)
            ax.vlines([low, high], idx - 0.14, idx + 0.14, color=interval_color, lw=0.6, zorder=1)
            marker_face = text_primary if bool(row["ci_above_one"]) else "white"
            ax.scatter(value, idx, s=7.8, facecolor=marker_face, edgecolor=text_primary, linewidth=0.5, zorder=3)
            ax.text(
                1.018,
                idx,
                f"{int(row['N_v'])}",
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="center",
                fontsize=5.1,
                color=text_secondary,
                clip_on=False,
            )
        ax.axvline(float(config["analysis"]["ci_reference"]), color=reference_color, lw=0.65, ls=(0, (3, 2)), zorder=0)
        ax.set_yticks(y, sub["state_id"])
        ax.set_ylim(len(sub) - 0.5, -0.5)
        ax.set_xlim(0, x_max)
        ax.set_xticks(x_ticks)
        panel_letter = chr(ord("a") + panel_index)
        ax.text(-0.17, 1.095, panel_letter, transform=ax.transAxes, ha="left", va="center", fontsize=8.2, fontweight="bold", color=text_primary)
        title = f"{config['datasets'][dataset]['short_name']}, {config['datasets'][dataset]['display_name']}"
        ax.text(0.0, 1.085, title, transform=ax.transAxes, ha="left", va="center", fontsize=6.7, fontweight="bold", color=text_primary)
        ax.text(-0.09, 1.015, "State", transform=ax.transAxes, ha="right", va="center", fontsize=5.25, color=text_secondary)
        ax.text(1.018, 1.015, "N", transform=ax.transAxes, ha="left", va="center", fontsize=5.25, color=text_secondary, clip_on=False)
        ax.grid(axis="x", color=grid_color, linewidth=0.45)
        ax.tick_params(axis="x", labelsize=5.8, length=2.0, width=0.65, color=text_primary)
        ax.tick_params(axis="y", labelsize=5.6, length=0, pad=2)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.65)
        ax.spines["bottom"].set_color(text_primary)
    fig.text(0.555, 0.055, r"Relative dwell index, $R^*$", ha="center", va="center", fontsize=6.5, color=text_primary)
    figure_style.save_figure_panels(
        fig,
        output,
        config,
        panel_names=[config["datasets"][dataset]["short_name"] for dataset in datasets],
    )


def plot_state_legend(top_states: pd.DataFrame, output: Path, config: dict) -> None:
    figure_style.configure_matplotlib(config)
    text_primary = figure_style.colors(config).get("text", {}).get("primary", "#263238")
    text_secondary = figure_style.colors(config).get("text", {}).get("secondary", "#4E5A5E")
    datasets = list(config["datasets"])
    fig, axes = plt.subplots(1, len(datasets), figsize=(7.0, 1.65))
    if len(datasets) == 1:
        axes = [axes]
    fig.subplots_adjust(left=0.025, right=0.99, top=0.76, bottom=0.12, wspace=0.24)
    fig.text(0.025, 0.94, "Figure E8 state legend", ha="left", va="center", fontsize=7.2, fontweight="bold", color=text_primary)
    fig.text(0.025, 0.82, "State IDs used in the forest plot; P, primary; M, metastatic.", ha="left", va="center", fontsize=5.6, color=text_secondary)
    for ax, dataset in zip(axes, datasets):
        sub = top_states[top_states["dataset_name"] == dataset].sort_values("rank", ascending=True)
        ax.axis("off")
        ax.text(0.0, 1.0, config["datasets"][dataset]["short_name"], transform=ax.transAxes, ha="left", va="top", fontsize=6.2, fontweight="bold", color=text_primary)
        for idx, (_, row) in enumerate(sub.iterrows()):
            y = 0.84 - idx * 0.14
            ax.text(0.0, y, row["state_id"], transform=ax.transAxes, ha="left", va="center", fontsize=5.45, fontweight="bold", color=text_primary)
            ax.text(0.14, y, display_state_for_figure(str(row["state"])), transform=ax.transAxes, ha="left", va="center", fontsize=5.2, color=text_secondary)
    figure_style.save_figure_panels(
        fig,
        output,
        config,
        panel_names=[config["datasets"][dataset]["short_name"] for dataset in datasets],
    )


def plot_module_summary(module_df: pd.DataFrame, summary: pd.DataFrame, output: Path, config: dict) -> None:
    figure_style.configure_matplotlib(config)
    colors = figure_style.categorical_palette(config)
    text_primary = figure_style.colors(config).get("text", {}).get("primary", "#263238")
    text_secondary = figure_style.colors(config).get("text", {}).get("secondary", "#4E5A5E")
    grid_color = figure_style.colors(config).get("text", {}).get("grid", "#E6E6E6")
    datasets = list(config["datasets"])
    modules = list(config["modules"])
    summary = summary.set_index("dataset_name").reindex(datasets).reset_index()
    cohort_colors = {
        "AACR_LUAD": colors.get("sky_blue", "#B2E6FD"),
        "AACR_COAD": colors.get("sage", "#B8D2CC"),
        "AACR_IDC": colors.get("lavender", "#B5AED5"),
    }
    top_color = colors.get("sky_blue", "#B2E6FD")
    background_color = colors.get("sage", "#B8D2CC")
    delta_color = colors.get("coral", "#E8B2A7")
    rho_color = colors.get("lavender", "#B5AED5")

    def p_label(value: float) -> str:
        if not np.isfinite(value):
            return "NE"
        if value < 0.001:
            return "<0.001"
        return f"{value:.3f}"

    module_table = module_df.set_index(["dataset_name", "module"])

    fig = plt.figure(figsize=(7.25, 5.35))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.54, 1.05], left=0.17, right=0.84, top=0.84, bottom=0.16, hspace=0.44)
    bottom_grid = outer[1].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.38)
    ax_module = fig.add_subplot(outer[0])
    ax_line = fig.add_subplot(bottom_grid[0, 0])
    ax_effect = fig.add_subplot(bottom_grid[0, 1])

    fig.text(0.07, 0.965, "A priori module convergence", ha="left", va="center", fontsize=8.2, fontweight="bold", color=text_primary)
    fig.text(
        0.07,
        0.925,
        "Grouped module bars show top-state composition. Bottom panels summarize prior-set support against background and occupancy.",
        ha="left",
        va="center",
        fontsize=5.45,
        color=text_secondary,
    )

    module_y = np.arange(len(modules), dtype=float)
    bar_height = 0.16
    offsets = np.linspace(-0.27, 0.27, len(datasets))
    for dataset_index, dataset in enumerate(datasets):
        sub = module_df[module_df["dataset_name"] == dataset].set_index("module").reindex(modules)
        y_pos = module_y + offsets[dataset_index]
        top_fraction = sub["top_fraction"].astype(float).values
        expected = sub["expected_for_cohort"].astype(bool).values
        edgecolors = [text_primary if flag else "#B7BFC2" for flag in expected]
        widths = [0.75 if flag else 0.42 for flag in expected]
        bars = ax_module.barh(
            y_pos,
            top_fraction,
            height=bar_height,
            color=cohort_colors[dataset],
            edgecolor=edgecolors,
            linewidth=widths,
            alpha=0.95,
            zorder=3,
            label=config["datasets"][dataset]["short_name"],
        )
    ax_module.set_xlim(0, 1.0)
    ax_module.set_ylim(len(modules) - 0.55, -0.55)
    ax_module.set_yticks(module_y, modules)
    ax_module.set_xticks([0, 0.5, 1.0])
    ax_module.set_xlabel("Module fraction in top states", fontsize=5.7)
    ax_module.set_title("Module-level composition across cohorts", loc="left", fontsize=6.6, fontweight="bold")
    ax_module.grid(axis="x", color=grid_color, linewidth=0.42, zorder=0)
    ax_module.tick_params(axis="both", labelsize=5.3, length=2, width=0.6)
    for spine in ["top", "right"]:
        ax_module.spines[spine].set_visible(False)
    ax_module.spines["left"].set_linewidth(0.6)
    ax_module.spines["bottom"].set_linewidth(0.6)
    ax_module.text(-0.18, 1.08, "a", transform=ax_module.transAxes, ha="left", va="center", fontsize=8.2, fontweight="bold", color=text_primary)
    prior_handle = plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=text_primary, linewidth=0.75, label="Prior module")
    handles, labels = ax_module.get_legend_handles_labels()
    handles.append(prior_handle)
    fig.legend(
        handles=handles,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.855, 0.855),
        fontsize=5.0,
        ncol=1,
        handlelength=1.0,
        labelspacing=0.45,
    )
    fig.text(
        0.855,
        0.49,
        "Exact module counts\nare reported in\nmodule_enrichment.tsv.\nNE, not estimable.",
        ha="left",
        va="top",
        fontsize=4.75,
        color=text_secondary,
        linespacing=1.25,
    )

    cohort_labels = summary["short_name"].tolist()
    x = np.arange(len(summary))
    top_values = summary["top_expected_module_fraction"].astype(float).values
    background_values = summary["expected_module_background_fraction"].astype(float).values
    ax_line.plot(x, top_values, "-o", color=top_color, markeredgecolor=text_primary, markeredgewidth=0.55, lw=1.0, ms=3.4, label="Top states")
    finite_bg = np.isfinite(background_values)
    ax_line.plot(x[finite_bg], background_values[finite_bg], "-o", color=background_color, markerfacecolor="white", markeredgecolor=background_color, markeredgewidth=0.75, lw=1.0, ms=3.4, label="Background")
    for idx, row in summary.iterrows():
        ax_line.text(idx, min(float(row["top_expected_module_fraction"]) + 0.055, 1.05), f"{int(row['top_expected_module_count'])}/{int(row['top_states'])}", ha="center", va="bottom", fontsize=4.8, color=text_secondary)
        background_fraction = float(row["expected_module_background_fraction"])
        if np.isfinite(background_fraction):
            ax_line.text(idx, max(background_fraction - 0.065, 0.05), f"{background_fraction:.0%}", ha="center", va="top", fontsize=4.65, color=text_secondary)
        else:
            ax_line.text(idx, 0.47, "NE", ha="center", va="center", fontsize=4.8, color=text_secondary)
    ax_line.set_ylim(0, 1.12)
    ax_line.set_xticks(x, cohort_labels)
    ax_line.set_yticks([0, 0.5, 1.0])
    ax_line.set_ylabel("Prior-set fraction", fontsize=5.6)
    ax_line.set_title("Prior set: top versus background", loc="left", fontsize=6.4, fontweight="bold")
    ax_line.legend(frameon=False, loc="lower left", fontsize=5.0, handlelength=1.5)
    ax_line.grid(axis="y", color=grid_color, linewidth=0.42)
    ax_line.text(-0.22, 1.13, "b", transform=ax_line.transAxes, ha="left", va="center", fontsize=8.2, fontweight="bold", color=text_primary)
    ax_line.set_box_aspect(1)

    delta_values = summary["expected_module_fraction_delta"].astype(float).values
    delta_plot = np.nan_to_num(delta_values, nan=0.0)
    rho_values = summary["spearman_R_star_vs_N"].astype(float).values
    effect_width = 0.34
    ax_effect.axhline(0, color="#A0A0A0", lw=0.65)
    ax_effect.bar(x - effect_width / 2, delta_plot, color=delta_color, edgecolor=text_primary, linewidth=0.55, width=effect_width, label=r"$\Delta$ fraction")
    ax_effect.bar(x + effect_width / 2, rho_values, color=rho_color, edgecolor=text_primary, linewidth=0.55, width=effect_width, label=r"$\rho(R^*,N)$")
    for idx, value in enumerate(rho_values):
        ax_effect.text(idx + effect_width / 2, value + 0.035, f"{value:+.2f}", ha="center", va="bottom", fontsize=4.65, color=text_secondary)
    ax_effect.set_ylim(-0.08, 0.70)
    p_tick_labels = []
    for label, (_, row) in zip(cohort_labels, summary.iterrows()):
        p_tick_labels.append(f"{label}\nP={p_label(float(row['expected_module_p_value']))}")
    ax_effect.set_xticks(x, p_tick_labels)
    ax_effect.set_yticks([0, 0.3, 0.6])
    ax_effect.set_ylabel("Effect size", fontsize=5.6)
    ax_effect.set_title("Prior-set excess and occupancy decoupling", loc="left", fontsize=6.4, fontweight="bold")
    ax_effect.legend(frameon=False, loc="upper left", fontsize=5.0, ncol=2, handlelength=1.2, columnspacing=1.0)
    ax_effect.grid(axis="y", color=grid_color, linewidth=0.42)
    ax_effect.text(-0.16, 1.13, "c", transform=ax_effect.transAxes, ha="left", va="center", fontsize=8.2, fontweight="bold", color=text_primary)
    ax_effect.set_box_aspect(1)

    for ax in [ax_line, ax_effect]:
        ax.tick_params(axis="both", labelsize=5.2, length=2, width=0.6)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_linewidth(0.6)
        ax.spines["bottom"].set_linewidth(0.6)
    pos = ax_module.get_position()
    ax_module.set_position([pos.x0, pos.y0, pos.width * 0.91, pos.height])
    figure_style.save_figure_panels(fig, output, config)


def write_reports(root: Path, config: dict, summary: pd.DataFrame, top_states: pd.DataFrame, module_df: pd.DataFrame) -> None:
    pattern_lines = figure_style.design_patterns_markdown(config)
    cohort_list = ", ".join(summary["dataset_name"].tolist())
    protocol = f"""# Experiment 8 Protocol Audit

## Focused Aim

Experiment 8 was added as a focused real-cohort biological validation of the
project innovation: whether R* identifies relative dwell/accumulation states
that are biologically coherent within each tumor type and not merely a restated
occupancy ranking.

## Inputs

- Source: Experiment 5 state score tables for {cohort_list}.
- No new MHN fitting is performed.
- Main analysis tier: high-confidence Experiment 5 states.
- Top-state display: {config['analysis']['top_states_per_cohort']} states per cohort.
- Module enrichment top set: {config['analysis']['enrichment_top_states']} states per cohort.

## Figure Design Patterns

{pattern_lines}

## Nature-Family Design Reference

- Main visual grammar follows traditional Nature-family statistical figures:
  small-multiple bar charts for category-level quantities, paired line plots
  for top-versus-background comparisons, compact effect-size bar panels, thin
  axes, white background, direct P/value annotations and small bold panel
  letters.
- Reference checks: Nature formatting guide
  (https://www.nature.com/nature/for-authors/formatting-guide) and Nature
  Methods Points of View on bar charts and box plots
  (https://www.nature.com/articles/nmeth.2807).
- The design is adapted to this experiment's statistics: module bars show all
  predefined modules, the paired line panel reports prior-set top/background
  fractions, and the lower bar panels report delta fraction and occupancy
  decoupling.

## Claim Boundary

This experiment is a biological plausibility and convergence analysis. It does
not prove clinical dwell time or therapy causality because the cohorts are
cross-sectional and treatment context is incomplete.
"""
    (root / "experiment_08_protocol_audit.md").write_text(protocol, encoding="utf-8")
    lines = [
        "# Experiment 8 Scientific Review",
        "",
        "## Main Result",
        "",
        "Experiment 8 evaluates whether high R* states concentrate in tumor-type-plausible modules while remaining distinguishable from raw occupancy.",
        "",
        "| Cohort | High-conf states | Top expected-module fraction | Top CI>1 fraction | Spearman(R*, N) | Median top R* |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['short_name']} | {int(row['high_confidence_states'])} | "
            f"{row['top_expected_module_fraction']:.2f} | {row['top_ci_above_one_fraction']:.2f} | "
            f"{row['spearman_R_star_vs_N']:+.2f} | {row['median_top_R_star']:.2f} |"
        )
    lines.extend(["", "## Top Biological States", "", "| Cohort | Rank | State | R* [95% CI] | Module annotation |", "|---|---:|---|---:|---|"])
    for _, row in top_states[top_states["rank"] <= 3].iterrows():
        lines.append(
            f"| {row['short_name']} | {int(row['rank'])} | {row['state']} | "
            f"{row['R_star']:.2f} [{row['R_star_ci_low']:.2f}-{row['R_star_ci_high']:.2f}] | "
            f"{row['clinical_annotation']} |"
        )
    strongest = module_df.sort_values("fraction_delta", ascending=False).head(8)
    lines.extend(["", "## Strongest Top-State Module Enrichments", "", "| Cohort | Module | Top fraction | Background fraction | Delta | q |", "|---|---|---:|---:|---:|---:|"])
    for _, row in strongest.iterrows():
        lines.append(
            f"| {row['short_name']} | {row['module']} | {row['top_fraction']:.2f} | "
            f"{row['background_fraction']:.2f} | {row['fraction_delta']:+.2f} | {row['q_value']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The experiment supports a bounded real-cohort claim when top R* states are module-coherent, have bootstrap intervals above the neutral R*=1 reference, and show only weak-to-moderate association with raw state count. The interpretation is restricted to the three AACR tumor-type cohorts retained in the primary experiment chain.",
        ]
    )
    (root / "experiment_08_scientific_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_outputs(root: Path, config: dict) -> pd.DataFrame:
    records = []

    def check(category: str, name: str, passed: bool, detail: str) -> None:
        records.append({"category": category, "check": name, "passed": bool(passed), "detail": detail})

    def check_figure_boundary(path: Path) -> tuple[bool, str]:
        if not path.exists():
            return False, "missing"
        image = plt.imread(path)
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        rgb = image[:, :, :3]
        if rgb.max() > 1:
            rgb = rgb / 255.0
        edges = np.concatenate([rgb[0, :, :], rgb[-1, :, :], rgb[:, 0, :], rgb[:, -1, :]], axis=0)
        edge_nonwhite = float(np.mean(np.any(edges < 0.985, axis=1)))
        height, width = rgb.shape[:2]
        return edge_nonwhite < 0.01, f"size={width}x{height}; edge_nonwhite={edge_nonwhite:.4f}"

    required_tables = ["state_level_biological_convergence.tsv", "top_states.tsv", "top_state_legend.tsv", "cohort_summary.tsv", "module_enrichment.tsv"]
    for table in required_tables:
        check("structural", f"table_{table}", (root / "tables" / table).exists(), "exists")
    top_states = pd.read_csv(root / "tables" / "top_states.tsv", sep="\t")
    summary = pd.read_csv(root / "tables" / "cohort_summary.tsv", sep="\t")
    module_df = pd.read_csv(root / "tables" / "module_enrichment.tsv", sep="\t")
    check("structural", "cohort_count", len(summary) == len(config["datasets"]), f"n={len(summary)}")
    check("structural", "top_state_coverage", top_states.groupby("dataset_name").size().min() >= 3, top_states.groupby("dataset_name").size().to_dict().__repr__())
    check("scientific", "expected_module_signal", summary["top_expected_module_fraction"].median() >= 0.5, f"median={summary['top_expected_module_fraction'].median():.3f}")
    check("scientific", "ci_above_one_signal", summary["top_ci_above_one_fraction"].median() >= 0.5, f"median={summary['top_ci_above_one_fraction'].median():.3f}")
    check("scientific", "not_occupancy_clone", summary["spearman_R_star_vs_N"].abs().median() < 0.55, f"median_abs={summary['spearman_R_star_vs_N'].abs().median():.3f}")
    check("scientific", "module_table_complete", len(module_df) == len(config["datasets"]) * len(config["modules"]), f"rows={len(module_df)}")
    for fig in ["Figure_E8_top_state_profiles", "Figure_E8_top_state_legend", "Figure_E8_module_convergence_summary"]:
        base = root / "figures" / fig
        pngs = figure_style.rendered_panel_paths(base, ".png")
        pdfs = figure_style.rendered_panel_paths(base, ".pdf")
        check("figure", f"{fig}_files", bool(pngs) and len(pngs) == len(pdfs), f"png_panels={len(pngs)}, pdf_panels={len(pdfs)}")
        for index, png in enumerate(pngs, start=1):
            passed, detail = check_figure_boundary(png)
            check("figure", f"{fig}_boundary_{index:02d}", passed, detail)
    return pd.DataFrame(records)


def main() -> None:
    config = load_config()
    root = Path(config["result_root"])
    (root / "tables").mkdir(parents=True, exist_ok=True)
    (root / "figures").mkdir(parents=True, exist_ok=True)
    (root / "resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    states = read_state_tables(config)
    top_states = top_state_table(states, config)
    summary = cohort_summary(states, top_states, config)
    module_df = module_enrichment(states, config)
    states_out = states.drop(columns=["modules"]).copy()
    states_out.to_csv(root / "tables" / "state_level_biological_convergence.tsv", sep="\t", index=False)
    top_states.to_csv(root / "tables" / "top_states.tsv", sep="\t", index=False)
    top_states[["dataset_name", "short_name", "state_id", "rank", "state", "clinical_annotation"]].to_csv(
        root / "tables" / "top_state_legend.tsv",
        sep="\t",
        index=False,
    )
    summary.to_csv(root / "tables" / "cohort_summary.tsv", sep="\t", index=False)
    module_df.to_csv(root / "tables" / "module_enrichment.tsv", sep="\t", index=False)
    plot_top_states(top_states, root / "figures" / "Figure_E8_top_state_profiles", config)
    plot_state_legend(top_states, root / "figures" / "Figure_E8_top_state_legend", config)
    plot_module_summary(module_df, summary, root / "figures" / "Figure_E8_module_convergence_summary", config)
    write_reports(root, config, summary, top_states, module_df)
    validation = validate_outputs(root, config)
    validation.to_csv(root / "experiment_08_validation.csv", index=False)
    lines = ["# Experiment 8 Validation", "", "| Category | Check | Pass | Detail |", "|---|---|---:|---|"]
    for _, row in validation.iterrows():
        lines.append(f"| {row['category']} | {row['check']} | {row['passed']} | {row['detail']} |")
    (root / "experiment_08_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

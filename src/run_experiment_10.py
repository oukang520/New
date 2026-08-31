"""Run Experiment 10: real-cohort main results.

This experiment consolidates the real-cohort Rel-ObsTQ-MHN results from
Experiment 5 and the biological annotation logic from Experiment 8. It does not
refit MHN; it turns the already audited state_scores.tsv files into the main
real-cohort evidence tables and figure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

import figure_style


CONFIG_PATH = Path("configs/experiment_10.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Rel-ObsTQ-MHN Experiment 10.")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--result-root")
    return parser.parse_args()


def split_modules(annotation: str) -> set[str]:
    if not isinstance(annotation, str) or not annotation.strip():
        return set()
    return {part.strip() for part in annotation.split(";") if part.strip() and part.strip() != "other/WT"}


def display_state(state: str, max_events: int = 2) -> str:
    stage_map = {"primary": "P", "metastatic": "M", "unspecified": "U"}
    if "::" not in state:
        return state
    stage, genotype = state.split("::", 1)
    prefix = stage_map.get(stage, stage[:1].upper())
    genes = [] if genotype == "WT" else genotype.split("+")
    if len(genes) > max_events:
        genotype = "+".join(genes[:max_events]) + "+..."
    return f"{prefix}:{genotype}"


def read_state_scores(config: dict) -> pd.DataFrame:
    frames = []
    root = Path(config["experiment_05_root"])
    for dataset, ds_cfg in config["datasets"].items():
        path = root / dataset / "tables" / "state_scores.tsv"
        df = pd.read_csv(path, sep="\t")
        df["dataset_name"] = dataset
        df["display_name"] = ds_cfg["display_name"]
        df["short_name"] = ds_cfg["short_name"]
        df["state_label"] = df["state"].map(display_state)
        df["modules"] = df["clinical_annotation"].map(split_modules)
        expected = set(ds_cfg["expected_modules"])
        df["expected_module"] = df["modules"].map(lambda modules, expected=expected: bool(modules & expected))
        df["ci_above_one"] = df["R_star_ci_low"].astype(float) > float(config["analysis"]["ci_reference"])
        df["stable"] = df["eligible_experiment5"].astype(bool)
        df["high_confidence"] = df["high_confidence"].astype(bool)
        df["log10_L_v"] = np.log10(df["L_v"].clip(lower=1e-12))
        df["log10_F_hat"] = np.log10(df["F_hat"].clip(lower=1e-12))
        df["log2_R_star"] = np.log2(df["R_star"].clip(lower=1e-12))
        df["log2_O_star"] = np.log2(df["O_star"].clip(lower=1e-12))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def cohort_summary(states: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows = []
    for dataset, ds_cfg in config["datasets"].items():
        sub = states[states["dataset_name"].eq(dataset)].copy()
        stable = sub[sub["stable"]].copy()
        high = sub[sub["high_confidence"]].copy()
        top = high.sort_values("R_star", ascending=False).head(int(config["analysis"]["top_states_per_cohort"]))
        top_o = stable.sort_values("O_star", ascending=False).head(int(config["analysis"]["top_observation_states_per_cohort"]))
        rows.append(
            {
                "dataset_name": dataset,
                "display_name": ds_cfg["display_name"],
                "short_name": ds_cfg["short_name"],
                "states_total": int(len(sub)),
                "eligible_states": int(len(stable)),
                "high_confidence_states": int(len(high)),
                "top_R_star_median": float(top["R_star"].median()) if len(top) else np.nan,
                "top_R_ci_above_one_fraction": float(top["ci_above_one"].mean()) if len(top) else np.nan,
                "top_expected_module_fraction": float(top["expected_module"].mean()) if len(top) else np.nan,
                "top_O_star_median": float(top_o["O_star"].median()) if len(top_o) else np.nan,
                "states_R_gt_1": int(stable["R_star"].gt(1).sum()),
                "states_O_gt_1": int(stable["O_star"].gt(1).sum()),
            }
        )
    return pd.DataFrame(rows)


def top_state_tables(states: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    top_rows = []
    o_rows = []
    explanation_rows = []
    top_n = int(config["analysis"]["top_states_per_cohort"])
    top_o_n = int(config["analysis"]["top_observation_states_per_cohort"])
    for dataset in config["datasets"]:
        high = states[states["dataset_name"].eq(dataset) & states["high_confidence"]].copy()
        stable = states[states["dataset_name"].eq(dataset) & states["stable"]].copy()
        top = high.sort_values("R_star", ascending=False).head(top_n).copy()
        for rank, (_, row) in enumerate(top.iterrows(), start=1):
            record = {
                "dataset_name": dataset,
                "short_name": row["short_name"],
                "rank": rank,
                "state": row["state"],
                "state_label": row["state_label"],
                "N_v": int(row["N_v"]),
                "L_v": float(row["L_v"]),
                "F_hat": float(row["F_hat"]),
                "R_star": float(row["R_star"]),
                "R_star_ci_low": float(row["R_star_ci_low"]),
                "R_star_ci_high": float(row["R_star_ci_high"]),
                "O_star": float(row["O_star"]),
                "dominant_predecessor": row.get("dominant_predecessor", ""),
                "dominant_predecessor_label": display_state(str(row.get("dominant_predecessor", ""))),
                "dominant_edge_probability": float(row["dominant_edge_probability"]) if pd.notna(row["dominant_edge_probability"]) else np.nan,
                "clinical_annotation": row["clinical_annotation"],
                "expected_module": bool(row["expected_module"]),
                "ci_above_one": bool(row["ci_above_one"]),
                "interpretation_flag": row["interpretation_flag"],
            }
            top_rows.append(record)
            if rank <= 3:
                explanation_rows.append(record)
        top_o = stable.sort_values("O_star", ascending=False).head(top_o_n).copy()
        for rank, (_, row) in enumerate(top_o.iterrows(), start=1):
            o_rows.append(
                {
                    "dataset_name": dataset,
                    "short_name": row["short_name"],
                    "rank": rank,
                    "state": row["state"],
                    "state_label": row["state_label"],
                    "N_v": int(row["N_v"]),
                    "L_v": float(row["L_v"]),
                    "Lhat_progression": float(row["Lhat_progression"]),
                    "O_star": float(row["O_star"]),
                    "R_star": float(row["R_star"]),
                    "clinical_annotation": row["clinical_annotation"],
                    "interpretation_flag": row["interpretation_flag"],
                }
            )
    return pd.DataFrame(top_rows), pd.DataFrame(o_rows), pd.DataFrame(explanation_rows)


def module_matrix(top_states: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows = []
    for dataset, ds_cfg in config["datasets"].items():
        sub = top_states[top_states["dataset_name"].eq(dataset)]
        for module in config["modules"]:
            rows.append(
                {
                    "dataset_name": dataset,
                    "short_name": ds_cfg["short_name"],
                    "module": module,
                    "expected_for_cohort": module in set(ds_cfg["expected_modules"]),
                    "top_fraction": float(sub["clinical_annotation"].map(lambda x, m=module: m in split_modules(x)).mean()) if len(sub) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def square_save(fig: plt.Figure, output: Path, config: dict) -> None:
    figure_style.save_figure(fig, output, config)


def save_figure(fig: plt.Figure, output: Path, config: dict) -> None:
    figure_style.save_figure(fig, output, config)


def cleanup_figure_outputs(base_path: Path) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".png", ".pdf"]:
        candidate = base_path.with_suffix(suffix)
        if candidate.exists():
            candidate.unlink()
    for candidate in base_path.parent.glob(f"{base_path.name}__*"):
        if candidate.suffix.lower() in {".png", ".pdf"}:
            candidate.unlink()


def plot_main_figure(
    states: pd.DataFrame,
    top_states: pd.DataFrame,
    module_summary: pd.DataFrame,
    output: Path,
    config: dict,
) -> None:
    figure_style.configure_matplotlib(config)
    colors = figure_style.colors(config)
    cat = figure_style.categorical_palette(config)
    text_primary = colors.get("text", {}).get("primary", "#263238")
    text_secondary = colors.get("text", {}).get("secondary", "#4E5A5E")
    grid_color = colors.get("text", {}).get("grid", "#E6E6E6")
    cohort_colors = {
        "LUAD": cat.get("lavender", "#B5AED5"),
        "COAD": cat.get("sky_blue", "#B2E6FD"),
        "IDC": cat.get("sage", "#B8D2CC"),
    }
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rstar_project",
        [cat.get("pale_yellow", "#FEEBB9"), cat.get("sage", "#B8D2CC"), cat.get("sky_blue", "#B2E6FD"), cat.get("lavender", "#B5AED5")],
    )
    fig = plt.figure(figsize=(7.2, 7.2))
    gs = fig.add_gridspec(2, 2, left=0.085, right=0.985, bottom=0.08, top=0.88, wspace=0.34, hspace=0.42)
    gs_a = gs[0, 0].subgridspec(2, 2, wspace=0.18, hspace=0.28)
    ax_landscape = [fig.add_subplot(gs_a[i, j]) for i in range(2) for j in range(2)]
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    datasets = list(config["datasets"])
    for idx, (ax, dataset) in enumerate(zip(ax_landscape, datasets)):
        ds = config["datasets"][dataset]
        sub = states[states["dataset_name"].eq(dataset) & states["stable"]].copy()
        max_n = int(config["analysis"]["max_scatter_states_per_cohort"])
        if len(sub) > max_n:
            sub = sub.nlargest(max_n, "N_v")
        sc = ax.scatter(
            sub["log10_F_hat"],
            sub["log10_L_v"],
            c=sub["log2_R_star"].clip(-1.5, 3.0),
            cmap=cmap,
            vmin=-1.5,
            vmax=3.0,
            s=np.clip(np.sqrt(sub["N_v"].astype(float)) * 1.2, 5, 30),
            edgecolor="white",
            linewidth=0.18,
            alpha=0.78,
            zorder=2,
        )
        highlight = top_states[top_states["dataset_name"].eq(dataset)].head(3)
        h = sub[sub["state"].isin(highlight["state"])]
        ax.scatter(
            h["log10_F_hat"],
            h["log10_L_v"],
            s=np.clip(np.sqrt(h["N_v"].astype(float)) * 1.8, 18, 45),
            facecolor="none",
            edgecolor=text_primary,
            linewidth=0.65,
            zorder=4,
        )
        ax.set_title(ds["short_name"], loc="left", fontsize=6.8, pad=2)
        ax.grid(color=grid_color, lw=0.35)
        ax.tick_params(labelsize=5.6, length=1.8, width=0.55)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.set_box_aspect(1)
        if idx in [2, 3]:
            ax.set_xlabel(r"$\log_{10}\hat F$", fontsize=6.0)
        else:
            ax.set_xticklabels([])
        if idx in [0, 2]:
            ax.set_ylabel(r"$\log_{10}L$", fontsize=6.0)
        else:
            ax.set_yticklabels([])
    ax_landscape[0].text(-0.44, 1.28, "a", transform=ax_landscape[0].transAxes, fontsize=10.5, fontweight="bold", ha="left", va="top", color=text_primary)
    ax_landscape[0].text(-0.02, 1.28, r"State landscapes: occupancy $L$ vs inflow $\hat F$", transform=ax_landscape[0].transAxes, fontsize=8.1, color=text_primary, ha="left", va="top")
    cax = fig.add_axes([0.205, 0.486, 0.155, 0.008])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=4.8, length=1.4, width=0.4)
    cb.set_label(r"$\log_2R^*$", fontsize=5.0, labelpad=-1)

    forest_n = int(config["analysis"]["forest_states_per_cohort"])
    dataset_order = {dataset: index for index, dataset in enumerate(config["datasets"])}
    forest = top_states.groupby("dataset_name", group_keys=False).head(forest_n).copy()
    forest["dataset_order"] = forest["dataset_name"].map(dataset_order)
    forest = forest.sort_values(["dataset_order", "rank"], ascending=[True, True]).reset_index(drop=True)
    y = np.arange(len(forest))
    for idx, row in forest.iterrows():
        color = cohort_colors.get(row["short_name"], "#B8D2CC")
        ax_b.hlines(idx, row["R_star_ci_low"], row["R_star_ci_high"], color=text_primary, lw=0.65, zorder=1)
        ax_b.vlines([row["R_star_ci_low"], row["R_star_ci_high"]], idx - 0.16, idx + 0.16, color=text_primary, lw=0.5, zorder=1)
        ax_b.scatter(row["R_star"], idx, s=21, color=color, edgecolor=text_primary, linewidth=0.45, zorder=3)
    ax_b.axvline(1.0, color="#999999", lw=0.65, ls=(0, (3, 2)))
    ax_b.set_yticks(y, [f"{r.short_name} {r.state_label}" for r in forest.itertuples()], fontsize=5.5)
    ax_b.invert_yaxis()
    ax_b.set_xlabel(r"Relative dwell, $R^*$")
    ax_b.set_title(r"Top high-confidence bottleneck states", loc="left", fontsize=8.1, pad=4)
    ax_b.text(-0.18, 1.08, "b", transform=ax_b.transAxes, fontsize=10.5, fontweight="bold", ha="left", va="top", color=text_primary)
    ax_b.grid(axis="x", color=grid_color, lw=0.45)
    for spine in ["top", "right"]:
        ax_b.spines[spine].set_visible(False)

    stable = states[states["stable"]].copy()
    stable["quad"] = np.select(
        [
            stable["R_star"].gt(1) & stable["O_star"].gt(1),
            stable["R_star"].gt(1) & stable["O_star"].le(1),
            stable["R_star"].le(1) & stable["O_star"].gt(1),
        ],
        ["R+ O+", "R+ only", "O+ only"],
        default="other",
    )
    quad_colors = {
        "R+ O+": cat.get("coral", "#E8B2A7"),
        "R+ only": cat.get("lavender", "#B5AED5"),
        "O+ only": cat.get("sky_blue", "#B2E6FD"),
        "other": cat.get("sage", "#B8D2CC"),
    }
    for quad, alpha, size in [("other", 0.34, 8), ("O+ only", 0.62, 10), ("R+ only", 0.62, 10), ("R+ O+", 0.70, 12)]:
        sub = stable[stable["quad"].eq(quad)]
        ax_c.scatter(
            sub["log2_R_star"].clip(-3, 4),
            sub["log2_O_star"].clip(-2, 4),
            s=size,
            color=quad_colors[quad],
            edgecolor="white",
            linewidth=0.12,
            alpha=alpha,
            label=quad,
        )
    ax_c.axhline(0, color="#999999", lw=0.65, ls=(0, (3, 2)))
    ax_c.axvline(0, color="#999999", lw=0.65, ls=(0, (3, 2)))
    ax_c.set_xlabel(r"Relative dwell, $\log_2R^*$")
    ax_c.set_ylabel(r"Observation enrichment, $\log_2O^*$")
    ax_c.set_title(r"Real states separate dwell and observation enrichment", loc="left", fontsize=8.1, pad=4)
    ax_c.text(-0.16, 1.08, "c", transform=ax_c.transAxes, fontsize=10.5, fontweight="bold", ha="left", va="top", color=text_primary)
    ax_c.legend(frameon=False, loc="upper left", fontsize=5.4, ncol=2, handlelength=0.8, columnspacing=0.8)
    ax_c.grid(color=grid_color, lw=0.45)
    for spine in ["top", "right"]:
        ax_c.spines[spine].set_visible(False)
    ax_c.set_box_aspect(1)

    cohort_order = [ds_cfg["short_name"] for ds_cfg in config["datasets"].values()]
    heat = (
        module_summary.pivot(index="module", columns="short_name", values="top_fraction")
        .reindex(index=config["modules"], columns=cohort_order)
    )
    ax_d.imshow(heat.to_numpy(dtype=float), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax_d.set_xticks(np.arange(len(heat.columns)), heat.columns, fontsize=6.2)
    ax_d.set_yticks(np.arange(len(heat.index)), heat.index, fontsize=5.6)
    for i, module in enumerate(heat.index):
        for j, cohort in enumerate(heat.columns):
            value = heat.loc[module, cohort]
            if np.isfinite(value):
                ax_d.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=5.1, color=text_primary)
    for _, row in module_summary[module_summary["expected_for_cohort"]].iterrows():
        i = list(heat.index).index(row["module"])
        j = list(heat.columns).index(row["short_name"])
        rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=text_primary, lw=0.55)
        ax_d.add_patch(rect)
    ax_d.set_title("Top-state biological modules", loc="left", fontsize=8.1, pad=4)
    ax_d.text(-0.18, 1.08, "d", transform=ax_d.transAxes, fontsize=10.5, fontweight="bold", ha="left", va="top", color=text_primary)
    ax_d.tick_params(length=0)
    ax_d.set_box_aspect(1)
    ax_d.text(0.0, -0.18, "Cell value: fraction among top R* states; boxes, cohort-prior modules.", transform=ax_d.transAxes, fontsize=5.4, color=text_secondary, ha="left", va="top")

    fig.text(0.085, 0.972, "Experiment 10 | Real-cohort Rel-ObsTQ-MHN main results", ha="left", va="top", fontsize=9.4, fontweight="bold", color=text_primary)
    fig.text(0.085, 0.947, r"Selected AACR cohorts; state-level $R^*$, $O^*$ and biological modules from audited Experiment 5/8 outputs", ha="left", va="top", fontsize=5.9, color=text_secondary)
    square_save(fig, output, config)


def plot_main_figure_single(
    states: pd.DataFrame,
    top_states: pd.DataFrame,
    module_summary: pd.DataFrame,
    output: Path,
    config: dict,
) -> None:
    figure_style.configure_matplotlib(config)
    colors = figure_style.colors(config)
    cat = figure_style.categorical_palette(config)
    text_primary = colors.get("text", {}).get("primary", "#263238")
    grid_color = colors.get("text", {}).get("grid", "#E6E6E6")
    cohort_colors = {
        "LUAD": cat.get("lavender", "#B5AED5"),
        "COAD": cat.get("sky_blue", "#B2E6FD"),
        "IDC": cat.get("sage", "#B8D2CC"),
    }
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rstar_project",
        [
            cat.get("pale_yellow", "#FEEBB9"),
            cat.get("sage", "#B8D2CC"),
            cat.get("sky_blue", "#B2E6FD"),
            cat.get("lavender", "#B5AED5"),
        ],
    )
    cleanup_figure_outputs(output)

    for dataset, ds in config["datasets"].items():
        sub = states[states["dataset_name"].eq(dataset) & states["stable"]].copy()
        max_n = int(config["analysis"]["max_scatter_states_per_cohort"])
        if len(sub) > max_n:
            sub = sub.nlargest(max_n, "N_v")
        fig, ax = plt.subplots(figsize=(3.15, 3.15))
        fig.subplots_adjust(left=0.17, right=0.83, bottom=0.17, top=0.94)
        sc = ax.scatter(
            sub["log10_F_hat"],
            sub["log10_L_v"],
            c=sub["log2_R_star"].clip(-1.5, 3.0),
            cmap=cmap,
            vmin=-1.5,
            vmax=3.0,
            s=np.clip(np.sqrt(sub["N_v"].astype(float)) * 1.2, 5, 30),
            edgecolor="white",
            linewidth=0.18,
            alpha=0.78,
            zorder=2,
        )
        highlight = top_states[top_states["dataset_name"].eq(dataset)].head(3)
        h = sub[sub["state"].isin(highlight["state"])]
        ax.scatter(
            h["log10_F_hat"],
            h["log10_L_v"],
            s=np.clip(np.sqrt(h["N_v"].astype(float)) * 1.8, 18, 45),
            facecolor="none",
            edgecolor=text_primary,
            linewidth=0.65,
            zorder=4,
        )
        ax.set_xlabel(r"$\log_{10}\hat F$")
        ax.set_ylabel(r"$\log_{10}L$")
        ax.grid(color=grid_color, lw=0.45)
        ax.set_box_aspect(1)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        cb = fig.colorbar(sc, ax=ax, pad=0.035, fraction=0.052)
        cb.set_label(r"$\log_2R^*$")
        save_figure(fig, output.with_name(f"{output.name}__{ds['short_name'].lower()}"), config)

    forest_n = int(config["analysis"]["forest_states_per_cohort"])
    dataset_order = {dataset: index for index, dataset in enumerate(config["datasets"])}
    forest = top_states.groupby("dataset_name", group_keys=False).head(forest_n).copy()
    forest["dataset_order"] = forest["dataset_name"].map(dataset_order)
    forest = forest.sort_values(["dataset_order", "rank"], ascending=[True, True]).reset_index(drop=True)
    y = np.arange(len(forest))
    fig, ax = plt.subplots(figsize=(3.95, 4.10))
    fig.subplots_adjust(left=0.43, right=0.96, bottom=0.12, top=0.96)
    for idx, row in forest.iterrows():
        color = cohort_colors.get(row["short_name"], "#B8D2CC")
        ax.hlines(idx, row["R_star_ci_low"], row["R_star_ci_high"], color=text_primary, lw=0.65, zorder=1)
        ax.vlines([row["R_star_ci_low"], row["R_star_ci_high"]], idx - 0.16, idx + 0.16, color=text_primary, lw=0.5, zorder=1)
        ax.scatter(row["R_star"], idx, s=21, color=color, edgecolor=text_primary, linewidth=0.45, zorder=3)
    finite_ci = forest[["R_star_ci_low", "R_star_ci_high"]].to_numpy(dtype=float)
    finite_ci = finite_ci[np.isfinite(finite_ci)]
    if finite_ci.size:
        x_min, x_max = float(finite_ci.min()), float(finite_ci.max())
        pad = max((x_max - x_min) * 0.08, 0.12)
        ax.set_xlim(max(0.0, x_min - pad), x_max + pad)
    ax.axvline(1.0, color="#999999", lw=0.65, ls=(0, (3, 2)))
    ax.set_yticks(y, [f"{r.short_name} {r.state_label}" for r in forest.itertuples()])
    ax.tick_params(axis="y", labelsize=5.5)
    ax.invert_yaxis()
    ax.set_xlabel(r"Relative dwell, $R^*$")
    ax.grid(axis="x", color=grid_color, lw=0.45)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    save_figure(fig, output.with_name(f"{output.name}__top_high_confidence_bottleneck_states"), config)

    stable = states[states["stable"]].copy()
    stable["quad"] = np.select(
        [
            stable["R_star"].gt(1) & stable["O_star"].gt(1),
            stable["R_star"].gt(1) & stable["O_star"].le(1),
            stable["R_star"].le(1) & stable["O_star"].gt(1),
        ],
        ["R+ O+", "R+ only", "O+ only"],
        default="other",
    )
    quad_colors = {
        "R+ O+": cat.get("coral", "#E8B2A7"),
        "R+ only": cat.get("lavender", "#B5AED5"),
        "O+ only": cat.get("sky_blue", "#B2E6FD"),
        "other": cat.get("sage", "#B8D2CC"),
    }
    fig, ax = plt.subplots(figsize=(3.25, 3.25))
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.17, top=0.96)
    for quad, alpha, size in [("other", 0.34, 8), ("O+ only", 0.62, 10), ("R+ only", 0.62, 10), ("R+ O+", 0.70, 12)]:
        sub = stable[stable["quad"].eq(quad)]
        ax.scatter(
            sub["log2_R_star"].clip(-3, 4),
            sub["log2_O_star"].clip(-2, 4),
            s=size,
            color=quad_colors[quad],
            edgecolor="white",
            linewidth=0.12,
            alpha=alpha,
            label=quad,
        )
    ax.axhline(0, color="#999999", lw=0.65, ls=(0, (3, 2)))
    ax.axvline(0, color="#999999", lw=0.65, ls=(0, (3, 2)))
    ax.set_xlabel(r"Relative dwell, $\log_2R^*$")
    ax.set_ylabel(r"Observation enrichment, $\log_2O^*$")
    ax.legend(frameon=False, loc="upper left", fontsize=5.4, ncol=2, handlelength=0.8, columnspacing=0.8)
    ax.grid(color=grid_color, lw=0.45)
    ax.set_box_aspect(1)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    save_figure(fig, output.with_name(f"{output.name}__real_states_separate_dwell_and_observation_enrichment"), config)

    cohort_order = [ds_cfg["short_name"] for ds_cfg in config["datasets"].values()]
    heat = (
        module_summary.pivot(index="module", columns="short_name", values="top_fraction")
        .reindex(index=config["modules"], columns=cohort_order)
    )
    fig, ax = plt.subplots(figsize=(3.35, 3.35))
    fig.subplots_adjust(left=0.37, right=0.98, bottom=0.13, top=0.96)
    ax.imshow(heat.to_numpy(dtype=float), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(heat.columns)), heat.columns, fontsize=6.2)
    ax.set_yticks(np.arange(len(heat.index)), heat.index, fontsize=5.6)
    for i, module in enumerate(heat.index):
        for j, cohort in enumerate(heat.columns):
            value = heat.loc[module, cohort]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=5.1, color=text_primary)
    for _, row in module_summary[module_summary["expected_for_cohort"]].iterrows():
        i = list(heat.index).index(row["module"])
        j = list(heat.columns).index(row["short_name"])
        rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=text_primary, lw=0.55)
        ax.add_patch(rect)
    ax.tick_params(length=0)
    ax.set_box_aspect(1)
    save_figure(fig, output.with_name(f"{output.name}__top_state_biological_modules"), config)


def write_reports(root: Path, config: dict, summary: pd.DataFrame, top_states: pd.DataFrame, top_o: pd.DataFrame) -> None:
    cohort_list = ", ".join(config["datasets"].keys())
    protocol = f"""# Experiment 10 Protocol Audit

## Protocol Section

Source document section: `16. 实验 10：真实队列主结果`.

Purpose: show in real cross-sectional cancer cohorts that R* and O* identify
stage-genotype states with biological interpretation.

## Inputs Used

- Experiment 5 `state_scores.tsv` for {cohort_list}.
- Experiment 8 cohort definitions and expected tumor-type modules.
- No new MHN fitting is performed in Experiment 10.

## Cohort Boundary

In this project the feasible, preselected real-cohort set is {cohort_list},
following the earlier dataset feasibility and extraction decisions.

## Figure Design Patterns

{figure_style.design_patterns_markdown(config)}
"""
    (root / "experiment_10_protocol_audit.md").write_text(protocol, encoding="utf-8")

    lines = [
        "# Experiment 10 Summary",
        "",
        "| Cohort | Eligible states | High-confidence states | Median top R* | Top CI>1 fraction | Top expected-module fraction | Median top O* |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples():
        lines.append(
            f"| {row.short_name} | {row.eligible_states} | {row.high_confidence_states} | {row.top_R_star_median:.3f} | {row.top_R_ci_above_one_fraction:.2f} | {row.top_expected_module_fraction:.2f} | {row.top_O_star_median:.3f} |"
        )
    (root / "experiment_10_summary.md").write_text("\n".join(lines), encoding="utf-8")

    sci = [
        "# Experiment 10 Scientific Review",
        "",
        "## Main Result",
        "",
        "Experiment 10 consolidates the real-cohort evidence: high R* states are extracted from each cohort, checked against bootstrap confidence intervals, and interpreted with tumor-type modules and O* residuals.",
        "",
        "## Top R* States",
        "",
        "| Cohort | Rank | State | R* [95% CI] | O* | N | Dominant predecessor | Module annotation |",
        "|---|---:|---|---:|---:|---:|---|---|",
    ]
    for row in top_states.groupby("dataset_name", group_keys=False).head(3).itertuples():
        sci.append(
            f"| {row.short_name} | {row.rank} | {row.state} | {row.R_star:.2f} [{row.R_star_ci_low:.2f}-{row.R_star_ci_high:.2f}] | {row.O_star:.2f} | {row.N_v} | {row.dominant_predecessor} | {row.clinical_annotation} |"
        )
    sci.extend(
        [
            "",
            "## Top O* States",
            "",
            "| Cohort | Rank | State | O* | R* | Interpretation |",
            "|---|---:|---|---:|---:|---|",
        ]
    )
    for row in top_o.groupby("dataset_name", group_keys=False).head(3).itertuples():
        sci.append(
            f"| {row.short_name} | {row.rank} | {row.state} | {row.O_star:.2f} | {row.R_star:.2f} | {row.interpretation_flag} |"
        )
    sci.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "R* is interpreted as relative state dwell/accumulation after controlling for model-derived inflow. O* is a progression-only occupancy residual and is not a clinical diagnosis rate. The interpretation is restricted to the selected AACR tumor-type cohorts.",
        ]
    )
    (root / "experiment_10_scientific_review.md").write_text("\n".join(sci), encoding="utf-8")

    design = f"""# Experiment 10 Figure Design Review

## Sources

{figure_style.design_sources_markdown(config)}

## Rules Applied

{figure_style.design_rules_markdown(config)}

## Design Choices

- The main figure is square and keeps four compact panels.
- Panel A follows the protocol requirement to show why top R* states are not
  simply high-frequency states: it plots occupancy L against inflow F_hat.
- Panel B reports exact bootstrap intervals for top high-confidence bottleneck
  states.
- Panel C separates relative dwell R* from observation enrichment O*.
- Panel D compresses tumor-type biological interpretation into a module matrix.
"""
    (root / "top_journal_figure_design_review.md").write_text(design, encoding="utf-8")


def save_resolved_config(root: Path, config: dict) -> None:
    (root / "resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.result_root:
        config["result_root"] = args.result_root
    root = Path(config["result_root"]).resolve()
    tables = root / "tables"
    figures = root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    save_resolved_config(root, config)

    states = read_state_scores(config)
    summary = cohort_summary(states, config)
    top_states, top_o, explanation = top_state_tables(states, config)
    modules = module_matrix(top_states, config)

    states.to_csv(tables / "combined_state_scores.tsv", sep="\t", index=False)
    summary.to_csv(tables / "cohort_main_summary.tsv", sep="\t", index=False)
    top_states.to_csv(tables / "top_real_cohort_bottleneck_states.tsv", sep="\t", index=False)
    top_o.to_csv(tables / "top_real_cohort_observation_enriched_states.tsv", sep="\t", index=False)
    explanation.to_csv(tables / "top_state_explanation_table.tsv", sep="\t", index=False)
    modules.to_csv(tables / "top_state_module_matrix.tsv", sep="\t", index=False)

    plot_main_figure_single(states, top_states, modules, figures / "Figure_E10_real_cohort_main_results", config)
    write_reports(root, config, summary, top_states, top_o)


if __name__ == "__main__":
    main()

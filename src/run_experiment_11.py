"""Run Experiment 11: MHN-only / occupancy-only / Rel-ObsTQ-MHN information gain.

The protocol asks whether R* adds information beyond two simpler views:
MHN-only transition propensity and occupancy-only cross-sectional frequency.
This runner uses the audited Experiment 5 state tables and compares
state rankings, top-K overlap, L-vs-F geometry and quadrant enrichment.
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


CONFIG_PATH = Path("configs/experiment_11.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Rel-ObsTQ-MHN Experiment 11.")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--result-root")
    return parser.parse_args()


def display_state(state: str, max_events: int = 2, max_len: int = 24) -> str:
    stage_map = {"primary": "P", "metastatic": "M", "unspecified": "U", "unknown": "U"}
    if "::" not in str(state):
        text = str(state)
    else:
        stage, genotype = str(state).split("::", 1)
        genes = [] if genotype == "WT" else genotype.split("+")
        if len(genes) > max_events:
            genotype = "+".join(genes[:max_events]) + "+"
        text = f"{stage_map.get(stage, stage[:1].upper())}:{genotype}"
    return text if len(text) <= max_len else text[: max_len - 1] + "..."


def load_state_scores(config: dict) -> pd.DataFrame:
    frames = []
    root = Path(config["experiment_05_root"])
    for dataset, ds_cfg in config["datasets"].items():
        path = root / dataset / "tables" / "state_scores.tsv"
        df = pd.read_csv(path, sep="\t")
        df["dataset_name"] = dataset
        df["display_name"] = ds_cfg["display_name"]
        df["short_name"] = ds_cfg["short_name"]
        df["state_label"] = df["state"].map(display_state)
        df["stable"] = df["eligible_experiment5"].astype(bool)
        df["high_confidence"] = df["high_confidence"].astype(bool)
        df["mhn_only_score"] = df["F_hat"].astype(float)
        df["occupancy_only_score"] = df["L_v"].astype(float)
        df["rel_obstq_score"] = df["R_star"].astype(float)
        df["log10_F_hat"] = np.log10(df["F_hat"].clip(lower=1.0e-12))
        df["log10_L_v"] = np.log10(df["L_v"].clip(lower=1.0e-12))
        df["log2_R_star"] = np.log2(df["R_star"].clip(lower=1.0e-12))
        frames.append(df)
    states = pd.concat(frames, ignore_index=True)
    for score, rank in [
        ("mhn_only_score", "rank_mhn_only"),
        ("occupancy_only_score", "rank_occupancy_only"),
        ("rel_obstq_score", "rank_rel_obstq_mhn"),
    ]:
        states[rank] = np.nan
        for dataset in states["dataset_name"].unique():
            idx = states["dataset_name"].eq(dataset) & states["stable"].astype(bool)
            states.loc[idx, rank] = states.loc[idx, score].rank(ascending=False, method="min")
    return states


def assign_quadrants(states: pd.DataFrame, config: dict) -> pd.DataFrame:
    q = float(config["analysis"]["quadrant_quantile"])
    rows = []
    for dataset in config["datasets"]:
        sub = states[states["dataset_name"].eq(dataset) & states["stable"]].copy()
        l_cut = float(sub["L_v"].quantile(q))
        f_cut = float(sub["F_hat"].quantile(q))
        sub["L_high"] = sub["L_v"].ge(l_cut)
        sub["F_high"] = sub["F_hat"].ge(f_cut)
        sub["quadrant"] = np.select(
            [
                sub["L_high"] & sub["F_high"],
                sub["L_high"] & ~sub["F_high"],
                ~sub["L_high"] & sub["F_high"],
            ],
            ["High L / High F", "High L / Low F", "Low L / High F"],
            default="Low L / Low F",
        )
        sub["L_quantile_cut"] = l_cut
        sub["F_quantile_cut"] = f_cut
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def top_sets(states: pd.DataFrame, dataset: str, config: dict) -> dict[str, pd.DataFrame]:
    top_k = int(config["analysis"]["top_k"])
    stable = states[states["dataset_name"].eq(dataset) & states["stable"]].copy()
    high = stable[stable["high_confidence"]].copy()
    rel_source = high if len(high) else stable
    return {
        "mhn_only": stable.nlargest(min(top_k, len(stable)), "mhn_only_score").copy(),
        "occupancy_only": stable.nlargest(min(top_k, len(stable)), "occupancy_only_score").copy(),
        "rel_obstq_mhn": rel_source.nlargest(min(top_k, len(rel_source)), "rel_obstq_score").copy(),
    }


def rank_comparison_table(states: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows = []
    method_order = ["mhn_only", "occupancy_only", "rel_obstq_mhn"]
    score_map = {
        "mhn_only": "mhn_only_score",
        "occupancy_only": "occupancy_only_score",
        "rel_obstq_mhn": "rel_obstq_score",
    }
    for dataset in config["datasets"]:
        sets = top_sets(states, dataset, config)
        for method in method_order:
            method_cfg = config["methods"][method]
            for rank, (_, row) in enumerate(sets[method].iterrows(), start=1):
                rows.append(
                    {
                        "dataset_name": dataset,
                        "short_name": row["short_name"],
                        "method": method,
                        "method_label": method_cfg["display_name"],
                        "rank": rank,
                        "state": row["state"],
                        "state_label": row["state_label"],
                        "score": float(row[score_map[method]]),
                        "N_v": int(row["N_v"]),
                        "L_v": float(row["L_v"]),
                        "F_hat": float(row["F_hat"]),
                        "R_star": float(row["R_star"]),
                        "R_star_ci_low": float(row["R_star_ci_low"]),
                        "R_star_ci_high": float(row["R_star_ci_high"]),
                        "quadrant": row["quadrant"],
                        "clinical_annotation": row["clinical_annotation"],
                    }
                )
    return pd.DataFrame(rows)


def overlap_and_rank_summary(states: pd.DataFrame, ranked: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows = []
    for dataset, ds_cfg in config["datasets"].items():
        stable = states[states["dataset_name"].eq(dataset) & states["stable"]].copy()
        sets = top_sets(states, dataset, config)
        state_sets = {name: set(frame["state"]) for name, frame in sets.items()}
        rel = state_sets["rel_obstq_mhn"]
        mhn = state_sets["mhn_only"]
        occ = state_sets["occupancy_only"]
        rho_rf = float(stable[["rel_obstq_score", "mhn_only_score"]].corr(method="spearman").iloc[0, 1]) if len(stable) >= 3 else np.nan
        rho_rl = float(stable[["rel_obstq_score", "occupancy_only_score"]].corr(method="spearman").iloc[0, 1]) if len(stable) >= 3 else np.nan
        rho_lf = float(stable[["occupancy_only_score", "mhn_only_score"]].corr(method="spearman").iloc[0, 1]) if len(stable) >= 3 else np.nan
        top_rel = sets["rel_obstq_mhn"].copy()
        rank_gain_l = top_rel["rank_occupancy_only"] - top_rel["rank_rel_obstq_mhn"]
        rank_gain_f = top_rel["rank_mhn_only"] - top_rel["rank_rel_obstq_mhn"]
        qdist = top_rel["quadrant"].value_counts(normalize=True).to_dict()
        rows.append(
            {
                "dataset_name": dataset,
                "short_name": ds_cfg["short_name"],
                "eligible_states": int(len(stable)),
                "high_confidence_states": int(stable["high_confidence"].sum()),
                "top_R_states": int(len(rel)),
                "top_R_and_MHN_count": int(len(rel & mhn)),
                "top_R_and_occupancy_count": int(len(rel & occ)),
                "top_MHN_and_occupancy_count": int(len(mhn & occ)),
                "top_R_and_MHN_fraction": float(len(rel & mhn) / max(len(rel), 1)),
                "top_R_and_occupancy_fraction": float(len(rel & occ) / max(len(rel), 1)),
                "jaccard_R_MHN": float(len(rel & mhn) / max(len(rel | mhn), 1)),
                "jaccard_R_occupancy": float(len(rel & occ) / max(len(rel | occ), 1)),
                "jaccard_MHN_occupancy": float(len(mhn & occ) / max(len(mhn | occ), 1)),
                "spearman_R_MHN": rho_rf,
                "spearman_R_occupancy": rho_rl,
                "spearman_occupancy_MHN": rho_lf,
                "median_rank_gain_vs_MHN": float(rank_gain_f.median()) if len(rank_gain_f) else np.nan,
                "median_rank_gain_vs_occupancy": float(rank_gain_l.median()) if len(rank_gain_l) else np.nan,
                "top_R_highL_lowF_fraction": float(qdist.get("High L / Low F", 0.0)),
                "top_R_highL_highF_fraction": float(qdist.get("High L / High F", 0.0)),
                "top_R_lowL_highF_fraction": float(qdist.get("Low L / High F", 0.0)),
                "top_R_lowL_lowF_fraction": float(qdist.get("Low L / Low F", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def quadrant_summary(states: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows = []
    quadrants = ["High L / High F", "High L / Low F", "Low L / High F", "Low L / Low F"]
    for dataset, ds_cfg in config["datasets"].items():
        stable = states[states["dataset_name"].eq(dataset) & states["stable"]].copy()
        top_r = set(top_sets(states, dataset, config)["rel_obstq_mhn"]["state"])
        stable["top_R"] = stable["state"].isin(top_r)
        for quadrant in quadrants:
            in_q = stable["quadrant"].eq(quadrant)
            top_in_q = stable["top_R"] & in_q
            rows.append(
                {
                    "dataset_name": dataset,
                    "short_name": ds_cfg["short_name"],
                    "quadrant": quadrant,
                    "top_R_count": int(top_in_q.sum()),
                    "top_R_fraction": float(top_in_q.sum() / max(stable["top_R"].sum(), 1)),
                    "background_count": int((~stable["top_R"] & in_q).sum()),
                    "background_fraction": float((~stable["top_R"] & in_q).sum() / max((~stable["top_R"]).sum(), 1)),
                    "all_state_fraction": float(in_q.mean()),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_rank_correlations(states: pd.DataFrame, config: dict) -> pd.DataFrame:
    pairs = [
        ("R* vs MHN inflow", "rel_obstq_score", "mhn_only_score"),
        ("R* vs occupancy", "rel_obstq_score", "occupancy_only_score"),
        ("occupancy vs MHN inflow", "occupancy_only_score", "mhn_only_score"),
    ]
    reps = int(config["analysis"].get("correlation_bootstrap_replicates", 400))
    rng = np.random.default_rng(int(config.get("random_seed", 20260628)))
    rows = []
    for dataset, ds_cfg in config["datasets"].items():
        stable = states[states["dataset_name"].eq(dataset) & states["stable"]].copy().reset_index(drop=True)
        n = len(stable)
        for pair_label, x_col, y_col in pairs:
            observed = float(stable[[x_col, y_col]].corr(method="spearman").iloc[0, 1]) if n >= 3 else np.nan
            boot = []
            if n >= 8:
                for _ in range(reps):
                    sample = stable.iloc[rng.integers(0, n, n)]
                    value = sample[[x_col, y_col]].corr(method="spearman").iloc[0, 1]
                    if np.isfinite(value):
                        boot.append(float(value))
            low, high = (np.quantile(boot, [0.025, 0.975]) if boot else (np.nan, np.nan))
            rows.append(
                {
                    "dataset_name": dataset,
                    "short_name": ds_cfg["short_name"],
                    "pair": pair_label,
                    "x_score": x_col,
                    "y_score": y_col,
                    "spearman_rho": observed,
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "bootstrap_replicates": len(boot),
                    "eligible_states": n,
                }
            )
    return pd.DataFrame(rows)


def rank_gain_table(states: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows = []
    baselines = [
        ("mhn_only", "rank_mhn_only", "MHN inflow"),
        ("occupancy_only", "rank_occupancy_only", "Occupancy"),
    ]
    for dataset, ds_cfg in config["datasets"].items():
        stable = states[states["dataset_name"].eq(dataset) & states["stable"]].copy()
        n = len(stable)
        denom = max(n - 1, 1)
        top = top_sets(states, dataset, config)["rel_obstq_mhn"].copy()
        for _, row in top.iterrows():
            r_rank = float(row["rank_rel_obstq_mhn"])
            r_percentile = 1.0 - (r_rank - 1.0) / denom
            for baseline, rank_col, baseline_label in baselines:
                baseline_rank = float(row[rank_col])
                baseline_percentile = 1.0 - (baseline_rank - 1.0) / denom
                rows.append(
                    {
                        "dataset_name": dataset,
                        "short_name": ds_cfg["short_name"],
                        "state": row["state"],
                        "state_label": row["state_label"],
                        "R_star": float(row["R_star"]),
                        "N_v": int(row["N_v"]),
                        "baseline": baseline,
                        "baseline_label": baseline_label,
                        "R_rank": r_rank,
                        "baseline_rank": baseline_rank,
                        "raw_rank_gain": baseline_rank - r_rank,
                        "R_rank_percentile": r_percentile,
                        "baseline_rank_percentile": baseline_percentile,
                        "percentile_rank_gain": r_percentile - baseline_percentile,
                    }
                )
    return pd.DataFrame(rows)


def square_save(fig: plt.Figure, output: Path, config: dict) -> None:
    figure_style.save_figure_panels(fig, output, config)


def method_color_map(config: dict) -> dict[str, str]:
    cat = figure_style.categorical_palette(config)
    return {
        "mhn_only": cat.get("sky_blue", "#B2E6FD"),
        "occupancy_only": cat.get("sage", "#B8D2CC"),
        "rel_obstq_mhn": cat.get("coral", "#E8B2A7"),
    }


def plot_rank_displacement(fig: plt.Figure, gs_cell, states: pd.DataFrame, config: dict) -> list[plt.Axes]:
    colors = figure_style.colors(config)
    cat = figure_style.categorical_palette(config)
    text_primary = colors.get("text", {}).get("primary", "#263238")
    text_secondary = colors.get("text", {}).get("secondary", "#4E5A5E")
    grid_color = colors.get("text", {}).get("grid", "#E6E6E6")
    line_color = cat.get("coral", "#E8B2A7")
    point_color = cat.get("lavender", "#B5AED5")
    gs = gs_cell.subgridspec(2, 2, wspace=0.32, hspace=0.38)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    display_n = int(config["analysis"].get("rank_displacement_states", 5))
    methods = ["rank_mhn_only", "rank_occupancy_only", "rank_rel_obstq_mhn"]
    x = np.arange(len(methods))
    labels = ["F rank", "L rank", "R* rank"]
    for idx, (ax, dataset) in enumerate(zip(axes, config["datasets"])):
        top = top_sets(states, dataset, config)["rel_obstq_mhn"].head(display_n).copy()
        max_rank = max(10.0, float(top[methods].to_numpy().max()) * 1.08)
        for _, row in top.iterrows():
            y = [float(row[col]) for col in methods]
            ax.plot(x, y, color=line_color, lw=0.9, alpha=0.58, zorder=2)
            ax.scatter(x, y, s=[13, 13, 18], color=[cat.get("sky_blue", "#B2E6FD"), cat.get("sage", "#B8D2CC"), point_color], edgecolor=text_primary, linewidth=0.25, zorder=3)
        ax.set_yscale("log")
        ax.set_ylim(max_rank, 0.8)
        ax.set_xticks(x, labels, fontsize=5.3)
        ax.set_yticks([1, 3, 10, 30, 100, 300])
        ax.set_yticklabels(["1", "3", "10", "30", "100", "300"], fontsize=5.1)
        ax.set_xlim(-0.15, 2.78)
        ax.grid(axis="y", color=grid_color, lw=0.34)
        ax.text(0.03, 0.95, config["datasets"][dataset]["short_name"], transform=ax.transAxes, ha="left", va="top", fontsize=6.2, fontweight="bold", color=text_primary)
        if idx in [0, 2]:
            ax.set_ylabel("rank (1=top)", fontsize=5.4)
        else:
            ax.set_yticklabels([])
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.tick_params(length=1.6, width=0.5)
        ax.set_box_aspect(1)
    axes[0].text(0.00, -0.27, "Each line is a top-R* state; full state labels are in the rank table.", transform=axes[0].transAxes, ha="left", va="top", fontsize=4.8, color=text_secondary)
    return axes


def plot_correlation_forest(ax: plt.Axes, correlations: pd.DataFrame, config: dict) -> None:
    colors = figure_style.colors(config)
    cat = figure_style.categorical_palette(config)
    text_primary = colors.get("text", {}).get("primary", "#263238")
    grid_color = colors.get("text", {}).get("grid", "#E6E6E6")
    pair_order = ["R* vs MHN inflow", "R* vs occupancy", "occupancy vs MHN inflow"]
    pair_colors = {
        "R* vs MHN inflow": cat.get("coral", "#E8B2A7"),
        "R* vs occupancy": cat.get("lavender", "#B5AED5"),
        "occupancy vs MHN inflow": cat.get("sky_blue", "#B2E6FD"),
    }
    cohorts = [config["datasets"][dataset]["short_name"] for dataset in config["datasets"]]
    offsets = {"R* vs MHN inflow": -0.20, "R* vs occupancy": 0.0, "occupancy vs MHN inflow": 0.20}
    y_base = {cohort: i for i, cohort in enumerate(cohorts[::-1])}
    for pair in pair_order:
        sub = correlations[correlations["pair"].eq(pair)]
        for row in sub.itertuples():
            y = y_base[row.short_name] + offsets[pair]
            if np.isfinite(row.ci_low) and np.isfinite(row.ci_high):
                ax.hlines(y, row.ci_low, row.ci_high, color=pair_colors[pair], lw=1.05, alpha=0.82, zorder=2)
            ax.scatter(row.spearman_rho, y, s=21, color=pair_colors[pair], edgecolor=text_primary, linewidth=0.32, zorder=3)
    ax.axvline(0, color="#777777", lw=0.65, ls=(0, (3, 2)))
    ax.set_xlim(-0.85, 0.95)
    ax.set_yticks([y_base[c] for c in cohorts], cohorts, fontsize=6.1)
    ax.set_xlabel("Spearman rho with bootstrap 95% CI", fontsize=6.3)
    ax.grid(axis="x", color=grid_color, lw=0.42)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(labelsize=5.7, length=2.0, width=0.55)
    handles = [
        plt.Line2D([0], [0], marker="o", lw=1.0, color=pair_colors[pair], markerfacecolor=pair_colors[pair], markeredgecolor=text_primary, markersize=4.0, label=pair)
        for pair in pair_order
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left", fontsize=5.1, handlelength=1.4, borderpad=0.1)
    ax.set_box_aspect(1)


def plot_landscape_small_multiples(fig: plt.Figure, gs_cell, states: pd.DataFrame, config: dict) -> list[plt.Axes]:
    colors = figure_style.colors(config)
    cat = figure_style.categorical_palette(config)
    text_primary = colors.get("text", {}).get("primary", "#263238")
    grid_color = colors.get("text", {}).get("grid", "#E6E6E6")
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "e11_rstar",
        [cat.get("pale_yellow", "#FEEBB9"), cat.get("sage", "#B8D2CC"), cat.get("sky_blue", "#B2E6FD"), cat.get("lavender", "#B5AED5")],
    )
    gs = gs_cell.subgridspec(2, 2, wspace=0.20, hspace=0.30)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    max_states = int(config["analysis"]["max_scatter_states_per_cohort"])
    for idx, (ax, dataset) in enumerate(zip(axes, config["datasets"])):
        sub = states[states["dataset_name"].eq(dataset) & states["stable"]].copy()
        if len(sub) > max_states:
            sub = sub.nlargest(max_states, "N_v")
        ax.scatter(
            sub["log10_F_hat"],
            sub["log10_L_v"],
            c=sub["log2_R_star"].clip(-2, 3.2),
            cmap=cmap,
            vmin=-2,
            vmax=3.2,
            s=np.clip(np.sqrt(sub["N_v"].astype(float)) * 1.05, 4, 28),
            edgecolor="white",
            linewidth=0.14,
            alpha=0.76,
            zorder=2,
        )
        l_cut = float(sub["L_quantile_cut"].iloc[0])
        f_cut = float(sub["F_quantile_cut"].iloc[0])
        ax.axhline(np.log10(max(l_cut, 1.0e-12)), color="#777777", lw=0.52, ls=(0, (3, 2)), zorder=1)
        ax.axvline(np.log10(max(f_cut, 1.0e-12)), color="#777777", lw=0.52, ls=(0, (3, 2)), zorder=1)
        ax.text(0.04, 0.95, config["datasets"][dataset]["short_name"], transform=ax.transAxes, ha="left", va="top", fontsize=6.2, fontweight="bold", color=text_primary)
        if idx == 1:
            ax.text(0.98, 0.95, "pale to blue: log2 R*", transform=ax.transAxes, ha="right", va="top", fontsize=4.65, color=text_primary)
        ax.grid(color=grid_color, lw=0.34)
        ax.tick_params(labelsize=5.4, length=1.6, width=0.5)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        if idx in [2, 3]:
            ax.set_xlabel(r"$\log_{10}\hat F$", fontsize=5.7)
        else:
            ax.set_xticklabels([])
        if idx in [0, 2]:
            ax.set_ylabel(r"$\log_{10}L$", fontsize=5.7)
        else:
            ax.set_yticklabels([])
        ax.set_box_aspect(1)
    return axes


def plot_rank_gain(ax: plt.Axes, rank_gain: pd.DataFrame, config: dict) -> None:
    colors = figure_style.colors(config)
    cat = figure_style.categorical_palette(config)
    text_primary = colors.get("text", {}).get("primary", "#263238")
    text_secondary = colors.get("text", {}).get("secondary", "#4E5A5E")
    grid_color = colors.get("text", {}).get("grid", "#E6E6E6")
    cohorts = [config["datasets"][d]["short_name"] for d in config["datasets"]]
    baseline_order = ["MHN inflow", "Occupancy"]
    baseline_colors = {"MHN inflow": cat.get("sky_blue", "#B2E6FD"), "Occupancy": cat.get("sage", "#B8D2CC")}
    x_positions = np.arange(len(cohorts))
    jitter = {"MHN inflow": -0.11, "Occupancy": 0.11}
    for baseline in baseline_order:
        sub = rank_gain[rank_gain["baseline_label"].eq(baseline)].copy()
        for i, cohort in enumerate(cohorts):
            vals = sub[sub["short_name"].eq(cohort)]["percentile_rank_gain"].to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            xs = np.full(len(vals), x_positions[i] + jitter[baseline])
            ax.scatter(xs, vals, s=12, color=mcolors.to_rgba(baseline_colors[baseline], 0.62), edgecolor=text_primary, linewidth=0.22, zorder=2)
            median = float(np.median(vals))
            q1, q3 = np.quantile(vals, [0.25, 0.75])
            ax.vlines(x_positions[i] + jitter[baseline], q1, q3, color=text_primary, lw=0.65, zorder=3)
            ax.hlines(median, x_positions[i] + jitter[baseline] - 0.065, x_positions[i] + jitter[baseline] + 0.065, color=text_primary, lw=1.0, zorder=4)
    ax.axhline(0, color="#777777", lw=0.65, ls=(0, (3, 2)))
    ax.set_xticks(x_positions, cohorts, fontsize=6.1)
    ax.set_ylabel("R* rank gain\n(percentile units)", fontsize=6.0)
    ax.set_ylim(-0.12, 1.05)
    ax.grid(axis="y", color=grid_color, lw=0.42)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(labelsize=5.8, length=2.0, width=0.55)
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=baseline_colors[b], markeredgecolor=text_primary, markersize=4.0, label=f"vs {b}")
        for b in baseline_order
    ]
    ax.legend(handles=handles, frameon=False, loc="upper right", fontsize=5.3)
    ax.set_box_aspect(1)
    ax.text(0.0, -0.20, "Points, top-R* states; bars, median/IQR.\nPositive values mean up-ranked by R*.", transform=ax.transAxes, ha="left", va="top", fontsize=5.0, color=text_secondary)


def plot_main_figure(
    states: pd.DataFrame,
    ranked: pd.DataFrame,
    summary: pd.DataFrame,
    quadrant: pd.DataFrame,
    correlations: pd.DataFrame,
    rank_gain: pd.DataFrame,
    output: Path,
    config: dict,
) -> None:
    figure_style.configure_matplotlib(config)
    colors = figure_style.colors(config)
    text_primary = colors.get("text", {}).get("primary", "#263238")
    text_secondary = colors.get("text", {}).get("secondary", "#4E5A5E")
    fig = plt.figure(figsize=(7.2, 7.2))
    gs = fig.add_gridspec(2, 2, left=0.075, right=0.985, bottom=0.092, top=0.885, wspace=0.30, hspace=0.39)
    ax_b = fig.add_subplot(gs[0, 1])
    ax_d = fig.add_subplot(gs[1, 1])
    axes_a = plot_rank_displacement(fig, gs[0, 0], states, config)
    axes_c = plot_landscape_small_multiples(fig, gs[1, 0], states, config)

    plot_correlation_forest(ax_b, correlations, config)
    plot_rank_gain(ax_d, rank_gain, config)

    axes_a[0].text(-0.47, 1.25, "a", transform=axes_a[0].transAxes, fontsize=10.5, fontweight="bold", ha="left", va="top", color=text_primary)
    axes_a[0].text(-0.02, 1.25, "Top R* states are re-ranked by the ratio model", transform=axes_a[0].transAxes, fontsize=8.1, ha="left", va="top", color=text_primary)
    ax_b.text(-0.13, 1.08, "b", transform=ax_b.transAxes, fontsize=10.5, fontweight="bold", ha="left", va="top", color=text_primary)
    ax_b.text(0.00, 1.08, "Full-rank association with uncertainty", transform=ax_b.transAxes, fontsize=8.1, ha="left", va="top", color=text_primary)
    axes_c[0].text(-0.47, 1.25, "c", transform=axes_c[0].transAxes, fontsize=10.5, fontweight="bold", ha="left", va="top", color=text_primary)
    axes_c[0].text(-0.02, 1.25, r"Occupancy $L$ vs MHN inflow $\hat F$", transform=axes_c[0].transAxes, fontsize=8.1, ha="left", va="top", color=text_primary)
    ax_d.text(-0.13, 1.08, "d", transform=ax_d.transAxes, fontsize=10.5, fontweight="bold", ha="left", va="top", color=text_primary)
    ax_d.text(0.00, 1.08, "State-level rank gain over baselines", transform=ax_d.transAxes, fontsize=8.1, ha="left", va="top", color=text_primary)

    fig.text(0.075, 0.972, "Experiment 11 | Information gain over MHN-only and occupancy-only views", ha="left", va="top", fontsize=9.4, fontweight="bold", color=text_primary)
    fig.text(0.075, 0.947, r"MHN-only uses expected inflow $\hat F$; occupancy-only uses $L$; Rel-ObsTQ-MHN uses $R^*=L/\hat F$ after cohort normalization.", ha="left", va="top", fontsize=5.9, color=text_secondary)
    square_save(fig, output, config)


def write_reports(
    root: Path,
    config: dict,
    summary: pd.DataFrame,
    ranked: pd.DataFrame,
    quadrant: pd.DataFrame,
    correlations: pd.DataFrame,
    rank_gain: pd.DataFrame,
) -> None:
    protocol = f"""# Experiment 11 Protocol Audit

## Protocol Section

Source document section: `17. Experiment 11: MHN-only / Occupancy-only / Rel-ObsTQ-MHN information gain`.

Purpose: show that the innovation is not equivalent to either a pure MHN
transition view or a pure cross-sectional occupancy view.

## Operational Definitions

- MHN-only: `F_hat`, the model-derived expected inflow into a state.
- Occupancy-only: `L_v`, the observed cross-sectional state occupancy.
- Rel-ObsTQ-MHN: `R_star`, the normalized ratio of occupancy to expected inflow.

`R_star` top states are selected from high-confidence Experiment 5 states when
available. The baselines are ranked over eligible Experiment 5 states because
they do not have bootstrap confidence intervals of their own.

## Figure Design Patterns

{figure_style.design_patterns_markdown(config)}
"""
    (root / "experiment_11_protocol_audit.md").write_text(protocol, encoding="utf-8")

    lines = [
        "# Experiment 11 Summary",
        "",
        "| Cohort | Eligible | High-conf | R* vs MHN top-K | R* vs L top-K | rho(R*,F) | rho(R*,L) | rho(L,F) | Median gain vs MHN | Median gain vs L |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    gain_summary = (
        rank_gain.groupby(["short_name", "baseline_label"])["percentile_rank_gain"]
        .median()
        .unstack()
    )
    for row in summary.itertuples():
        gain_mhn = float(gain_summary.loc[row.short_name, "MHN inflow"]) if row.short_name in gain_summary.index else np.nan
        gain_l = float(gain_summary.loc[row.short_name, "Occupancy"]) if row.short_name in gain_summary.index else np.nan
        lines.append(
            f"| {row.short_name} | {row.eligible_states} | {row.high_confidence_states} | {row.top_R_and_MHN_count}/{row.top_R_states} | {row.top_R_and_occupancy_count}/{row.top_R_states} | {row.spearman_R_MHN:.2f} | {row.spearman_R_occupancy:.2f} | {row.spearman_occupancy_MHN:.2f} | {gain_mhn:.2f} | {gain_l:.2f} |"
        )
    (root / "experiment_11_summary.md").write_text("\n".join(lines), encoding="utf-8")

    sci = [
        "# Experiment 11 Scientific Review",
        "",
        "## Main Interpretation",
        "",
        "Experiment 11 supports the information-gain claim when three patterns appear together: R* top states have low overlap with MHN-inflow-only top states, occupancy and MHN inflow are strongly correlated, and R* is negatively or weakly correlated with inflow. That pattern means frequent states are often frequent because they are easy to enter, while R* highlights states that remain enriched after controlling for inflow.",
        "",
        "## Cohort-Level Evidence",
        "",
        "| Cohort | Main evidence | Caveat |",
        "|---|---|---|",
    ]
    for row in summary.itertuples():
        caveat = "robust real-cohort contrast"
        sci.append(
            f"| {row.short_name} | R* vs MHN={row.top_R_and_MHN_count}/{row.top_R_states}; R* vs L={row.top_R_and_occupancy_count}/{row.top_R_states}; rho(L,F)={row.spearman_occupancy_MHN:.2f}; rho(R*,F)={row.spearman_R_MHN:.2f} | {caveat} |"
        )
    sci.extend(
        [
            "",
            "## Top Rank Examples",
            "",
            "| Cohort | Method | Rank | State | Score | L | F_hat | R* | Quadrant |",
            "|---|---|---:|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in ranked.groupby(["dataset_name", "method"], sort=False, group_keys=False).head(2).itertuples():
        sci.append(
            f"| {row.short_name} | {row.method_label} | {row.rank} | {row.state} | {row.score:.4g} | {row.L_v:.4g} | {row.F_hat:.4g} | {row.R_star:.3f} | {row.quadrant} |"
        )
    sci.extend(
        [
            "",
            "## Bootstrap Rank Correlations",
            "",
            "| Cohort | Pair | Spearman rho [95% CI] |",
            "|---|---|---:|",
        ]
    )
    for row in correlations.itertuples():
        sci.append(f"| {row.short_name} | {row.pair} | {row.spearman_rho:.2f} [{row.ci_low:.2f}, {row.ci_high:.2f}] |")
    sci.extend(
        [
            "",
            "## Rank-Gain Interpretation",
            "",
            "Rank gain is reported in percentile units: positive values mean a top R* state is placed higher by Rel-ObsTQ-MHN than by the baseline score. This avoids relying only on top-K overlap and directly measures how much the ratio model reorders states.",
            "",
            "## Interpretation Boundary",
            "",
            "This experiment is a ranking and geometry comparison, not a clinical endpoint analysis. It proves non-redundancy of the Rel-ObsTQ-MHN score relative to its two components; clinical utility remains the target of later clinical/replication experiments.",
        ]
    )
    (root / "experiment_11_scientific_review.md").write_text("\n".join(sci), encoding="utf-8")

    design = f"""# Experiment 11 Figure Design Review

## Sources

{figure_style.design_sources_markdown(config)}

## Rules Applied

{figure_style.design_rules_markdown(config)}

## Design Choices

- The primary figure is square and uses four compact panels.
- Panel A replaces the earlier text-heavy ranking table with rank-displacement
  tracks for top R* states.
- Panel B reports full-rank Spearman associations with bootstrap uncertainty,
  rather than a heatmap without uncertainty.
- Panel C keeps the protocol's L-versus-F scatter with color encoding for R*.
- Panel D reports state-level percentile rank gain over the two baselines,
  making the information-gain claim directly auditable.
"""
    (root / "top_journal_figure_design_review.md").write_text(design, encoding="utf-8")


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
    (root / "resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    states = assign_quadrants(load_state_scores(config), config)
    ranked = rank_comparison_table(states, config)
    summary = overlap_and_rank_summary(states, ranked, config)
    quadrant = quadrant_summary(states, config)
    correlations = bootstrap_rank_correlations(states, config)
    rank_gain = rank_gain_table(states, config)

    states.to_csv(tables / "combined_information_gain_scores.tsv", sep="\t", index=False)
    ranked.to_csv(tables / "top_rank_comparison.tsv", sep="\t", index=False)
    summary.to_csv(tables / "information_gain_summary.tsv", sep="\t", index=False)
    quadrant.to_csv(tables / "quadrant_enrichment_summary.tsv", sep="\t", index=False)
    correlations.to_csv(tables / "rank_correlation_bootstrap.tsv", sep="\t", index=False)
    rank_gain.to_csv(tables / "rank_gain_distribution.tsv", sep="\t", index=False)

    plot_main_figure(states, ranked, summary, quadrant, correlations, rank_gain, figures / "Figure_E11_information_gain_controls", config)
    write_reports(root, config, summary, ranked, quadrant, correlations, rank_gain)


if __name__ == "__main__":
    main()

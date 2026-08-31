"""Run Experiment 15: innovation-specific falsification controls.

This experiment is deliberately narrow. It asks whether high R* states remain
exceptional against structurally matched real-state decoys, and whether the same
top states survive when the learned L-F pairing is falsified within comparable
state strata.
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
from matplotlib.lines import Line2D

import figure_style


CONFIG_PATH = Path("configs/experiment_15.yaml")
OBSOLETE_TABLES = [
    "bootstrap_uncertainty_summary.tsv",
    "bootstrap_top_state_details.tsv",
    "negative_control_permutation_replicates.tsv",
    "negative_control_permutation_summary.tsv",
    "null_backbone_simulation_repeats.tsv",
    "null_backbone_simulation_summary.tsv",
    "rare_state_threshold_sensitivity.tsv",
]
OBSOLETE_FIGURES = [
    "Figure_E15_uncertainty_negative_controls.pdf",
    "Figure_E15_uncertainty_negative_controls.png",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Rel-ObsTQ-MHN Experiment 15.")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--result-root")
    return parser.parse_args()


def finite_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].replace([np.inf, -np.inf], np.nan)


def read_state_tables(dataset: str, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(config["experiment_05_root"]) / dataset / "tables"
    scores = pd.read_csv(root / "state_scores.tsv", sep="\t")
    top = pd.read_csv(root / "top_bottleneck_states_high_confidence.tsv", sep="\t")
    return scores, top


def add_match_bins(scores: pd.DataFrame, config: dict) -> pd.DataFrame:
    work = scores.copy()
    bins = int(config["analysis"].get("match_quantile_bins", 4))
    work["log_N_v"] = np.log1p(work["N_v"].astype(float))
    work["log_F_hat"] = np.log10(work["F_hat"].astype(float) + float(config["analysis"].get("epsilon", 1.0e-6)))
    for column, out in [("log_N_v", "N_bin"), ("log_F_hat", "F_bin")]:
        values = finite_series(work, column)
        if values.nunique(dropna=True) <= 1:
            work[out] = 0
        else:
            work[out] = pd.qcut(values.rank(method="first"), q=min(bins, values.nunique()), labels=False, duplicates="drop")
    return work


def eligible_scores(scores: pd.DataFrame, config: dict) -> pd.DataFrame:
    min_count = int(config["analysis"].get("minimum_state_count", 5))
    min_inflow = float(config["analysis"].get("minimum_inflow", 1.0e-6))
    work = scores.copy()
    keep = (
        work["eligible_experiment5"].astype(bool)
        & work["N_v"].ge(min_count)
        & work["F_hat"].ge(min_inflow)
        & finite_series(work, "R_star").notna()
        & finite_series(work, "L_v").notna()
        & finite_series(work, "F_hat").notna()
    )
    return add_match_bins(work.loc[keep].copy(), config)


def matched_decoy_contrast_for_dataset(dataset: str, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores, top = read_state_tables(dataset, config)
    eligible = eligible_scores(scores, config)
    short = config["datasets"][dataset]["short_name"]
    top_k = int(config["analysis"].get("top_k", 10))
    min_decoys = int(config["analysis"].get("matched_decoy_minimum_decoys", 5))
    selected = top.head(top_k).copy()
    selected_states = set(selected["state"].astype(str))
    eligible_by_state = eligible.set_index("state", drop=False)
    rows = []
    for rank, row in enumerate(selected.itertuples(), start=1):
        if row.state not in eligible_by_state.index:
            continue
        target = eligible_by_state.loc[row.state]
        base = eligible[~eligible["state"].astype(str).isin(selected_states)].copy()
        tiers = [
            ("stage+events+Nbin+Fbin", base[
                base["stage"].eq(target["stage"])
                & base["event_count"].eq(target["event_count"])
                & base["N_bin"].eq(target["N_bin"])
                & base["F_bin"].eq(target["F_bin"])
            ]),
            ("stage+events+Fbin", base[
                base["stage"].eq(target["stage"])
                & base["event_count"].eq(target["event_count"])
                & base["F_bin"].eq(target["F_bin"])
            ]),
            ("stage+events", base[
                base["stage"].eq(target["stage"])
                & base["event_count"].eq(target["event_count"])
            ]),
            ("stage+near_events", base[
                base["stage"].eq(target["stage"])
                & (base["event_count"] - int(target["event_count"])).abs().le(1)
            ]),
            ("all_eligible", base),
        ]
        match_tier = "all_eligible"
        decoys = base
        for tier, candidate in tiers:
            if len(candidate) >= min_decoys or tier == "all_eligible":
                match_tier = tier
                decoys = candidate
                break
        decoy_r = finite_series(decoys, "R_star").dropna()
        decoy_log = np.log2(decoy_r.clip(lower=1.0e-12))
        target_r = float(target["R_star"])
        target_log = float(np.log2(max(target_r, 1.0e-12)))
        rows.append(
            {
                "dataset_name": dataset,
                "short_name": short,
                "rank": int(rank),
                "state": str(row.state),
                "stage": str(target["stage"]),
                "event_count": int(target["event_count"]),
                "N_v": float(target["N_v"]),
                "F_hat": float(target["F_hat"]),
                "R_star": target_r,
                "decoy_count": int(len(decoy_r)),
                "match_tier": match_tier,
                "decoy_median_R_star": float(decoy_r.median()) if len(decoy_r) else np.nan,
                "decoy_q90_R_star": float(decoy_r.quantile(0.90)) if len(decoy_r) else np.nan,
                "matched_percentile": float((decoy_r <= target_r).mean()) if len(decoy_r) else np.nan,
                "log2_R_advantage": target_log - float(decoy_log.median()) if len(decoy_log) else np.nan,
            }
        )
    details = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "dataset_name": dataset,
                "short_name": short,
                "top_states_tested": int(len(details)),
                "median_matched_percentile": float(details["matched_percentile"].median()) if len(details) else np.nan,
                "fraction_above_decoy_q90": float(details["matched_percentile"].ge(0.90).mean()) if len(details) else np.nan,
                "median_log2_R_advantage": float(details["log2_R_advantage"].median()) if len(details) else np.nan,
                "median_decoy_count": float(details["decoy_count"].median()) if len(details) else np.nan,
            }
        ]
    )
    return details, summary


def shuffle_within_strata(values: pd.Series, strata: pd.Series, rng: np.random.Generator) -> np.ndarray:
    shuffled = values.to_numpy(dtype=float).copy()
    for _, index in strata.groupby(strata).groups.items():
        positions = np.asarray(index, dtype=int)
        if len(positions) > 1:
            shuffled[positions] = rng.permutation(shuffled[positions])
    return shuffled


def inflow_pairing_falsification_for_dataset(dataset: str, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores, top = read_state_tables(dataset, config)
    eligible = eligible_scores(scores, config).reset_index(drop=True)
    short = config["datasets"][dataset]["short_name"]
    top_k = min(int(config["analysis"].get("top_k", 10)), len(top))
    repeats = int(config["analysis"].get("inflow_shuffle_replicates", 400))
    epsilon = float(config["analysis"].get("epsilon", 1.0e-6))
    top_states = set(top.head(top_k)["state"].astype(str))
    strata = eligible["stage"].astype(str) + "|e" + eligible["event_count"].astype(str)
    rng = np.random.default_rng(int(config.get("random_seed", 20260628)) + sum(ord(char) for char in dataset) + 150000)
    rows = []
    for repeat in range(1, repeats + 1):
        shuffled_f = shuffle_within_strata(eligible["F_hat"].astype(float), strata, rng)
        raw = eligible["L_v"].astype(float).to_numpy() / (shuffled_f + epsilon)
        normalizer = np.nanmedian(raw)
        r_shuffle = raw / normalizer if np.isfinite(normalizer) and normalizer > 0 else np.full(len(raw), np.nan)
        work = eligible[["state"]].copy()
        work["R_shuffle"] = r_shuffle
        shuffled_top = set(work.sort_values("R_shuffle", ascending=False, na_position="last").head(top_k)["state"].astype(str))
        overlap = len(top_states.intersection(shuffled_top))
        rows.append(
            {
                "dataset_name": dataset,
                "short_name": short,
                "repeat": int(repeat),
                "top_k": int(top_k),
                "top_overlap": int(overlap),
                "top_overlap_fraction": float(overlap / max(top_k, 1)),
                "median_top_R_shuffle": float(work.sort_values("R_shuffle", ascending=False, na_position="last").head(top_k)["R_shuffle"].median()),
            }
        )
    replicates = pd.DataFrame(rows)
    summary = (
        replicates.groupby(["dataset_name", "short_name"], as_index=False)
        .agg(
            repeats=("repeat", "count"),
            top_k=("top_k", "first"),
            median_shuffled_overlap=("top_overlap_fraction", "median"),
            q1_shuffled_overlap=("top_overlap_fraction", lambda s: s.quantile(0.25)),
            q3_shuffled_overlap=("top_overlap_fraction", lambda s: s.quantile(0.75)),
            exact_recovery_fraction=("top_overlap_fraction", lambda s: float((s >= 1.0).mean())),
            median_top_R_shuffle=("median_top_R_shuffle", "median"),
        )
    )
    summary["observed_overlap_fraction"] = 1.0
    summary["median_overlap_loss"] = summary["observed_overlap_fraction"] - summary["median_shuffled_overlap"]
    return replicates, summary


def run_analysis(config: dict) -> dict[str, pd.DataFrame]:
    matched_details = []
    matched_summary = []
    inflow_reps = []
    inflow_summary = []
    for dataset in config["datasets"]:
        details, summary = matched_decoy_contrast_for_dataset(dataset, config)
        matched_details.append(details)
        matched_summary.append(summary)
        reps, isum = inflow_pairing_falsification_for_dataset(dataset, config)
        inflow_reps.append(reps)
        inflow_summary.append(isum)
    return {
        "matched_details": pd.concat(matched_details, ignore_index=True),
        "matched_summary": pd.concat(matched_summary, ignore_index=True),
        "inflow_replicates": pd.concat(inflow_reps, ignore_index=True),
        "inflow_summary": pd.concat(inflow_summary, ignore_index=True),
    }


def save_square(fig: plt.Figure, output: Path, config: dict) -> None:
    figure_style.save_figure_panels(fig, output, config)


def plot_main_figure(tables: dict[str, pd.DataFrame], output: Path, config: dict) -> None:
    figure_style.configure_matplotlib(config)
    colors = figure_style.colors(config)
    cat = figure_style.categorical_palette(config)
    text_primary = colors.get("text", {}).get("primary", "#263238")
    text_secondary = colors.get("text", {}).get("secondary", "#4E5A5E")
    grid_color = colors.get("text", {}).get("grid", "#E6E6E6")
    cohort_colors = [cat.get("lavender", "#B5AED5"), cat.get("sky_blue", "#B2E6FD"), cat.get("sage", "#B8D2CC"), cat.get("coral", "#E8B2A7")]
    short_names = [cfg["short_name"] for cfg in config["datasets"].values()]
    color_by_short = dict(zip(short_names, cohort_colors))
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
        "delta_r", [cat.get("pale_yellow", "#FEEBB9"), cat.get("sage", "#B8D2CC"), cat.get("sky_blue", "#B2E6FD"), cat.get("lavender", "#B5AED5")]
    )

    fig = plt.figure(figsize=(7.2, 7.2))
    fig.text(0.075, 0.972, "Experiment 15 | Innovation-specific falsification controls", ha="left", va="top", fontsize=9.4, fontweight="bold", color=text_primary)
    fig.text(0.075, 0.947, "Two non-redundant tests: matched real-state decoys and within-stratum inflow-pairing falsification.", ha="left", va="top", fontsize=5.8, color=text_secondary)

    ax_a = fig.add_axes([0.075, 0.565, 0.865, 0.300])
    matched = tables["matched_details"].copy()
    y_lookup = {short: len(short_names) - idx for idx, short in enumerate(short_names)}
    for short in short_names:
        y0 = y_lookup[short]
        ax_a.axhline(y0, color=grid_color, lw=0.35, zorder=0)
    for row in matched.itertuples():
        y = y_lookup[row.short_name] + (0.22 - 0.044 * ((int(row.rank) - 1) % 10))
        color_value = np.clip(float(row.log2_R_advantage), -0.5, 4.0)
        size = float(np.clip(10 + 2.3 * np.sqrt(max(float(row.decoy_count), 1.0)), 14, 46))
        ax_a.scatter(row.matched_percentile, y, s=size, color=cmap((color_value + 0.5) / 4.5), edgecolor=text_primary, linewidth=0.28, zorder=3)
    for short in short_names:
        sub = matched[matched["short_name"].eq(short)]
        if len(sub):
            med = float(sub["matched_percentile"].median())
            ax_a.plot([med, med], [y_lookup[short] - 0.30, y_lookup[short] + 0.30], color=color_by_short[short], lw=1.0, zorder=2)
            ax_a.text(1.015, y_lookup[short], f"{short}  med={med:.2f}", ha="left", va="center", fontsize=5.0, color=text_secondary, clip_on=False)
    ax_a.axvspan(0.90, 1.00, color=cat.get("pale_yellow", "#FEEBB9"), alpha=0.30, zorder=0)
    ax_a.set_xlim(0, 1.13)
    ax_a.set_ylim(0.45, len(short_names) + 0.55)
    ax_a.set_yticks([y_lookup[short] for short in short_names], short_names)
    ax_a.set_xlabel("Matched-decoy percentile of each top R* state", fontsize=6.2)
    ax_a.set_ylabel("Cohort", fontsize=6.2)
    ax_a.grid(axis="x", color=grid_color, lw=0.35)
    ax_a.text(0.90, len(short_names) + 0.48, "upper decile of matched decoys", ha="left", va="top", fontsize=4.7, color=text_secondary)
    for spine in ["top", "right"]:
        ax_a.spines[spine].set_visible(False)
    ax_a.tick_params(labelsize=5.7, length=2.0, width=0.55)
    ax_a.text(-0.08, 1.13, "a", transform=ax_a.transAxes, fontsize=10.5, fontweight="bold", ha="left", va="top", color=text_primary)
    ax_a.text(0.00, 1.13, "Top R* states against matched real-state decoys", transform=ax_a.transAxes, fontsize=8.0, ha="left", va="top", color=text_primary)

    ax_b = fig.add_axes([0.075, 0.155, 0.865, 0.305])
    reps = tables["inflow_replicates"].copy()
    summary = tables["inflow_summary"].copy()
    rng = np.random.default_rng(424242)
    for short in short_names:
        y0 = y_lookup[short]
        sub = reps[reps["short_name"].eq(short)]
        if len(sub):
            sample = sub.sample(n=min(140, len(sub)), random_state=7)
            jitter = rng.uniform(-0.19, 0.19, len(sample))
            ax_b.scatter(sample["top_overlap_fraction"], y0 + jitter, s=7, color=color_by_short[short], alpha=0.18, edgecolor="none", zorder=2)
        row = summary[summary["short_name"].eq(short)].iloc[0]
        ax_b.plot([row.q1_shuffled_overlap, row.q3_shuffled_overlap], [y0, y0], color=color_by_short[short], lw=2.0, solid_capstyle="round", zorder=3)
        ax_b.scatter(row.median_shuffled_overlap, y0, s=30, color=color_by_short[short], edgecolor=text_primary, linewidth=0.35, zorder=4)
        ax_b.scatter(1.0, y0, s=34, marker="D", color=cat.get("pale_yellow", "#FEEBB9"), edgecolor=text_primary, linewidth=0.35, zorder=5)
        ax_b.text(1.025, y0, f"loss={row.median_overlap_loss:.2f}", ha="left", va="center", fontsize=5.0, color=text_secondary, clip_on=False)
    ax_b.set_xlim(-0.03, 1.18)
    ax_b.set_ylim(0.45, len(short_names) + 0.55)
    ax_b.set_yticks([y_lookup[short] for short in short_names], short_names)
    ax_b.set_xlabel("Top-K retained after within-stratum F reassignment", fontsize=6.2)
    ax_b.set_ylabel("Cohort", fontsize=6.2)
    ax_b.grid(axis="x", color=grid_color, lw=0.35)
    ax_b.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=cat.get("sage", "#B8D2CC"), markeredgecolor=text_primary, markeredgewidth=0.35, markersize=4.0, label="shuffled F pairing"),
            Line2D([0], [0], marker="D", color="none", markerfacecolor=cat.get("pale_yellow", "#FEEBB9"), markeredgecolor=text_primary, markeredgewidth=0.35, markersize=4.0, label="observed pairing"),
        ],
        loc="lower right",
        frameon=False,
        fontsize=5.0,
        borderpad=0.1,
        handletextpad=0.4,
    )
    for spine in ["top", "right"]:
        ax_b.spines[spine].set_visible(False)
    ax_b.tick_params(labelsize=5.7, length=2.0, width=0.55)
    ax_b.text(-0.08, 1.12, "b", transform=ax_b.transAxes, fontsize=10.5, fontweight="bold", ha="left", va="top", color=text_primary)
    ax_b.text(0.00, 1.12, "State-specific expected inflow pairing falsification", transform=ax_b.transAxes, fontsize=8.0, ha="left", va="top", color=text_primary)

    fig.text(0.075, 0.065, "Panel a controls for stage, event burden, observed support and expected-inflow scale; panel b preserves marginal L and F but breaks their state-specific pairing.", ha="left", va="center", fontsize=5.1, color=text_secondary)
    save_square(fig, output, config)


def write_reports(root: Path, config: dict, tables: dict[str, pd.DataFrame]) -> None:
    matched_summary = tables["matched_summary"]
    inflow_summary = tables["inflow_summary"]
    lines = [
        "# Experiment 15 Summary",
        "",
        "## A. Matched Real-State Decoy Contrast",
        "",
        "| Cohort | Top states | Median matched percentile | Fraction above decoy q90 | Median log2 R* advantage | Median decoys |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in matched_summary.itertuples():
        lines.append(f"| {row.short_name} | {row.top_states_tested} | {row.median_matched_percentile:.3f} | {row.fraction_above_decoy_q90:.2f} | {row.median_log2_R_advantage:.2f} | {row.median_decoy_count:.0f} |")
    lines.extend(
        [
            "",
            "## B. State-Specific F-Pairing Falsification",
            "",
            "| Cohort | Repeats | Median shuffled Top-K retained | IQR | Exact recovery fraction | Median overlap loss |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in inflow_summary.itertuples():
        iqr = row.q3_shuffled_overlap - row.q1_shuffled_overlap
        lines.append(f"| {row.short_name} | {row.repeats} | {row.median_shuffled_overlap:.3f} | {iqr:.3f} | {row.exact_recovery_fraction:.3f} | {row.median_overlap_loss:.3f} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Experiment 15 is now a focused falsification experiment for the relative dwell-time innovation. It does not repeat bootstrap stability, clinical validation, topology recovery, or rare-state threshold QC. The two retained tests ask whether top R* states are exceptional among matched real states, and whether the top-state identity depends on the learned state-specific expected-inflow pairing.",
        ]
    )
    (root / "experiment_15_summary.md").write_text("\n".join(lines), encoding="utf-8")

    protocol = f"""# Experiment 15 Protocol Audit

## Revised Purpose

Experiment 15 is reduced to two innovation-specific controls:

1. Matched real-state decoy contrast: compare each top R* state with eligible
   non-top states matched as closely as possible on stage, event count, observed
   support and expected-inflow scale.
2. State-specific F-pairing falsification: shuffle `F_hat` within stage/event
   strata while keeping observed occupancy `L_v` fixed, then measure whether the
   same top R* states are recovered.

Removed components: bootstrap uncertainty, broad event/stage permutation,
rare-state threshold sensitivity and Experiment 6 null-backbone simulation. Those
checks either duplicated prior experiments or did not directly sharpen the
relative dwell-time innovation.

## Figure Design Patterns

{figure_style.design_patterns_markdown(config)}
"""
    (root / "experiment_15_protocol_audit.md").write_text(protocol, encoding="utf-8")

    review = f"""# Experiment 15 Scientific Review

Experiment 15 now tests a narrower and more publishable claim: high R* states
should not be explainable by coarse structural features of the state table, and
their identity should depend on the learned pairing between observed occupancy
`L_v` and expected inflow `F_hat`.

Panel A is a matched-real-decoy analysis rather than another bootstrap or
replication experiment. A top state scoring in the upper tail of matched decoys
means the relative dwell signal remains unusual even among states with similar
stage, event burden, observed support and inflow scale.

Panel B is a real-cohort falsification of the denominator pairing. It preserves
the marginal distributions of `L_v` and `F_hat` within comparable state strata
but breaks their state-specific alignment. Loss of top-K identity after this
operation supports the key innovation: `R*` depends on the state-specific
observed-versus-expected dwell relationship, not only on the marginal
distributions of occupancy or inflow.

This experiment should be treated as a focused falsification control. It is not
a new biological discovery panel and not a repeat of the earlier simulation,
clinical, topology, or bootstrap analyses.

## Design Sources

{figure_style.design_sources_markdown(config)}

## Design Rules

{figure_style.design_rules_markdown(config)}
"""
    (root / "experiment_15_scientific_review.md").write_text(review, encoding="utf-8")

    design = f"""# Experiment 15 Figure Design Review

The revised figure uses two non-redundant panels. Panel A is a strip-style
matched-decoy percentile display with direct cohort medians. Panel B is a raw
replicate distribution with IQR bars for F-pairing falsification. This avoids
reusing the earlier four-panel bootstrap/permutation/sensitivity layout and
keeps the visual emphasis on the relative dwell-time mechanism.

## Public Figure Style Rules Used

{figure_style.design_rules_markdown(config)}
"""
    (root / "top_journal_figure_design_review.md").write_text(design, encoding="utf-8")


def cleanup_obsolete_tables(tables_dir: Path) -> None:
    for name in OBSOLETE_TABLES:
        path = tables_dir / name
        if path.exists():
            path.unlink()


def cleanup_obsolete_figures(figures_dir: Path) -> None:
    for name in OBSOLETE_FIGURES:
        path = figures_dir / name
        if path.exists():
            path.unlink()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.result_root:
        config["result_root"] = args.result_root
    root = Path(config["result_root"]).resolve()
    tables_dir = root / "tables"
    figures_dir = root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    cleanup_obsolete_tables(tables_dir)
    cleanup_obsolete_figures(figures_dir)
    (root / "resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    tables = run_analysis(config)
    tables["matched_details"].to_csv(tables_dir / "matched_decoy_contrast.tsv", sep="\t", index=False)
    tables["matched_summary"].to_csv(tables_dir / "matched_decoy_summary.tsv", sep="\t", index=False)
    tables["inflow_replicates"].to_csv(tables_dir / "inflow_pairing_falsification_replicates.tsv", sep="\t", index=False)
    tables["inflow_summary"].to_csv(tables_dir / "inflow_pairing_falsification_summary.tsv", sep="\t", index=False)
    plot_main_figure(tables, figures_dir / "Figure_E15_innovation_falsification_controls", config)
    write_reports(root, config, tables)


if __name__ == "__main__":
    main()

"""Render the selected factor-profile figure for Experiment 7.

The script uses the balanced Experiment 7 summary tables and the shared project
figure style. It changes only figure layout, not the experiment data.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

import figure_style


CONFIG_PATH = Path("configs/experiment_07_balanced.yaml")
TOPOLOGIES = ["linear", "branching", "mutual_exclusivity", "mixed"]
TOPOLOGY_LABELS = {
    "linear": "Linear",
    "branching": "Branching",
    "mutual_exclusivity": "Mutual excl.",
    "mixed": "Mixed",
}
TOPOLOGY_SHORT = {
    "linear": "Lin",
    "branching": "Br",
    "mutual_exclusivity": "ME",
    "mixed": "Mix",
}
PLACEMENTS = ["early_stage", "middle_stage", "late_stage", "pathway_specific"]
PLACEMENT_LABELS = {
    "early_stage": "Early",
    "middle_stage": "Middle",
    "late_stage": "Late",
    "pathway_specific": "Pathway",
}


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def prepare_metrics(root: Path) -> pd.DataFrame:
    metrics = pd.read_csv(root / "tables" / "repeat_metrics.tsv", sep="\t")
    metrics["gain"] = metrics["spearman_R_star"] - metrics["spearman_occupancy"]
    return metrics


def factor_summary(metrics: pd.DataFrame, group_col: str, labels: list[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for value, label in labels:
        sub = metrics[metrics[group_col] == value]
        rows.append(
            {
                "label": label,
                "rho_med": sub["spearman_R_star"].median(),
                "rho_p10": sub["spearman_R_star"].quantile(0.10),
                "rho_q1": sub["spearman_R_star"].quantile(0.25),
                "rho_q3": sub["spearman_R_star"].quantile(0.75),
                "rho_p90": sub["spearman_R_star"].quantile(0.90),
                "occ_med": sub["spearman_occupancy"].median(),
                "occ_p10": sub["spearman_occupancy"].quantile(0.10),
                "occ_q1": sub["spearman_occupancy"].quantile(0.25),
                "occ_q3": sub["spearman_occupancy"].quantile(0.75),
                "occ_p90": sub["spearman_occupancy"].quantile(0.90),
                "gain_med": sub["gain"].median(),
            }
        )
    return pd.DataFrame(rows)


def render_design_d(metrics: pd.DataFrame, output: Path, config: dict) -> None:
    metrics = metrics.copy()
    if "gain" not in metrics.columns:
        metrics["gain"] = metrics["spearman_R_star"] - metrics["spearman_occupancy"]
    colors = figure_style.categorical_palette(config)
    rho_color = colors.get("sky_blue", "#B2E6FD")
    occ_color = colors.get("sage", "#B8D2CC")
    gain_color = colors.get("coral", "#E8B2A7")
    text_primary = figure_style.colors(config).get("text", {}).get("primary", "#263238")
    text_secondary = figure_style.colors(config).get("text", {}).get("secondary", "#4E5A5E")
    panels = [
        ("Topology", factor_summary(metrics, "topology", [(t, TOPOLOGY_LABELS[t]) for t in TOPOLOGIES])),
        ("Long-dwell placement", factor_summary(metrics, "bottleneck_placement", [(p, PLACEMENT_LABELS[p]) for p in PLACEMENTS])),
        ("Sparsity", factor_summary(metrics, "sparsity", [(0.05, "5%"), (0.10, "10%"), (0.20, "20%")])),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.25), sharex=True)
    fig.subplots_adjust(left=0.08, right=0.975, top=0.73, bottom=0.17, wspace=0.34)
    fig.suptitle("D. Factor profile of relative dwell-time recovery", x=0.02, y=0.975, ha="left", fontsize=10.8, fontweight="bold")
    fig.text(
        0.02,
        0.895,
        "Median with IQR and 10-90% range across simulation repeats; right labels show median Δρ over occupancy.",
        ha="left",
        va="center",
        fontsize=6.5,
        color=text_secondary,
    )
    legend_ax = fig.add_axes([0.70, 0.835, 0.25, 0.08])
    legend_ax.axis("off")
    legend_ax.plot([0.02, 0.16], [0.68, 0.68], color=rho_color, lw=3.0, solid_capstyle="round")
    legend_ax.scatter([0.09], [0.68], s=26, facecolor=rho_color, edgecolor=text_primary, linewidth=0.45, zorder=3)
    legend_ax.text(0.20, 0.68, r"$R^*$", va="center", fontsize=6.5, color=text_primary)
    legend_ax.plot([0.47, 0.61], [0.68, 0.68], color=occ_color, lw=3.0, solid_capstyle="round")
    legend_ax.scatter([0.54], [0.68], s=25, facecolor="white", edgecolor=occ_color, linewidth=1.1, zorder=3)
    legend_ax.text(0.65, 0.68, "Occupancy", va="center", fontsize=6.5, color=text_primary)
    legend_ax.plot([0.02, 0.16], [0.26, 0.26], color=text_primary, lw=0.7, alpha=0.35)
    legend_ax.text(0.20, 0.26, "median shift", va="center", fontsize=5.7, color=text_secondary)
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    for ax, (title, df) in zip(axes, panels):
        y = np.arange(len(df))
        for idx, row in df.iterrows():
            ax.plot([row["occ_med"], row["rho_med"]], [idx, idx], color=text_primary, lw=0.7, alpha=0.25, zorder=1)
            ax.plot([row["occ_p10"], row["occ_p90"]], [idx + 0.10, idx + 0.10], color=occ_color, lw=1.2, alpha=0.38, solid_capstyle="round", zorder=1)
            ax.plot([row["rho_p10"], row["rho_p90"]], [idx - 0.10, idx - 0.10], color=rho_color, lw=1.2, alpha=0.45, solid_capstyle="round", zorder=1)
            ax.plot([row["occ_q1"], row["occ_q3"]], [idx + 0.10, idx + 0.10], color=occ_color, lw=3.2, alpha=0.86, solid_capstyle="round", zorder=2)
            ax.plot([row["rho_q1"], row["rho_q3"]], [idx - 0.10, idx - 0.10], color=rho_color, lw=3.2, alpha=0.92, solid_capstyle="round", zorder=2)
            ax.scatter(row["occ_med"], idx + 0.10, s=26, facecolor="white", edgecolor=occ_color, linewidth=1.1, zorder=3)
            ax.scatter(row["rho_med"], idx - 0.10, s=28, facecolor=rho_color, edgecolor=text_primary, linewidth=0.45, zorder=3)
            delta = float(row["gain_med"])
            ax.text(
                0.875,
                idx,
                f"{delta:+.2f}",
                ha="right",
                va="center",
                fontsize=5.9,
                color=gain_color if delta < 0 else text_primary,
            )
        ax.set_yticks(y, df["label"])
        ax.set_xlim(-0.10, 0.90)
        ax.set_title(title, loc="left", fontsize=8.0, pad=4, fontweight="bold")
        ax.grid(axis="x", color="#E6E6E6", linewidth=0.55)
        ax.tick_params(axis="x", labelsize=6.3, length=2)
        ax.tick_params(axis="y", labelsize=6.3, length=0)
        ax.text(0.875, -0.55, r"$\Delta\rho$", ha="right", va="center", fontsize=6.2, fontweight="bold", color=text_secondary)
        ax.invert_yaxis()
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
            ax.spines[spine].set_color(text_primary)
        ax.spines["bottom"].set_linewidth(0.7)
        ax.spines["bottom"].set_color(text_primary)
    fig.text(0.50, 0.055, "Spearman correlation with true relative dwell time", ha="center", va="center", fontsize=7.2)
    figure_style.save_figure(fig, output, config)


def write_design_report(root: Path) -> None:
    text = """# Experiment 7 Selected Figure Design

## Selected Design

The selected Experiment 7 visualization is a factor-level profile. It
compresses the topology robustness experiment into interpretable summaries for
topology, long-dwell placement and sparsity while retaining uncertainty.

## Public Design Scheme Audit For Selected D

Shared design patterns adopted:

1. `shared_axis_small_multiples`: topology, long-dwell placement and sparsity
   are shown as aligned panels on the same x scale.
2. `summary_with_raw_points`: repeat-level evidence is summarized as median,
   IQR and 10-90% ranges rather than as a single point estimate.
3. `direct_in_panel_annotation`: median delta rho is placed as a right-side
   numeric column in each panel.
4. `embedded_micro_legends`: the R* / occupancy / median-shift legend is placed
   in the top margin to avoid a detached legend block.

Shared patterns reviewed but not used:

- `boxed_context_insets` and `phase_bands_and_arrows` are not used because
  Experiment 7 is a factor robustness profile rather than a mechanistic
  time-series figure.
- Matrix-first rules remain useful for the full 48-condition view, but the
  selected D design intentionally prioritizes a compact factor-level summary.

## Use Recommendation

Use this figure as the Experiment 7 main-text visualization. The old exploratory
matrix/facet/ranking variants were removed after selection.

## Retained Output

- `figures/Figure_E7_topology_robustness`
"""
    (root / "experiment_07_selected_figure_design.md").write_text(text, encoding="utf-8")


def main() -> None:
    config = load_config()
    figure_style.configure_matplotlib(config)
    root = Path(config["result_root"])
    metrics = prepare_metrics(root)
    render_design_d(metrics, root / "figures" / "Figure_E7_topology_robustness", config)
    write_design_report(root)


if __name__ == "__main__":
    main()

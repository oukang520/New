"""Run Experiment 16: real-cohort relative dwell-time gene topology.

This experiment is the visual bridge from real input data to the method's core
innovation: high-R* states embedded in an MHN-derived gene-addition topology.
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
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import figure_style


CONFIG_PATH = Path("configs/experiment_16.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Rel-ObsTQ-MHN Experiment 16.")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--result-root")
    return parser.parse_args()


def genotype_events(genotype: str) -> list[str]:
    text = str(genotype)
    return [] if text == "WT" or not text.strip() else text.split("+")


def compact_genotype(genotype: str, max_events: int = 3) -> str:
    events = genotype_events(genotype)
    if not events:
        return "WT"
    if len(events) <= max_events:
        return "+".join(events)
    return "+".join(events[:max_events]) + "+..."


def compact_state(state: str, max_events: int = 3) -> str:
    stage, genotype = str(state).split("::", 1)
    prefix = "P" if stage == "primary" else "M"
    return f"{prefix}:{compact_genotype(genotype, max_events)}"


def event_added(source_state: str, target_state: str) -> str:
    _, source_genotype = source_state.split("::", 1)
    _, target_genotype = target_state.split("::", 1)
    source = set(genotype_events(source_genotype))
    target = set(genotype_events(target_genotype))
    added = sorted(target.difference(source))
    return added[0] if added else ""


def read_dataset_tables(dataset: str, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores = pd.read_csv(Path(config["experiment_05_root"]) / dataset / "tables" / "state_scores.tsv", sep="\t")
    top = pd.read_csv(Path(config["experiment_05_root"]) / dataset / "tables" / "top_bottleneck_states_high_confidence.tsv", sep="\t")
    edges = pd.read_csv(Path(config["experiment_04_root"]) / dataset / "tables" / "predecessor_edges_rule_a_one_step.tsv", sep="\t")
    return scores, top, edges


def build_path(target_state: str, score_by_state: dict[str, dict], max_depth: int) -> list[str]:
    path = [target_state]
    seen = {target_state}
    current = target_state
    for _ in range(max_depth):
        row = score_by_state.get(current)
        if row is None:
            break
        predecessor = str(row.get("dominant_predecessor", "") or "")
        if not predecessor or predecessor == "nan" or predecessor == current or predecessor in seen:
            break
        path.insert(0, predecessor)
        seen.add(predecessor)
        current = predecessor
        if predecessor.endswith("::WT"):
            break
    return path


def select_top_states(top: pd.DataFrame, scores: pd.DataFrame, config: dict) -> pd.DataFrame:
    total_n = int(config["analysis"]["top_paths_per_cohort"])
    core_n = int(config["analysis"].get("top_rstar_paths_per_cohort", min(4, total_n)))
    long_n = int(config["analysis"].get("long_event_paths_per_cohort", max(0, total_n - core_n)))
    long_event_threshold = int(config["analysis"].get("long_event_event_count_threshold", 3))
    long_min_count = int(config["analysis"].get("long_event_minimum_state_count", config["analysis"].get("minimum_state_count", 5)))

    if len(top) >= core_n:
        core = top.head(core_n).copy()
    else:
        core = scores[scores["eligible_experiment5"].astype(bool)].nlargest(core_n, "R_star").copy()
    core["selection_type"] = "top_rstar"

    selected_states = set(core["state"].astype(str))
    long_pool = scores.copy()
    long_pool = long_pool[~long_pool["state"].astype(str).isin(selected_states)]
    long_pool = long_pool[
        long_pool["R_star"].replace([np.inf, -np.inf], np.nan).notna()
        & long_pool["event_count"].gt(long_event_threshold)
        & long_pool["N_v"].ge(long_min_count)
        & long_pool["dominant_predecessor"].notna()
        & long_pool["dominant_predecessor"].astype(str).ne("")
        & long_pool["dominant_predecessor"].astype(str).ne("nan")
    ].copy()
    long_selected = long_pool.sort_values(["R_star", "N_v"], ascending=[False, False]).head(long_n)
    long_selected["selection_type"] = "long_event_rstar"

    selected = pd.concat([core, long_selected], ignore_index=True, sort=False)
    if len(selected) < total_n:
        filler_pool = scores.copy()
        filler_pool = filler_pool[~filler_pool["state"].astype(str).isin(set(selected["state"].astype(str)))]
        filler_pool = filler_pool[
            filler_pool["R_star"].replace([np.inf, -np.inf], np.nan).notna()
            & filler_pool["eligible_experiment5"].astype(bool)
        ].copy()
        filler = filler_pool.sort_values(["R_star", "N_v"], ascending=[False, False]).head(total_n - len(selected))
        filler["selection_type"] = "fallback_rstar"
        selected = pd.concat([selected, filler], ignore_index=True, sort=False)

    selected = selected.head(total_n).copy()
    selected["selection_rank"] = range(1, len(selected) + 1)
    return selected


def build_topology_tables(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    node_rows = []
    edge_rows = []
    path_rows = []
    audit_rows = []
    for dataset, ds_cfg in config["datasets"].items():
        scores, top, edges = read_dataset_tables(dataset, config)
        short = ds_cfg["short_name"]
        selected = select_top_states(top, scores, config)
        score_by_state = scores.set_index("state").to_dict(orient="index")
        edge_lookup = {
            (str(row.source_state), str(row.target_state)): row
            for row in edges.itertuples(index=False)
        }
        selected_states = set(selected["state"].astype(str))
        path_count = 0
        for selected_row in selected.itertuples():
            target = str(selected_row.state)
            path = build_path(target, score_by_state, int(config["analysis"]["max_path_depth"]))
            path_count += 1
            for position, state in enumerate(path):
                state_info = score_by_state.get(state, {})
                stage, genotype = state.split("::", 1)
                path_rows.append(
                    {
                        "dataset_name": dataset,
                        "short_name": short,
                        "path_rank": int(getattr(selected_row, "selection_rank", path_count)),
                        "selection_type": str(getattr(selected_row, "selection_type", "")),
                        "path_position": int(position),
                        "state": state,
                        "target_state": target,
                        "is_target": state == target,
                    }
                )
                node_rows.append(
                    {
                        "dataset_name": dataset,
                        "short_name": short,
                        "path_rank": int(getattr(selected_row, "selection_rank", path_count)),
                        "selection_type": str(getattr(selected_row, "selection_type", "")),
                        "path_position": int(position),
                        "state": state,
                        "stage": stage,
                        "genotype": genotype,
                        "event_count": int(state_info.get("event_count", len(genotype_events(genotype))) if pd.notna(state_info.get("event_count", np.nan)) else len(genotype_events(genotype))),
                        "N_v": float(state_info.get("N_v", np.nan)),
                        "L_v": float(state_info.get("L_v", np.nan)),
                        "F_hat": float(state_info.get("F_hat", np.nan)),
                        "R_star": float(state_info.get("R_star", np.nan)),
                        "log2_R_star": float(state_info.get("log2_R_star", np.nan)),
                        "stability_high_confidence": float(state_info.get("stability_high_confidence", np.nan)),
                        "is_selected_top_state": state in selected_states,
                        "is_path_target": state == target,
                        "label": compact_state(state, 3),
                    }
                )
            for source, target_state in zip(path[:-1], path[1:]):
                edge = edge_lookup.get((source, target_state))
                target_info = score_by_state.get(target_state, {})
                edge_rows.append(
                    {
                        "dataset_name": dataset,
                        "short_name": short,
                        "path_rank": int(getattr(selected_row, "selection_rank", path_count)),
                        "selection_type": str(getattr(selected_row, "selection_type", "")),
                        "source_state": source,
                        "target_state": target_state,
                        "event_added": str(edge.event_added) if edge is not None else event_added(source, target_state),
                        "edge_probability": float(edge.edge_probability) if edge is not None else np.nan,
                        "inflow_contribution": float(edge.inflow_contribution) if edge is not None else float(target_info.get("dominant_contribution", np.nan)),
                        "target_R_star": float(target_info.get("R_star", np.nan)),
                    }
                )
        audit_rows.append(
            {
                "dataset_name": dataset,
                "short_name": short,
                "display_paths": int(path_count),
                "top_rstar_paths": int(selected["selection_type"].eq("top_rstar").sum()),
                "long_event_rstar_paths": int(selected["selection_type"].eq("long_event_rstar").sum()),
                "fallback_paths": int(selected["selection_type"].eq("fallback_rstar").sum()),
                "unique_nodes": int(pd.DataFrame([row for row in node_rows if row["dataset_name"] == dataset])["state"].nunique()),
                "edges": int(sum(row["dataset_name"] == dataset for row in edge_rows)),
                "median_target_R_star": float(selected["R_star"].median()) if "R_star" in selected else np.nan,
            }
        )
    nodes = pd.DataFrame(node_rows).drop_duplicates(["dataset_name", "path_rank", "state", "path_position"])
    edges = pd.DataFrame(edge_rows)
    paths = pd.DataFrame(path_rows)
    audit = pd.DataFrame(audit_rows)
    return nodes, edges, paths, audit


def save_square(fig: plt.Figure, output: Path, config: dict) -> None:
    figure_style.save_figure_panels(fig, output, config)


def draw_pipeline(ax: plt.Axes, audit: pd.DataFrame, config: dict, colors: dict) -> None:
    ax.axis("off")
    text_primary = colors.get("text", {}).get("primary", "#263238")
    text_secondary = colors.get("text", {}).get("secondary", "#4E5A5E")
    grid_color = colors.get("text", {}).get("grid", "#E6E6E6")
    total_nodes = int(audit["unique_nodes"].sum())
    total_edges = int(audit["edges"].sum())
    blocks = [
        ("Real input", "p=15 driver matrix\nstage::genotype states"),
        ("MHN backbone", "learn P(u -> v)\nfrom real cohorts"),
        ("Expected inflow", r"$F_v=\sum_u L_uP(u\to v)$" + f"\n{total_edges} dominant path edges"),
        ("Relative dwell", r"$R^*_v=L_v/F_v$" + f"\n{total_nodes} displayed path nodes"),
    ]
    x0 = 0.02
    width = 0.215
    gap = 0.030
    for i, (title, body) in enumerate(blocks):
        x = x0 + i * (width + gap)
        rect = FancyBboxPatch(
            (x, 0.25),
            width,
            0.54,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            facecolor="#F8FAFA",
            edgecolor=grid_color,
            linewidth=0.65,
        )
        ax.add_patch(rect)
        ax.text(x + 0.015, 0.66, title, ha="left", va="center", fontsize=6.5, fontweight="bold", color=text_primary)
        ax.text(x + 0.015, 0.45, body, ha="left", va="center", fontsize=5.2, color=text_secondary, linespacing=1.25)
        if i < len(blocks) - 1:
            ax.annotate(
                "",
                xy=(x + width + gap * 0.75, 0.52),
                xytext=(x + width + gap * 0.18, 0.52),
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color=text_secondary, shrinkA=0, shrinkB=0),
            )


def draw_topology_panel(
    ax: plt.Axes,
    dataset: str,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    config: dict,
    cmap: mcolors.Colormap,
    norm: mcolors.Normalize,
    color: str,
    text_primary: str,
    text_secondary: str,
    grid_color: str,
    show_xlabel: bool = True,
) -> None:
    ax.set_facecolor("white")
    sub_nodes = nodes[nodes["dataset_name"].eq(dataset)].copy()
    sub_edges = edges[edges["dataset_name"].eq(dataset)].copy()
    if sub_nodes.empty:
        ax.axis("off")
        return
    path_order = sorted(sub_nodes["path_rank"].unique())
    y_by_rank = {rank: len(path_order) - i for i, rank in enumerate(path_order)}
    position = {}
    for row in sub_nodes.itertuples():
        position[(row.path_rank, row.state)] = (float(row.path_position), float(y_by_rank[row.path_rank]))
    max_x = max(float(sub_nodes["path_position"].max()), 1.0)
    right_pad = 1.95 if max_x <= 5 else 2.35
    ax.set_xlim(-0.35, max_x + right_pad)
    ax.set_ylim(0.30, len(path_order) + 0.85)
    for rank in path_order:
        ax.axhline(y_by_rank[rank], color=grid_color, lw=0.35, zorder=0)
    max_contrib = np.nanmax(sub_edges["inflow_contribution"].to_numpy(dtype=float)) if len(sub_edges) else np.nan
    for row in sub_edges.itertuples():
        source_key = (row.path_rank, row.source_state)
        target_key = (row.path_rank, row.target_state)
        if source_key not in position or target_key not in position:
            continue
        x1, y1 = position[source_key]
        x2, y2 = position[target_key]
        width = 0.50
        if np.isfinite(max_contrib) and max_contrib > 0 and np.isfinite(row.inflow_contribution):
            width = 0.40 + 1.05 * float(row.inflow_contribution) / max_contrib
        arrow = FancyArrowPatch(
            (x1 + 0.10, y1),
            (x2 - 0.10, y2),
            arrowstyle="-|>",
            mutation_scale=5.8,
            linewidth=width,
            color=color,
            alpha=0.58,
            zorder=1,
        )
        ax.add_patch(arrow)
        if row.event_added:
            ax.text((x1 + x2) / 2, y1 + 0.08, str(row.event_added), ha="center", va="bottom", fontsize=3.9, color=text_secondary)
    for row in sub_nodes.itertuples():
        x, y = position[(row.path_rank, row.state)]
        value = float(row.log2_R_star) if np.isfinite(row.log2_R_star) else np.nan
        face = cmap(norm(np.clip(value, norm.vmin, norm.vmax))) if np.isfinite(value) else "#F2F2F2"
        size = 28.0
        if np.isfinite(row.N_v):
            size = float(np.clip(18 + 6 * np.sqrt(row.N_v), 26, 82))
        edge_lw = 0.32 + (0.62 if bool(row.is_path_target) else 0.0)
        ax.scatter(x, y, s=size, facecolor=face, edgecolor=text_primary, linewidth=edge_lw, zorder=3)
        if row.path_position == 0:
            ax.text(x, y - 0.22, "WT", ha="center", va="top", fontsize=4.0, color=text_secondary)
        if bool(row.is_path_target):
            is_long_event_target = str(getattr(row, "selection_type", "")) == "long_event_rstar"
            if is_long_event_target and np.isfinite(row.R_star):
                label = f"R*={row.R_star:.1f}"
            else:
                label = f"{row.label}  R*={row.R_star:.1f}" if np.isfinite(row.R_star) else row.label
            target_font = 4.05 if is_long_event_target else 4.25
            label_box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.25}
            ax.text(x + 0.26, y, label, ha="left", va="center", fontsize=target_font, color=text_primary, bbox=label_box)
    ax.set_xticks(range(int(max_x) + 1))
    ax.set_xlabel("Gene events added along dominant predecessor path" if show_xlabel else "", fontsize=5.2)
    ax.set_yticks([])
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.55)
    ax.tick_params(axis="x", labelsize=4.7, length=2.0, width=0.5)


def plot_main_figure(nodes: pd.DataFrame, edges: pd.DataFrame, audit: pd.DataFrame, output: Path, config: dict) -> None:
    figure_style.configure_matplotlib(config)
    colors = figure_style.colors(config)
    cat = figure_style.categorical_palette(config)
    text_primary = colors.get("text", {}).get("primary", "#263238")
    text_secondary = colors.get("text", {}).get("secondary", "#4E5A5E")
    grid_color = colors.get("text", {}).get("grid", "#E6E6E6")
    cohort_colors = [cat.get("lavender", "#B5AED5"), cat.get("sky_blue", "#B2E6FD"), cat.get("sage", "#B8D2CC"), cat.get("coral", "#E8B2A7")]
    palette = figure_style.continuous_palette("dwell_rank", config)
    cmap = mcolors.LinearSegmentedColormap.from_list("rstar", palette)
    norm = mcolors.Normalize(
        vmin=float(config["analysis"]["color_log2_r_min"]),
        vmax=float(config["analysis"]["color_log2_r_max"]),
    )
    fig = plt.figure(figsize=(7.2, 7.2))
    fig.text(0.075, 0.972, "Experiment 16 | Real-cohort relative dwell topology", ha="left", va="top", fontsize=9.3, fontweight="bold", color=text_primary)
    fig.text(
        0.075,
        0.947,
        "Rows 1-4 preserve top R* routes; rows 5-6 add the highest-R* remaining states with >3 events.",
        ha="left",
        va="top",
        fontsize=5.8,
        color=text_secondary,
    )

    panel_positions = [
        [0.075, 0.565, 0.405, 0.310],
        [0.550, 0.565, 0.390, 0.310],
        [0.075, 0.145, 0.405, 0.310],
        [0.550, 0.145, 0.390, 0.310],
    ]
    panel_letters = ["a", "b", "c", "d"]
    for idx, ((dataset, ds_cfg), pos) in enumerate(zip(config["datasets"].items(), panel_positions)):
        ax = fig.add_axes(pos)
        draw_topology_panel(
            ax,
            dataset,
            nodes,
            edges,
            config,
            cmap,
            norm,
            cohort_colors[idx],
            text_primary,
            text_secondary,
            grid_color,
            show_xlabel=idx >= 2,
        )
        ax.text(-0.16, 1.070, panel_letters[idx], transform=ax.transAxes, fontsize=10.5, fontweight="bold", ha="left", va="top", color=text_primary)
        ax.text(0.00, 1.070, ds_cfg["display_name"], transform=ax.transAxes, fontsize=7.3, fontweight="bold", ha="left", va="top", color=text_primary)
    cax = fig.add_axes([0.375, 0.060, 0.250, 0.012])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=4.8, length=1.6, width=0.45)
    cb.outline.set_linewidth(0.35)
    cb.set_label(r"node color: $\log_2 R^*$", fontsize=5.2, labelpad=1)
    fig.text(0.075, 0.080, "Node area: observed state count; bold outline: displayed top R* target", ha="left", va="center", fontsize=4.9, color=text_secondary)
    fig.text(0.075, 0.061, "Arrow label: added driver event along the dominant predecessor path.", ha="left", va="center", fontsize=4.9, color=text_secondary)
    save_square(fig, output, config)


def write_reports(root: Path, config: dict, audit: pd.DataFrame, nodes: pd.DataFrame, edges: pd.DataFrame) -> None:
    cohort_list = ", ".join(config["datasets"].keys())
    lines = [
        "# Experiment 16 Summary",
        "",
        "## Real-Cohort Topology Audit",
        "",
        "| Cohort | Display paths | Top R* paths | Long-event paths | Unique nodes | Edges | Median target R* |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in audit.itertuples():
        lines.append(f"| {row.short_name} | {row.display_paths} | {row.top_rstar_paths} | {row.long_event_rstar_paths} | {row.unique_nodes} | {row.edges} | {row.median_target_R_star:.2f} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This experiment is a real-data topology display. It follows dominant MHN-derived predecessor paths and overlays the state-level relative dwell score R*. It is not a new statistical benchmark; its purpose is to make the method's central object visible: high-R* states positioned on gene-addition trajectories.",
            "",
            "For each cohort, paths 1-4 preserve the original top-R* route selection. Paths 5-6 are selected from the remaining states with event_count > 3 by descending R*, using a minimum displayed-state count to avoid single-sample artifacts.",
            "",
            "The selected AACR cohorts show clear high-R* terminal states. The experiment now focuses on the three retained tumor-type cohorts so the real-data topology display is not diluted by sample-limited cohorts.",
        ]
    )
    (root / "experiment_16_summary.md").write_text("\n".join(lines), encoding="utf-8")

    protocol = f"""# Experiment 16 Protocol Audit

## Rationale

The first 15 experiments validate pieces of the method, but none fully displays
the real-cohort topology carrying relative dwell R*. Experiment 16 fills that
gap by embedding real cohort states in dominant MHN predecessor paths.

## Inputs

- Experiment 5 state-level R* and bootstrap-supported top states.
- Experiment 4 one-step MHN-derived predecessor edges and inflow contribution.
- Real experiment-ready cohorts: {cohort_list}.

## Path Selection

- Paths 1-4: unchanged top-R* routes from the high-confidence Experiment 5 state list.
- Paths 5-6: highest-R* remaining states with event_count > 3 and sufficient observed support.

## Figure Design Patterns

{figure_style.design_patterns_markdown(config)}
"""
    (root / "experiment_16_protocol_audit.md").write_text(protocol, encoding="utf-8")

    review = f"""# Experiment 16 Scientific Review

Experiment 16 is the direct visual bridge from real mutation data to the
Rel-ObsTQ-MHN innovation. Each path is constructed by recursively following the
dominant predecessor of a selected R* state. The first four paths retain the
strongest high-confidence R* targets, while the final two paths add longer
multi-event targets selected by R* among remaining states with event_count > 3.
Edges therefore represent the learned MHN/inflow-supported gene-addition route,
while node color represents relative dwell time.

The topology must be interpreted as a dominant local path summary, not as a full
patient lineage tree. The full node and edge tables are exported for audit. The
scope is restricted to the retained AACR cohorts, where the relative-dwell
contrast is strong enough for the real-topology display.

## Design Sources

{figure_style.design_sources_markdown(config)}

## Design Rules

{figure_style.design_rules_markdown(config)}
"""
    (root / "experiment_16_scientific_review.md").write_text(review, encoding="utf-8")

    design = f"""# Experiment 16 Figure Design Review

The figure uses four compact real-cohort dominant-path topologies without a
separate method-flow schematic. This design was chosen instead of a force-directed
graph because the biological ordering is gene-addition count, and the scientific
claim concerns where high relative dwell states sit along that ordered topology.
Direct labels are used for target states; intermediate labels are kept to the
added driver event to avoid unreadable genotype strings.

## Public Figure Style Rules Used

{figure_style.design_rules_markdown(config)}
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
    nodes, edges, paths, audit = build_topology_tables(config)
    nodes.to_csv(tables / "real_topology_nodes.tsv", sep="\t", index=False)
    edges.to_csv(tables / "real_topology_edges.tsv", sep="\t", index=False)
    paths.to_csv(tables / "real_topology_paths.tsv", sep="\t", index=False)
    audit.to_csv(tables / "real_topology_audit.tsv", sep="\t", index=False)
    plot_main_figure(nodes, edges, audit, figures / "Figure_E16_real_relative_dwell_topology", config)
    write_reports(root, config, audit, nodes, edges)


if __name__ == "__main__":
    main()

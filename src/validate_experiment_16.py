"""Validate Experiment 16 real-cohort relative dwell topology outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_16.yaml")
    parser.add_argument("--result-root")
    return parser.parse_args()


def check(records: list[dict], category: str, name: str, passed: bool, detail: str) -> None:
    records.append({"category": category, "check": name, "passed": bool(passed), "detail": detail})


def figure_boundary(path: Path, min_width: int = 3800, min_height: int = 3800) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    with Image.open(path) as image:
        width, height = image.size
        array = np.asarray(image.convert("RGB"))
    border = np.concatenate(
        [
            array[:10].reshape(-1, 3),
            array[-10:].reshape(-1, 3),
            array[:, :10].reshape(-1, 3),
            array[:, -10:].reshape(-1, 3),
        ]
    )
    edge_nonwhite = float(np.mean(np.any(border < 245, axis=1)))
    aspect = width / height if height else np.inf
    ok = width >= min_width and height >= min_height and 0.96 <= aspect <= 1.04 and edge_nonwhite < 0.05
    return ok, f"size={width}x{height}; aspect={aspect:.3f}; edge_nonwhite={edge_nonwhite:.4f}"


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    root = Path(args.result_root or config["result_root"]).resolve()
    resolved = root / "resolved_config.json"
    if resolved.exists():
        config = json.loads(resolved.read_text(encoding="utf-8"))
    tables = root / "tables"
    records: list[dict] = []
    required = [
        "real_topology_nodes.tsv",
        "real_topology_edges.tsv",
        "real_topology_paths.tsv",
        "real_topology_audit.tsv",
    ]
    missing = [name for name in required if not (tables / name).exists()]
    check(records, "structural", "required_tables", not missing, "OK" if not missing else "; ".join(missing))
    if missing:
        pd.DataFrame(records).to_csv(root / "experiment_16_validation.csv", index=False)
        raise SystemExit(1)

    nodes = pd.read_csv(tables / "real_topology_nodes.tsv", sep="\t")
    edges = pd.read_csv(tables / "real_topology_edges.tsv", sep="\t")
    paths = pd.read_csv(tables / "real_topology_paths.tsv", sep="\t")
    audit = pd.read_csv(tables / "real_topology_audit.tsv", sep="\t")
    expected_datasets = set(config["datasets"])
    expected_paths = int(config["analysis"]["top_paths_per_cohort"])

    check(records, "structural", "dataset_coverage", set(audit["dataset_name"]) == expected_datasets, str(sorted(audit["dataset_name"].unique())))
    node_ok = set(nodes["dataset_name"]) == expected_datasets and nodes["R_star"].replace([np.inf, -np.inf], np.nan).notna().any()
    check(records, "topology", "nodes", bool(node_ok), f"rows={len(nodes)}")
    edge_ok = set(edges["dataset_name"]) == expected_datasets and len(edges) > 0 and edges["source_state"].notna().all() and edges["target_state"].notna().all()
    check(records, "topology", "edges", bool(edge_ok), f"rows={len(edges)}")
    path_counts = paths.groupby("dataset_name")["path_rank"].nunique().to_dict()
    path_ok = all(path_counts.get(dataset, 0) == expected_paths for dataset in expected_datasets)
    check(records, "topology", "path_counts", bool(path_ok), str(path_counts))
    expected_top = int(config["analysis"].get("top_rstar_paths_per_cohort", 4))
    expected_long = int(config["analysis"].get("long_event_paths_per_cohort", max(0, expected_paths - expected_top)))
    long_event_threshold = int(config["analysis"].get("long_event_event_count_threshold", 3))
    path_level = paths[["dataset_name", "path_rank", "selection_type"]].drop_duplicates()
    selection_counts = path_level.groupby(["dataset_name", "selection_type"])["path_rank"].nunique().unstack(fill_value=0)
    selection_ok = True
    for dataset in expected_datasets:
        subset = path_level[path_level["dataset_name"].eq(dataset)]
        top_ok = subset[subset["path_rank"].le(expected_top)]["selection_type"].eq("top_rstar").all()
        long_ok = subset[subset["path_rank"].gt(expected_top)]["selection_type"].eq("long_event_rstar").all()
        selection_ok = selection_ok and bool(top_ok and long_ok)
    target_nodes_for_selection = nodes[nodes["is_path_target"].astype(bool)].copy()
    long_targets = target_nodes_for_selection[target_nodes_for_selection["selection_type"].eq("long_event_rstar")]
    long_target_ok = len(long_targets) == len(expected_datasets) * expected_long and long_targets["event_count"].gt(long_event_threshold).all()
    check(
        records,
        "topology",
        "path_selection",
        bool(selection_ok and long_target_ok),
        selection_counts.to_dict() | {"long_target_min_event_count": int(long_targets["event_count"].min()) if len(long_targets) else None},
    )
    target_nodes = nodes[nodes["is_path_target"].astype(bool)].copy()
    finite_targets = target_nodes["R_star"].replace([np.inf, -np.inf], np.nan).notna().all()
    expected_target_count = len(expected_datasets) * expected_paths
    target_ok = len(target_nodes) == expected_target_count and finite_targets
    cohort_medians = target_nodes.groupby("dataset_name")["R_star"].median().round(3).to_dict()
    check(
        records,
        "topology",
        "target_R_star",
        bool(target_ok),
        f"targets={len(target_nodes)}/{expected_target_count}; min_R={target_nodes['R_star'].min():.3f}; medians={cohort_medians}",
    )
    audit_ok = audit["display_paths"].eq(expected_paths).all() and audit["unique_nodes"].gt(0).all() and audit["edges"].gt(0).all()
    check(records, "statistics", "audit", bool(audit_ok), f"rows={len(audit)}")

    figure_base = root / "figures" / "Figure_E16_real_relative_dwell_topology"
    check(records, "figure", "figure_files", figure_base.with_suffix(".png").exists() and figure_base.with_suffix(".pdf").exists(), f"png={figure_base.with_suffix('.png').exists()}; pdf={figure_base.with_suffix('.pdf').exists()}")
    boundary_ok, boundary_detail = figure_boundary(figure_base.with_suffix(".png"))
    check(records, "figure", "figure_boundary", boundary_ok, boundary_detail)
    for report in [
        "experiment_16_protocol_audit.md",
        "experiment_16_summary.md",
        "experiment_16_scientific_review.md",
        "top_journal_figure_design_review.md",
    ]:
        check(records, "structural", f"report_{report}", (root / report).exists(), "exists" if (root / report).exists() else "missing")

    result = pd.DataFrame(records)
    result.to_csv(root / "experiment_16_validation.csv", index=False)
    lines = ["# Experiment 16 Validation", "", "| Category | Check | Passed | Detail |", "|---|---|---:|---|"]
    for row in result.itertuples():
        lines.append(f"| {row.category} | {row.check} | {'PASS' if row.passed else 'FAIL'} | {row.detail} |")
    lines.append("")
    lines.append(f"- Overall validation: {'PASS' if result['passed'].all() else 'FAIL'} ({int(result['passed'].sum())}/{len(result)})")
    (root / "experiment_16_validation.md").write_text("\n".join(lines), encoding="utf-8")
    if not result["passed"].all():
        raise SystemExit(1)


if __name__ == "__main__":
    main()

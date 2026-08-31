"""Validate Experiment 15 innovation-specific falsification outputs."""

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
    parser.add_argument("--config", default="configs/experiment_15.yaml")
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
        "matched_decoy_contrast.tsv",
        "matched_decoy_summary.tsv",
        "inflow_pairing_falsification_replicates.tsv",
        "inflow_pairing_falsification_summary.tsv",
    ]
    missing = [name for name in required if not (tables / name).exists()]
    check(records, "structural", "required_tables", not missing, "OK" if not missing else "; ".join(missing))
    if missing:
        pd.DataFrame(records).to_csv(root / "experiment_15_validation.csv", index=False)
        raise SystemExit(1)

    matched = pd.read_csv(tables / "matched_decoy_contrast.tsv", sep="\t")
    matched_summary = pd.read_csv(tables / "matched_decoy_summary.tsv", sep="\t")
    reps = pd.read_csv(tables / "inflow_pairing_falsification_replicates.tsv", sep="\t")
    inflow = pd.read_csv(tables / "inflow_pairing_falsification_summary.tsv", sep="\t")
    expected_datasets = set(config["datasets"])

    check(records, "structural", "dataset_coverage", set(matched_summary["dataset_name"]) == expected_datasets and set(inflow["dataset_name"]) == expected_datasets, str(sorted(matched_summary["dataset_name"].unique())))
    matched_ok = (
        set(matched["dataset_name"]) == expected_datasets
        and matched["matched_percentile"].between(0, 1).all()
        and matched["decoy_count"].gt(0).all()
        and matched["R_star"].replace([np.inf, -np.inf], np.nan).notna().all()
    )
    check(records, "matched_decoy", "state_contrast", bool(matched_ok), f"rows={len(matched)}; min_decoys={matched['decoy_count'].min() if len(matched) else 'NA'}")
    summary_ok = (
        len(matched_summary) == len(expected_datasets)
        and matched_summary["median_matched_percentile"].between(0, 1).all()
        and matched_summary["fraction_above_decoy_q90"].between(0, 1).all()
    )
    check(records, "matched_decoy", "summary", bool(summary_ok), f"rows={len(matched_summary)}")

    expected_repeats = int(config["analysis"].get("inflow_shuffle_replicates", 400))
    reps_ok = (
        set(reps["dataset_name"]) == expected_datasets
        and reps.groupby("dataset_name")["repeat"].nunique().eq(expected_repeats).all()
        and reps["top_overlap_fraction"].between(0, 1).all()
    )
    check(records, "falsification", "inflow_pairing_replicates", bool(reps_ok), f"rows={len(reps)}; expected_repeats={expected_repeats}")
    inflow_ok = (
        len(inflow) == len(expected_datasets)
        and inflow["median_shuffled_overlap"].between(0, 1).all()
        and inflow["median_overlap_loss"].between(0, 1).all()
    )
    check(records, "falsification", "inflow_pairing_summary", bool(inflow_ok), f"rows={len(inflow)}")

    figure_base = root / "figures" / "Figure_E15_innovation_falsification_controls"
    check(records, "figure", "figure_files", figure_base.with_suffix(".png").exists() and figure_base.with_suffix(".pdf").exists(), f"png={figure_base.with_suffix('.png').exists()}; pdf={figure_base.with_suffix('.pdf').exists()}")
    boundary_ok, boundary_detail = figure_boundary(figure_base.with_suffix(".png"))
    check(records, "figure", "figure_boundary", boundary_ok, boundary_detail)
    for report in [
        "experiment_15_protocol_audit.md",
        "experiment_15_summary.md",
        "experiment_15_scientific_review.md",
        "top_journal_figure_design_review.md",
    ]:
        check(records, "structural", f"report_{report}", (root / report).exists(), "exists" if (root / report).exists() else "missing")

    result = pd.DataFrame(records)
    result.to_csv(root / "experiment_15_validation.csv", index=False)
    lines = ["# Experiment 15 Validation", "", "| Category | Check | Passed | Detail |", "|---|---|---:|---|"]
    for row in result.itertuples():
        lines.append(f"| {row.category} | {row.check} | {'PASS' if row.passed else 'FAIL'} | {row.detail} |")
    lines.append("")
    lines.append(f"- Overall validation: {'PASS' if result['passed'].all() else 'FAIL'} ({int(result['passed'].sum())}/{len(result)})")
    (root / "experiment_15_validation.md").write_text("\n".join(lines), encoding="utf-8")
    if not result["passed"].all():
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Validate Experiment 13 split-cohort replication outputs."""

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
    parser.add_argument("--config", default="configs/experiment_13.yaml")
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
        "split_replication_metrics.tsv",
        "representative_split_state_scores.tsv",
        "clinical_direction_by_split.tsv",
        "split_replication_audit.tsv",
        "experiment_13_summary.tsv",
    ]
    missing = [name for name in required if not (tables / name).exists()]
    check(records, "structural", "required_tables", not missing, "OK" if not missing else "; ".join(missing))
    if missing:
        pd.DataFrame(records).to_csv(root / "experiment_13_validation.csv", index=False)
        raise SystemExit(1)

    metrics = pd.read_csv(tables / "split_replication_metrics.tsv", sep="\t")
    representative = pd.read_csv(tables / "representative_split_state_scores.tsv", sep="\t")
    clinical = pd.read_csv(tables / "clinical_direction_by_split.tsv", sep="\t")
    audit = pd.read_csv(tables / "split_replication_audit.tsv", sep="\t")
    summary = pd.read_csv(tables / "experiment_13_summary.tsv", sep="\t")
    expected = set(config["datasets"])
    split_repeats = int(config["analysis"]["split_repeats"])
    clinical_repeats = int(config["analysis"]["clinical_repeats"])
    top_k = int(config["analysis"]["top_k"])

    check(records, "structural", "dataset_coverage", set(summary["dataset_name"]) == expected and set(audit["dataset_name"]) == expected, str(sorted(summary["dataset_name"])))
    repeat_counts = metrics.groupby("dataset_name")["repeat"].nunique().to_dict()
    check(records, "replication", "split_repeat_counts", all(repeat_counts.get(dataset, 0) == split_repeats for dataset in expected), str(repeat_counts))
    clinical_counts = clinical.groupby("dataset_name")["repeat"].nunique().to_dict()
    check(records, "replication", "clinical_repeat_counts", all(clinical_counts.get(dataset, 0) == clinical_repeats for dataset in expected), str(clinical_counts))
    evaluable = summary[summary["evaluable"].astype(bool)]
    non_evaluable = summary[~summary["evaluable"].astype(bool)]
    common_ok = len(evaluable) >= 3 and evaluable["median_common_states"].ge(int(config["analysis"]["minimum_common_states"])).all()
    check(records, "statistics", "common_state_support", bool(common_ok), f"evaluable={evaluable['short_name'].tolist()}; underpowered={non_evaluable['short_name'].tolist()}")
    metrics_with_support = metrics[metrics["common_states"].ge(int(config["analysis"]["minimum_common_states"]))]
    rho_ok = metrics_with_support["spearman_rho"].between(-1, 1).all() and metrics_with_support["spearman_rho"].notna().all()
    check(records, "statistics", "spearman_range", bool(rho_ok), f"supported_rows={len(metrics_with_support)}; unsupported_nan={metrics['spearman_rho'].isna().sum()}")
    overlap_ok = metrics["top_overlap"].between(0, top_k).all()
    check(records, "statistics", "top_overlap_range", bool(overlap_ok), f"min={metrics['top_overlap'].min()}; max={metrics['top_overlap'].max()}")
    null_ok = metrics["top_overlap_null_mean"].dropna().ge(0).all() and metrics["top_overlap_enrichment"].dropna().ge(0).all()
    check(records, "statistics", "top_overlap_null_model", bool(null_ok), f"null_mean_median={metrics['top_overlap_null_mean'].median():.3f}")
    direction_ok = metrics["direction_concordance"].between(0, 1).all()
    check(records, "statistics", "direction_concordance_range", bool(direction_ok), f"nan={metrics['direction_concordance'].isna().sum()}")
    summary_ok = (
        evaluable["median_spearman_rho"].between(-1, 1).all()
        and summary["median_top10_overlap"].between(0, top_k).all()
        and evaluable["median_top10_null_mean"].notna().all()
        and evaluable["median_top10_enrichment"].notna().all()
        and summary["median_direction_concordance"].between(0, 1).all()
        and non_evaluable["median_spearman_rho"].isna().all()
    )
    check(records, "statistics", "summary_metric_ranges", bool(summary_ok), f"rows={len(summary)}; non_evaluable={len(non_evaluable)}")
    rep_ok = set(representative["dataset_name"]) == expected and representative["repeat"].eq(1).all()
    check(records, "replication", "representative_split_rows", bool(rep_ok), f"rows={len(representative)}")
    estimable_hr = clinical["split_A_ok"].astype(bool) & clinical["split_B_ok"].astype(bool)
    check(records, "clinical", "clinical_direction_estimable", int(estimable_hr.sum()) >= len(expected), f"estimable={int(estimable_hr.sum())}/{len(clinical)}")
    figure_base = root / "figures" / "Figure_E13_split_replication"
    check(records, "figure", "figure_files", figure_base.with_suffix(".png").exists() and figure_base.with_suffix(".pdf").exists(), f"png={figure_base.with_suffix('.png').exists()}; pdf={figure_base.with_suffix('.pdf').exists()}")
    boundary_ok, boundary_detail = figure_boundary(figure_base.with_suffix(".png"))
    check(records, "figure", "figure_boundary", boundary_ok, boundary_detail)
    for report in [
        "experiment_13_protocol_audit.md",
        "experiment_13_summary.md",
        "experiment_13_scientific_review.md",
        "top_journal_figure_design_review.md",
    ]:
        check(records, "structural", f"report_{report}", (root / report).exists(), "exists" if (root / report).exists() else "missing")

    result = pd.DataFrame(records)
    result.to_csv(root / "experiment_13_validation.csv", index=False)
    lines = [
        "# Experiment 13 Validation",
        "",
        "| Category | Check | Passed | Detail |",
        "|---|---|---:|---|",
    ]
    for row in result.itertuples():
        lines.append(f"| {row.category} | {row.check} | {'PASS' if row.passed else 'FAIL'} | {row.detail} |")
    lines.append("")
    lines.append(f"- Overall validation: {'PASS' if result['passed'].all() else 'FAIL'} ({int(result['passed'].sum())}/{len(result)})")
    (root / "experiment_13_validation.md").write_text("\n".join(lines), encoding="utf-8")
    if not result["passed"].all():
        raise SystemExit(1)


if __name__ == "__main__":
    main()

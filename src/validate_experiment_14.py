"""Validate Experiment 14 ablation and backbone replacement outputs."""

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
    parser.add_argument("--config", default="configs/experiment_14.yaml")
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
        "variant_state_scores.tsv",
        "variant_clinical_results.tsv",
        "variant_split_stability.tsv",
        "experiment_14_summary.tsv",
        "experiment_14_audit.tsv",
        "simulation_positive_control.tsv",
        "relative_dwell_decomposition.tsv",
        "backbone_top_state_retention.tsv",
        "relative_dwell_rank_lift.tsv",
    ]
    missing = [name for name in required if not (tables / name).exists()]
    check(records, "structural", "required_tables", not missing, "OK" if not missing else "; ".join(missing))
    if missing:
        pd.DataFrame(records).to_csv(root / "experiment_14_validation.csv", index=False)
        raise SystemExit(1)

    scores = pd.read_csv(tables / "variant_state_scores.tsv", sep="\t")
    clinical = pd.read_csv(tables / "variant_clinical_results.tsv", sep="\t")
    stability = pd.read_csv(tables / "variant_split_stability.tsv", sep="\t")
    summary = pd.read_csv(tables / "experiment_14_summary.tsv", sep="\t")
    simulation = pd.read_csv(tables / "simulation_positive_control.tsv", sep="\t")
    decomposition = pd.read_csv(tables / "relative_dwell_decomposition.tsv", sep="\t")
    retention = pd.read_csv(tables / "backbone_top_state_retention.tsv", sep="\t")
    rank_lift = pd.read_csv(tables / "relative_dwell_rank_lift.tsv", sep="\t")
    expected_datasets = set(config["datasets"])
    expected_variants = set(config["variants"])
    expected_rows = len(expected_datasets) * len(expected_variants)
    check(records, "structural", "dataset_coverage", set(summary["dataset_name"]) == expected_datasets, str(sorted(summary["dataset_name"].unique())))
    variant_counts = summary.groupby("dataset_name")["variant"].apply(set).to_dict()
    check(records, "structural", "variant_coverage", all(variant_counts.get(dataset, set()) == expected_variants for dataset in expected_datasets), str({k: sorted(v) for k, v in variant_counts.items()}))
    check(records, "statistics", "summary_row_count", len(summary) == expected_rows, f"rows={len(summary)}; expected={expected_rows}")
    clinical_ok = clinical["c_index"].between(0, 1).all() and clinical["hr"].gt(0).all() and clinical["ok"].astype(bool).sum() >= expected_rows - 1
    check(records, "clinical", "clinical_results", bool(clinical_ok), f"ok={int(clinical['ok'].astype(bool).sum())}/{len(clinical)}")
    stability_expected = len(expected_datasets) * len(expected_variants) * int(config["analysis"]["split_repeats"])
    stability_ok = len(stability) == stability_expected and stability["top_overlap"].between(0, int(config["analysis"]["top_k"])).all()
    check(records, "stability", "split_stability", bool(stability_ok), f"rows={len(stability)}; expected={stability_expected}")
    score_ok = scores["score"].replace([np.inf, -np.inf], np.nan).notna().sum() > 0 and set(scores["variant"]) == expected_variants
    check(records, "statistics", "state_scores", bool(score_ok), f"rows={len(scores)}")
    expected_top_rows = len(expected_datasets) * int(config["analysis"]["top_k"])
    decomp_ok = len(decomposition) == expected_top_rows and decomposition["L_v"].gt(0).all() and decomposition["F_MHN"].gt(0).all() and decomposition["R_raw"].gt(0).all()
    check(records, "relative_dwell", "decomposition", bool(decomp_ok), f"rows={len(decomposition)}; expected={expected_top_rows}")
    expected_retention_rows = len(expected_datasets) * (len(expected_variants) - 1)
    retention_ok = (
        len(retention) == expected_retention_rows
        and retention["retained_top_k"].between(0, int(config["analysis"]["top_k"])).all()
        and retention["retention_fraction"].between(0, 1).all()
        and retention["random_expected_overlap"].gt(0).all()
        and retention["enrichment_vs_random"].replace([np.inf, -np.inf], np.nan).notna().all()
    )
    check(records, "relative_dwell", "denominator_replacement", bool(retention_ok), f"rows={len(retention)}; expected={expected_retention_rows}")
    lift_ok = len(rank_lift) == expected_top_rows and rank_lift["percentile_R"].between(0, 100).all() and rank_lift["percentile_L"].between(0, 100).all()
    check(records, "relative_dwell", "rank_lift", bool(lift_ok), f"rows={len(rank_lift)}; expected={expected_top_rows}")
    sim_endpoints = {"Spearman", "Bottleneck ROC AUC", "Top-5 precision", "Recall@5"}
    sim_ok = set(simulation["endpoint"]) == sim_endpoints and simulation["R_star_median"].between(0, 1).all() and simulation["occupancy_median"].between(-1, 1).all()
    check(records, "simulation", "positive_control_digest", bool(sim_ok), str(sorted(simulation["endpoint"].tolist())))
    deltas = summary.copy()
    delta_ok = np.allclose(deltas["delta_c_index_vs_full"], deltas["c_index"] - deltas["full_c_index"], equal_nan=True)
    check(records, "statistics", "delta_vs_full", bool(delta_ok), "checked")
    figure_base = root / "figures" / "Figure_E14_ablation_backbone"
    check(records, "figure", "figure_files", figure_base.with_suffix(".png").exists() and figure_base.with_suffix(".pdf").exists(), f"png={figure_base.with_suffix('.png').exists()}; pdf={figure_base.with_suffix('.pdf').exists()}")
    boundary_ok, boundary_detail = figure_boundary(figure_base.with_suffix(".png"))
    check(records, "figure", "figure_boundary", boundary_ok, boundary_detail)
    for report in [
        "experiment_14_protocol_audit.md",
        "experiment_14_summary.md",
        "experiment_14_scientific_review.md",
        "top_journal_figure_design_review.md",
    ]:
        check(records, "structural", f"report_{report}", (root / report).exists(), "exists" if (root / report).exists() else "missing")

    result = pd.DataFrame(records)
    result.to_csv(root / "experiment_14_validation.csv", index=False)
    lines = [
        "# Experiment 14 Validation",
        "",
        "| Category | Check | Passed | Detail |",
        "|---|---|---:|---|",
    ]
    for row in result.itertuples():
        lines.append(f"| {row.category} | {row.check} | {'PASS' if row.passed else 'FAIL'} | {row.detail} |")
    lines.append("")
    lines.append(f"- Overall validation: {'PASS' if result['passed'].all() else 'FAIL'} ({int(result['passed'].sum())}/{len(result)})")
    (root / "experiment_14_validation.md").write_text("\n".join(lines), encoding="utf-8")
    if not result["passed"].all():
        raise SystemExit(1)


if __name__ == "__main__":
    main()

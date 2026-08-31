"""Validate Experiment 11 information-gain control outputs."""

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
    parser.add_argument("--config", default="configs/experiment_11.yaml")
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
    required_tables = [
        "combined_information_gain_scores.tsv",
        "top_rank_comparison.tsv",
        "information_gain_summary.tsv",
        "quadrant_enrichment_summary.tsv",
        "rank_correlation_bootstrap.tsv",
        "rank_gain_distribution.tsv",
    ]
    missing = [name for name in required_tables if not (tables / name).exists()]
    check(records, "structural", "required_tables", not missing, "OK" if not missing else "; ".join(missing))
    if missing:
        result = pd.DataFrame(records)
        result.to_csv(root / "experiment_11_validation.csv", index=False)
        raise SystemExit(1)

    states = pd.read_csv(tables / "combined_information_gain_scores.tsv", sep="\t")
    ranked = pd.read_csv(tables / "top_rank_comparison.tsv", sep="\t")
    summary = pd.read_csv(tables / "information_gain_summary.tsv", sep="\t")
    quadrant = pd.read_csv(tables / "quadrant_enrichment_summary.tsv", sep="\t")
    correlations = pd.read_csv(tables / "rank_correlation_bootstrap.tsv", sep="\t")
    rank_gain = pd.read_csv(tables / "rank_gain_distribution.tsv", sep="\t")
    expected = set(config["datasets"])
    check(records, "structural", "dataset_coverage", set(summary["dataset_name"]) == expected, str(sorted(summary["dataset_name"])))
    check(records, "structural", "state_dataset_coverage", set(states["dataset_name"]) == expected, str(sorted(states["dataset_name"].unique())))

    method_counts = ranked.groupby(["dataset_name", "method"]).size().to_dict()
    top_k = int(config["analysis"]["top_k"])
    counts_ok = True
    details = []
    for dataset in expected:
        stable_count = int(states[states["dataset_name"].eq(dataset) & states["stable"].astype(bool)].shape[0])
        high_count = int(states[states["dataset_name"].eq(dataset) & states["stable"].astype(bool) & states["high_confidence"].astype(bool)].shape[0])
        for method in config["methods"]:
            expected_count = min(top_k, high_count if method == "rel_obstq_mhn" and high_count else stable_count)
            observed = int(method_counts.get((dataset, method), 0))
            counts_ok = counts_ok and observed == expected_count
            details.append(f"{dataset}:{method}={observed}/{expected_count}")
    check(records, "structural", "rank_table_topk_counts", counts_ok, "; ".join(details[:8]))

    mhn_error = float(np.nanmax(np.abs(states["mhn_only_score"] - states["F_hat"])))
    occ_error = float(np.nanmax(np.abs(states["occupancy_only_score"] - states["L_v"])))
    rel_error = float(np.nanmax(np.abs(states["rel_obstq_score"] - states["R_star"])))
    check(records, "math", "score_identity", max(mhn_error, occ_error, rel_error) < 1e-12, f"max_error={max(mhn_error, occ_error, rel_error):.3g}")

    median_r_mhn_overlap = float(summary["top_R_and_MHN_fraction"].median())
    median_r_occ_overlap = float(summary["top_R_and_occupancy_fraction"].median())
    check(records, "scientific", "R_not_MHN_only_topk", median_r_mhn_overlap <= 0.50, f"median_overlap_fraction={median_r_mhn_overlap:.3f}")
    check(records, "scientific", "R_not_occupancy_only_topk", median_r_occ_overlap <= 0.65, f"median_overlap_fraction={median_r_occ_overlap:.3f}")
    median_lf = float(summary["spearman_occupancy_MHN"].median())
    median_rf = float(summary["spearman_R_MHN"].median())
    check(records, "scientific", "occupancy_tracks_inflow", median_lf >= 0.55, f"median_rho_L_F={median_lf:.3f}")
    check(records, "scientific", "R_contrasts_inflow", median_rf < 0.0, f"median_rho_R_F={median_rf:.3f}")

    corr_rows_expected = len(config["datasets"]) * 3
    ci_ok = (
        len(correlations) == corr_rows_expected
        and correlations["spearman_rho"].between(-1, 1).all()
        and (correlations["ci_low"] <= correlations["spearman_rho"]).all()
        and (correlations["ci_high"] >= correlations["spearman_rho"]).all()
    )
    check(records, "scientific", "bootstrap_rank_correlation_table", bool(ci_ok), f"rows={len(correlations)}; expected={corr_rows_expected}")

    gain_counts = rank_gain.groupby(["dataset_name", "baseline"]).size().to_dict()
    gain_count_ok = True
    gain_details = []
    for row in summary.itertuples():
        for baseline in ["mhn_only", "occupancy_only"]:
            observed = int(gain_counts.get((row.dataset_name, baseline), 0))
            expected_count = int(row.top_R_states)
            gain_count_ok = gain_count_ok and observed == expected_count
            gain_details.append(f"{row.short_name}:{baseline}={observed}/{expected_count}")
    gain_range_ok = rank_gain["percentile_rank_gain"].between(-1, 1).all()
    median_gain_mhn = float(rank_gain[rank_gain["baseline"].eq("mhn_only")]["percentile_rank_gain"].median())
    median_gain_occ = float(rank_gain[rank_gain["baseline"].eq("occupancy_only")]["percentile_rank_gain"].median())
    median_gain_occ_selected = float(rank_gain[rank_gain["baseline"].eq("occupancy_only")]["percentile_rank_gain"].median())
    check(records, "structural", "rank_gain_table_counts", gain_count_ok and gain_range_ok, "; ".join(gain_details[:8]))
    check(records, "scientific", "rank_gain_profile", median_gain_mhn > 0 and median_gain_occ_selected >= 0, f"median_gain_mhn={median_gain_mhn:.3f}; median_gain_occ_all={median_gain_occ:.3f}; median_gain_occ_selected={median_gain_occ_selected:.3f}")

    q_rows_expected = len(config["datasets"]) * 4
    q_fraction_ok = quadrant.groupby("dataset_name")["top_R_fraction"].sum().sub(1).abs().max() < 1e-9
    check(records, "structural", "quadrant_matrix_shape", len(quadrant) == q_rows_expected, f"rows={len(quadrant)}; expected={q_rows_expected}")
    check(records, "math", "quadrant_top_fraction_sums", bool(q_fraction_ok), "per-cohort top fractions sum to 1")

    figure_base = root / "figures" / "Figure_E11_information_gain_controls"
    check(records, "figure", "figure_files", figure_base.with_suffix(".png").exists() and figure_base.with_suffix(".pdf").exists(), f"png={figure_base.with_suffix('.png').exists()}; pdf={figure_base.with_suffix('.pdf').exists()}")
    boundary_ok, boundary_detail = figure_boundary(figure_base.with_suffix(".png"))
    check(records, "figure", "figure_boundary", boundary_ok, boundary_detail)
    for report in [
        "experiment_11_protocol_audit.md",
        "experiment_11_summary.md",
        "experiment_11_scientific_review.md",
        "top_journal_figure_design_review.md",
    ]:
        check(records, "structural", f"report_{report}", (root / report).exists(), "exists" if (root / report).exists() else "missing")

    result = pd.DataFrame(records)
    result.to_csv(root / "experiment_11_validation.csv", index=False)
    lines = [
        "# Experiment 11 Validation",
        "",
        "| Category | Check | Passed | Detail |",
        "|---|---|---:|---|",
    ]
    for row in result.itertuples():
        lines.append(f"| {row.category} | {row.check} | {'PASS' if row.passed else 'FAIL'} | {row.detail} |")
    lines.append("")
    lines.append(f"- Overall validation: {'PASS' if result['passed'].all() else 'FAIL'} ({int(result['passed'].sum())}/{len(result)})")
    (root / "experiment_11_validation.md").write_text("\n".join(lines), encoding="utf-8")
    if not result["passed"].all():
        raise SystemExit(1)


if __name__ == "__main__":
    main()

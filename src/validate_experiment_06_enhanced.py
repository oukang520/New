"""Validate enhanced Experiment 6 positive-control recovery stress test."""

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
    parser.add_argument("--config", default="configs/experiment_06_enhanced.yaml")
    parser.add_argument("--result-root", help="Override result root.")
    return parser.parse_args()


def check(records: list[dict], category: str, name: str, passed: bool, detail: str) -> None:
    records.append(
        {
            "category": category,
            "check": name,
            "passed": bool(passed),
            "detail": detail,
        }
    )


def figure_boundary(path: Path) -> tuple[bool, str]:
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
    ok = width >= 4500 and height >= 3200 and edge_nonwhite < 0.04
    return ok, f"size={width}x{height}; edge_nonwhite={edge_nonwhite:.4f}"


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
        "true_theta.tsv",
        "true_edge_list.tsv",
        "topology_audit.tsv",
        "truth_states.tsv",
        "truth_selection_candidate_audit.tsv",
        "lambda_cv.tsv",
        "repeat_metrics.tsv",
        "repeat_curves.tsv",
        "state_recovery_long.tsv",
        "estimated_theta_long.tsv",
        "sensitivity_metrics.tsv",
        "oracle_theta_repeat_metrics.tsv",
        "performance_summary_table.tsv",
        "repeat_file_manifest.tsv",
        "representative_state_scores.tsv",
        "experiment_06_enhanced_summary.tsv",
    ]
    missing = [name for name in required if not (tables / name).exists()]
    check(records, "structural", "required_tables", not missing, "OK" if not missing else ";".join(missing))
    if missing:
        pd.DataFrame(records).to_csv(root / "experiment_06_enhanced_validation.csv", index=False)
        raise SystemExit(1)

    theta = pd.read_csv(tables / "true_theta.tsv", sep="\t", index_col=0)
    edges = pd.read_csv(tables / "true_edge_list.tsv", sep="\t")
    topology = pd.read_csv(tables / "topology_audit.tsv", sep="\t")
    truth = pd.read_csv(tables / "truth_states.tsv", sep="\t")
    candidates = pd.read_csv(tables / "truth_selection_candidate_audit.tsv", sep="\t")
    cv = pd.read_csv(tables / "lambda_cv.tsv", sep="\t")
    metrics = pd.read_csv(tables / "repeat_metrics.tsv", sep="\t")
    curves = pd.read_csv(tables / "repeat_curves.tsv", sep="\t")
    states = pd.read_csv(tables / "state_recovery_long.tsv", sep="\t")
    estimates = pd.read_csv(tables / "estimated_theta_long.tsv", sep="\t")
    sensitivity = pd.read_csv(tables / "sensitivity_metrics.tsv", sep="\t")
    oracle = pd.read_csv(tables / "oracle_theta_repeat_metrics.tsv", sep="\t")
    performance = pd.read_csv(tables / "performance_summary_table.tsv", sep="\t")
    manifest = pd.read_csv(tables / "repeat_file_manifest.tsv", sep="\t")
    representative = pd.read_csv(tables / "representative_state_scores.tsv", sep="\t")

    p = int(config["simulation"]["events"])
    repeat_count = int(config["simulation"]["repeats"])
    n = int(config["simulation"]["samples_per_repeat"])
    matrix = theta.to_numpy(dtype=float)
    off_diag = matrix.copy()
    np.fill_diagonal(off_diag, 0.0)
    density = np.count_nonzero(off_diag) / (p * (p - 1))
    check(records, "structural", "theta_shape", theta.shape == (p, p), f"shape={theta.shape}")
    check(records, "structural", "edge_density_declared", 0.12 <= density <= 0.20, f"density={density:.4f}; label={config['simulation']['topology_label']}")
    check(records, "structural", "edge_list_consistency", len(edges) == int(np.count_nonzero(off_diag)), f"edge_rows={len(edges)}, nonzero={np.count_nonzero(off_diag)}")
    check(records, "structural", "mixed_edge_signs", (edges["log_effect"] > 0).any() and (edges["log_effect"] < 0).any(), edges["effect"].value_counts().to_dict().__str__())
    check(records, "structural", "topology_audit_matches", int(topology.iloc[0]["off_diagonal_nonzero_edges"]) == len(edges), topology.iloc[0].to_dict().__str__())

    truth_counts = truth["truth_class"].value_counts().to_dict()
    check(records, "structural", "truth_counts", truth_counts.get("bottleneck", 0) == 3 and truth_counts.get("fast", 0) == 3, truth_counts.__str__())
    check(records, "structural", "truth_selection_audited", candidates["selected"].sum() >= 6, f"selected_rows={int(candidates['selected'].sum())}, audited_rows={len(candidates)}")
    check(records, "structural", "stress_test_truth_rules", truth["selection_mode"].eq(config["truth_selection"]["mode"]).all(), truth[["state", "truth_class", "pilot_count", "event_count"]].to_dict("records").__str__())
    check(records, "structural", "independent_lambda_calibration", cv.get("source", pd.Series([""])).astype(str).str.contains("independent|manual|skip_cv").any(), cv.tail(3).to_dict("records").__str__())

    check(records, "structural", "repeat_count", len(metrics) == repeat_count and len(manifest) == repeat_count, f"metrics={len(metrics)}, manifest={len(manifest)}, expected={repeat_count}")
    manifest_ok = True
    snapshots_ok = True
    trajectories_ok = True
    for _, row in manifest.iterrows():
        for column in ["trajectory_file", "snapshot_file", "state_scores_file", "theta_file", "metrics_file"]:
            manifest_ok = manifest_ok and (root / row[column]).exists()
        snapshots_ok = snapshots_ok and int(row["snapshot_rows"]) == n
        trajectories_ok = trajectories_ok and int(row["trajectory_rows"]) >= n
    check(records, "structural", "repeat_files_exist", manifest_ok, "all manifest paths present" if manifest_ok else "missing file")
    check(records, "structural", "snapshot_rows", snapshots_ok, manifest["snapshot_rows"].describe().to_dict().__str__())
    check(records, "structural", "trajectory_rows", trajectories_ok, manifest["trajectory_rows"].describe().to_dict().__str__())
    check(records, "structural", "estimated_theta_rows", len(estimates) == repeat_count * p * p, f"rows={len(estimates)}, expected={repeat_count * p * p}")
    check(records, "structural", "curve_grid", len(curves) == repeat_count * 2 * 101, f"rows={len(curves)}, expected={repeat_count * 2 * 101}")

    eligible = states["eligible"].astype(bool)
    identity = states.loc[eligible, "L_v"] / (states.loc[eligible, "F_hat"] + float(config["state_scoring"]["epsilon"]))
    raw_error = float(np.max(np.abs(identity - states.loc[eligible, "R_raw"]))) if eligible.any() else np.inf
    medians = states.loc[eligible].groupby("repeat")["R_star"].median() if eligible.any() else pd.Series(dtype=float)
    check(records, "structural", "R_raw_identity", raw_error < 1e-9, f"max_error={raw_error:.3g}")
    check(records, "structural", "R_star_normalization", np.allclose(medians, 1.0, atol=1e-8, rtol=1e-8), f"range={medians.min() if len(medians) else 'NA'}-{medians.max() if len(medians) else 'NA'}")
    check(records, "structural", "sensitivity_thresholds_present", set(np.round(sensitivity["minimum_inflow"].astype(float), 12)) == set(np.round(np.array(config["state_scoring"]["sensitivity_minimum_inflows"], dtype=float), 12)), sensitivity["minimum_inflow"].drop_duplicates().to_list().__str__())
    check(records, "structural", "oracle_rows", len(oracle) == repeat_count, f"rows={len(oracle)}, expected={repeat_count}")
    expected_endpoints = {"Spearman", "Bottleneck ROC AUC", "Bottleneck AP", "Top-5 precision", "Recall@5"}
    performance_ok = expected_endpoints.issubset(set(performance["endpoint"])) and {
        "R_star_median",
        "R_star_q1",
        "R_star_q3",
        "R_star_mean",
        "R_star_sd",
        "R_star_min",
        "R_star_max",
        "R_star_perfect_repeat_fraction",
        "paired_delta_median",
        "paired_p_value",
    }.issubset(set(performance.columns))
    check(records, "structural", "performance_summary_table", performance_ok, performance[["endpoint", "R_star_median", "R_star_min", "R_star_max"]].to_dict("records").__str__())
    check(records, "structural", "representative_truth_coverage", set(representative["truth_class"]).issuperset({"bottleneck", "fast"}), representative["truth_class"].value_counts().to_dict().__str__())

    figure_base = root / "figures" / "Figure_E6_enhanced_bottleneck_recovery"
    figure_ok, figure_detail = figure_boundary(figure_base.with_suffix(".png"))
    check(records, "structural", "figure_files", figure_base.with_suffix(".png").exists() and figure_base.with_suffix(".pdf").exists(), f"png={figure_base.with_suffix('.png').exists()}, pdf={figure_base.with_suffix('.pdf').exists()}")
    check(records, "structural", "figure_boundary", figure_ok, figure_detail)
    for report in [
        "experiment_06_enhanced_protocol_audit.md",
        "experiment_06_enhanced_summary.md",
        "experiment_06_enhanced_scientific_review.md",
        "top_journal_figure_design_review.md",
    ]:
        check(records, "structural", f"report_{report}", (root / report).exists(), "exists" if (root / report).exists() else "missing")

    med = metrics.median(numeric_only=True)
    success = config["success"]
    check(records, "scientific_success", "spearman_success", med["spearman_R_star"] >= float(success["median_spearman_minimum"]), f"median={med['spearman_R_star']:.4f}, threshold={success['median_spearman_minimum']}")
    check(records, "scientific_success", "auc_success", med["bottleneck_auc_R_star"] >= float(success["median_bottleneck_auc_minimum"]), f"median={med['bottleneck_auc_R_star']:.4f}, threshold={success['median_bottleneck_auc_minimum']}")
    check(records, "scientific_success", "top5_success", med["top5_precision_R_star"] >= float(success["median_top5_precision_minimum"]), f"median={med['top5_precision_R_star']:.4f}, threshold={success['median_top5_precision_minimum']}")
    check(records, "scientific_success", "occupancy_improvement", med["bottleneck_auc_R_star"] > med["bottleneck_auc_occupancy"], f"R*_AUC={med['bottleneck_auc_R_star']:.4f}; occupancy_AUC={med['bottleneck_auc_occupancy']:.4f}")

    result = pd.DataFrame(records)
    result.to_csv(root / "experiment_06_enhanced_validation.csv", index=False)
    lines = [
        "# Experiment 6 Enhanced Validation",
        "",
        "| Category | Check | Passed | Detail |",
        "|---|---|---:|---|",
    ]
    for _, row in result.iterrows():
        lines.append(f"| {row['category']} | {row['check']} | {'PASS' if row['passed'] else 'FAIL'} | {row['detail']} |")
    structural = result[result["category"] == "structural"]
    scientific = result[result["category"] == "scientific_success"]
    lines.extend(
        [
            "",
            f"- Structural execution: {'PASS' if structural['passed'].all() else 'FAIL'} ({int(structural['passed'].sum())}/{len(structural)})",
            f"- Scientific success: {'PASS' if scientific['passed'].all() else 'FAIL'} ({int(scientific['passed'].sum())}/{len(scientific)})",
        ]
    )
    (root / "experiment_06_enhanced_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(result.to_string(index=False))
    if not result["passed"].all():
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Validate Experiment 7 topology and multipath robustness outputs."""

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
    parser.add_argument("--config", default="configs/experiment_07.yaml")
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


def figure_boundary(path: Path, min_width: int = 5200, min_height: int = 3600) -> tuple[bool, str]:
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
    ok = width >= min_width and height >= min_height and edge_nonwhite < 0.05
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
        "combo_manifest.tsv",
        "repeat_metrics.tsv",
        "state_recovery_long.tsv",
        "truth_states.tsv",
        "truth_selection_candidate_audit.tsv",
        "true_edge_list.tsv",
        "topology_audit.tsv",
        "combo_summary.tsv",
        "experiment_07_global_summary.tsv",
    ]
    missing = [name for name in required if not (tables / name).exists()]
    check(records, "structural", "required_tables", not missing, "OK" if not missing else ";".join(missing))
    if missing:
        pd.DataFrame(records).to_csv(root / "experiment_07_validation.csv", index=False)
        raise SystemExit(1)

    manifest = pd.read_csv(tables / "combo_manifest.tsv", sep="\t")
    metrics = pd.read_csv(tables / "repeat_metrics.tsv", sep="\t")
    states = pd.read_csv(tables / "state_recovery_long.tsv", sep="\t")
    truth = pd.read_csv(tables / "truth_states.tsv", sep="\t")
    candidates = pd.read_csv(tables / "truth_selection_candidate_audit.tsv", sep="\t")
    edges = pd.read_csv(tables / "true_edge_list.tsv", sep="\t")
    topology = pd.read_csv(tables / "topology_audit.tsv", sep="\t")
    summary = pd.read_csv(tables / "combo_summary.tsv", sep="\t")
    overall = pd.read_csv(tables / "experiment_07_global_summary.tsv", sep="\t")

    repeat_count = int(config["simulation"]["repeats"])
    expected_metric_rows = int(len(manifest) * repeat_count)
    check(records, "structural", "combo_count", len(summary) == len(manifest), f"summary={len(summary)}, manifest={len(manifest)}")
    check(records, "structural", "repeat_metric_rows", len(metrics) == expected_metric_rows, f"rows={len(metrics)}, expected={expected_metric_rows}")
    check(records, "structural", "metric_combo_coverage", set(metrics["combo_id"]) == set(manifest["combo_id"]), f"metric_combos={metrics['combo_id'].nunique()}")
    check(records, "structural", "summary_combo_coverage", set(summary["combo_id"]) == set(manifest["combo_id"]), f"summary_combos={summary['combo_id'].nunique()}")

    truth_counts = truth.groupby(["combo_id", "truth_class"]).size().unstack(fill_value=0)
    truth_ok = (
        ("bottleneck" in truth_counts.columns)
        and ("fast" in truth_counts.columns)
        and truth_counts["bottleneck"].eq(int(config["simulation"]["bottleneck_states"])).all()
        and truth_counts["fast"].eq(int(config["simulation"]["fast_states"])).all()
        and len(truth_counts) == len(manifest)
    )
    check(records, "structural", "truth_state_counts", truth_ok, truth_counts.head().to_dict().__str__())
    bottleneck_truth = truth[truth["truth_class"].eq("bottleneck")].copy()
    placement_stage = {"early_stage": "S1", "middle_stage": "S2", "late_stage": "S3"}
    placement_ok = True
    placement_details = []
    for placement, expected_stage in placement_stage.items():
        sub = bottleneck_truth[bottleneck_truth["bottleneck_placement"].eq(placement)]
        if not sub.empty:
            ok = sub["stage"].eq(expected_stage).all()
            placement_ok = placement_ok and bool(ok)
            placement_details.append(f"{placement}:{sub['stage'].value_counts().to_dict()}")
    pathway_sub = bottleneck_truth[bottleneck_truth["bottleneck_placement"].eq("pathway_specific")]
    if not pathway_sub.empty:
        pathway_ok = pathway_sub["pathway_hits"].ge(2).all()
        placement_ok = placement_ok and bool(pathway_ok)
        placement_details.append(f"pathway_specific_hits_min={int(pathway_sub['pathway_hits'].min())}")
    check(records, "structural", "bottleneck_placement_consistency", placement_ok, "; ".join(placement_details))
    selected_by_combo = candidates.groupby("combo_id")["selected"].sum()
    check(records, "structural", "truth_selection_audited", selected_by_combo.ge(6).all(), selected_by_combo.describe().to_dict().__str__())

    p = int(config["simulation"]["events"])
    expected_edges = {
        float(s): int(round(float(s) * p * (p - 1))) for s in config["simulation"]["sparsities"]
    }
    topology_ok = True
    density_details = []
    for _, row in topology.iterrows():
        target = expected_edges[float(row["sparsity"])]
        ok = int(row["off_diagonal_nonzero_edges"]) == target
        topology_ok = topology_ok and ok
        density_details.append(f"{row['combo_id']}={row['off_diagonal_nonzero_edges']}/{target}")
    check(records, "structural", "exact_sparsity_edge_counts", topology_ok, "; ".join(density_details[:8]))
    signs = edges.groupby("combo_id")["effect"].nunique()
    check(records, "structural", "edge_sign_diversity", signs.ge(1).all() and (edges["effect"].eq("inhibiting").any()), signs.describe().to_dict().__str__())
    scaffold = topology.groupby("topology")["scaffold_edges"].min().to_dict()
    check(records, "structural", "scaffold_edges_present", all(value >= 8 for value in scaffold.values()), scaffold.__str__())

    required_metric_columns = {
        "spearman_R_star",
        "bottleneck_auc_R_star",
        "top5_precision_R_star",
        "bottleneck_recall_at5_R_star",
        "stable_states",
        "stable_bottlenecks",
    }
    check(records, "structural", "metric_columns", required_metric_columns.issubset(metrics.columns), sorted(required_metric_columns - set(metrics.columns)).__str__())
    finite_spearman_fraction = float(np.isfinite(metrics["spearman_R_star"]).mean())
    if config.get("plot", {}).get("figure_focus") == "relative_dwell_time_only":
        check(
            records,
            "structural",
            "finite_metric_fraction",
            finite_spearman_fraction >= 0.80,
            f"spearman={finite_spearman_fraction:.3f}",
        )
    else:
        finite_auc_fraction = float(np.isfinite(metrics["bottleneck_auc_R_star"]).mean())
        check(
            records,
            "structural",
            "finite_metric_fraction",
            finite_auc_fraction >= 0.90 and finite_spearman_fraction >= 0.80,
            f"auc={finite_auc_fraction:.3f}; spearman={finite_spearman_fraction:.3f}",
        )

    eligible = states["eligible"].astype(bool)
    identity_error = np.nan
    if eligible.any():
        raw = states.loc[eligible, "L_v"] / (states.loc[eligible, "F_hat"] + float(config["state_scoring"]["epsilon"]))
        identity_error = float(np.nanmax(np.abs(raw - states.loc[eligible, "R_raw"])))
    check(records, "structural", "R_raw_identity", np.isfinite(identity_error) and identity_error < 1e-9, f"max_error={identity_error:.3g}")
    medians = states.loc[eligible].groupby(["combo_id", "repeat"])["R_star"].median()
    normalization_ok = bool(np.allclose(medians.dropna(), 1.0, atol=1e-8, rtol=1e-8)) if len(medians) else False
    check(records, "structural", "R_star_normalization", normalization_ok, f"median_range={medians.min() if len(medians) else 'NA'}-{medians.max() if len(medians) else 'NA'}")

    figure_base = root / "figures" / "Figure_E7_topology_robustness"
    if config.get("plot", {}).get("figure_focus") == "relative_dwell_time_only":
        figure_ok, figure_detail = figure_boundary(figure_base.with_suffix(".png"), min_width=4800, min_height=1700)
    else:
        figure_ok, figure_detail = figure_boundary(figure_base.with_suffix(".png"))
    check(records, "structural", "figure_files", figure_base.with_suffix(".png").exists() and figure_base.with_suffix(".pdf").exists(), f"png={figure_base.with_suffix('.png').exists()}, pdf={figure_base.with_suffix('.pdf').exists()}")
    check(records, "structural", "figure_boundary", figure_ok, figure_detail)
    for report in [
        "experiment_07_protocol_audit.md",
        "top_journal_figure_design_review.md",
        "experiment_07_summary.md",
        "experiment_07_scientific_review.md",
    ]:
        check(records, "structural", f"report_{report}", (root / report).exists(), "exists" if (root / report).exists() else "missing")

    global_row = overall.iloc[0]
    if config.get("plot", {}).get("figure_focus") == "relative_dwell_time_only":
        spearman_gain = float(global_row["global_spearman_R_star_median"] - global_row["global_spearman_occupancy_median"])
        check(
            records,
            "robustness_profile",
            "spearman_D_R_star",
            True,
            f"median={global_row['global_spearman_R_star_median']:.4f}; occupancy={global_row['global_spearman_occupancy_median']:.4f}",
        )
        check(
            records,
            "robustness_profile",
            "spearman_gain_vs_occupancy",
            True,
            f"gain={spearman_gain:+.4f}",
        )
    else:
        check(
            records,
            "robustness_profile",
            "spearman_D_R_star",
            True,
            f"median={global_row['global_spearman_R_star_median']:.4f}; occupancy={global_row['global_spearman_occupancy_median']:.4f}",
        )
        check(
            records,
            "robustness_profile",
            "long_dwell_auc_R_star",
            True,
            f"median={global_row['global_bottleneck_auc_R_star_median']:.4f}; occupancy={global_row['global_bottleneck_auc_occupancy_median']:.4f}",
        )
        check(
            records,
            "robustness_profile",
            "top5_long_dwell_precision",
            True,
            f"median={global_row['global_top5_precision_R_star_median']:.4f}; occupancy={global_row['global_top5_precision_occupancy_median']:.4f}",
        )
        check(
            records,
            "robustness_profile",
            "stable_truth_observability",
            True,
            f"stable_states={global_row['global_stable_states_median']:.1f}; stable_long_dwell={global_row['global_stable_bottlenecks_median']:.1f}; ineligible_truth={global_row['global_ineligible_truth_states_median']:.1f}",
        )

    result = pd.DataFrame(records)
    result.to_csv(root / "experiment_07_validation.csv", index=False)
    lines = [
        "# Experiment 7 Validation",
        "",
        "| Category | Check | Passed | Detail |",
        "|---|---|---:|---|",
    ]
    for _, row in result.iterrows():
        status = "INFO" if row["category"] == "robustness_profile" else ("PASS" if row["passed"] else "FAIL")
        lines.append(f"| {row['category']} | {row['check']} | {status} | {row['detail']} |")
    structural = result[result["category"] == "structural"]
    profile = result[result["category"] == "robustness_profile"]
    lines.extend(
        [
            "",
            f"- Structural execution: {'PASS' if structural['passed'].all() else 'FAIL'} ({int(structural['passed'].sum())}/{len(structural)})",
            f"- Robustness profile: descriptive INFO metrics ({len(profile)} rows); no empirical success thresholds are applied.",
        ]
    )
    (root / "experiment_07_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(result.to_string(index=False))
    if not structural["passed"].all():
        raise SystemExit(1)


if __name__ == "__main__":
    main()

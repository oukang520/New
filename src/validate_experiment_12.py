"""Validate Experiment 12 clinical association outputs."""

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
    parser.add_argument("--config", default="configs/experiment_12.yaml")
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
        "patient_clinical_scores.tsv",
        "clinical_endpoint_audit.tsv",
        "cox_rstar_results.tsv",
        "cindex_comparison.tsv",
        "km_curves.tsv",
        "km_group_summary.tsv",
        "stage_subgroup_cox.tsv",
        "rmst_difference.tsv",
        "landmark_survival_difference.tsv",
        "model_fit_audit.tsv",
    ]
    missing = [name for name in required if not (tables / name).exists()]
    check(records, "structural", "required_tables", not missing, "OK" if not missing else "; ".join(missing))
    if missing:
        pd.DataFrame(records).to_csv(root / "experiment_12_validation.csv", index=False)
        raise SystemExit(1)

    patients = pd.read_csv(tables / "patient_clinical_scores.tsv", sep="\t")
    audit = pd.read_csv(tables / "clinical_endpoint_audit.tsv", sep="\t")
    cox = pd.read_csv(tables / "cox_rstar_results.tsv", sep="\t")
    cindex = pd.read_csv(tables / "cindex_comparison.tsv", sep="\t")
    km = pd.read_csv(tables / "km_group_summary.tsv", sep="\t")
    subgroup = pd.read_csv(tables / "stage_subgroup_cox.tsv", sep="\t")
    rmst = pd.read_csv(tables / "rmst_difference.tsv", sep="\t")
    landmark = pd.read_csv(tables / "landmark_survival_difference.tsv", sep="\t")
    fit_audit = pd.read_csv(tables / "model_fit_audit.tsv", sep="\t")
    expected = set(config["datasets"])
    check(records, "structural", "dataset_coverage", set(audit["dataset_name"]) == expected, str(sorted(audit["dataset_name"])))
    duplicated = patients.duplicated(["dataset_name", "patient_id"]).sum()
    check(records, "clinical", "one_row_per_patient", duplicated == 0, f"duplicates={duplicated}")
    check(records, "clinical", "positive_followup", patients["followup_time_days"].gt(float(config["analysis"]["minimum_followup_days"])).all(), f"min={patients['followup_time_days'].min():.2f}")
    event_ok = audit["events"].ge(int(config["analysis"]["minimum_events_for_cox"])).all()
    check(records, "clinical", "minimum_events", bool(event_ok), "; ".join(f"{r.short_name}:{r.events}" for r in audit.itertuples()))
    r_models = cox[cox["model"].eq("R_star_adjusted")]
    cox_ok = set(r_models["dataset_name"]) == expected and r_models["hr_per_sd"].between(0.05, 20).all() and (r_models["ci_low"] < r_models["ci_high"]).all()
    check(records, "statistics", "adjusted_cox_results", bool(cox_ok), f"rows={len(r_models)}")
    cindex_models = set(config["analysis"]["cindex_models"])
    cindex_counts = cindex.groupby("dataset_name")["model"].apply(set).to_dict()
    cindex_ok = all(cindex_counts.get(dataset, set()) == cindex_models for dataset in expected) and cindex["c_index"].between(0, 1).all()
    check(records, "statistics", "cindex_models", bool(cindex_ok), str({k: sorted(v) for k, v in cindex_counts.items()}))
    km_counts = km.groupby(["dataset_name", "R_group"]).size().unstack(fill_value=0)
    km_ok = all(set(km[km["dataset_name"].eq(dataset)]["R_group"]) == {"High R*", "Low R*"} for dataset in expected)
    check(records, "statistics", "km_groups", bool(km_ok), km_counts.to_string())
    rmst_ok = set(rmst["dataset_name"]) == expected and rmst["delta_rmst_months"].notna().all() and (rmst["ci_low_months"] <= rmst["ci_high_months"]).all()
    check(records, "statistics", "rmst_difference", bool(rmst_ok), f"rows={len(rmst)}")
    landmark_expected = len(config["datasets"]) * len(config["analysis"]["landmark_years"])
    estimable_landmark = landmark[landmark["estimable"].astype(bool)]
    nonestimable_landmark = landmark[~landmark["estimable"].astype(bool)]
    landmark_ok = (
        len(landmark) == landmark_expected
        and estimable_landmark["high_survival"].between(0, 1).all()
        and estimable_landmark["low_survival"].between(0, 1).all()
        and nonestimable_landmark["delta_percentage_points"].isna().all()
    )
    check(records, "statistics", "landmark_survival", bool(landmark_ok), f"rows={len(landmark)}; expected={landmark_expected}; nonestimable={len(nonestimable_landmark)}")
    check(records, "statistics", "obsolete_quintile_removed", not (tables / "rstar_quintile_survival.tsv").exists(), "old panel C quintile table absent")
    check(records, "statistics", "obsolete_tertile_removed", not (tables / "rstar_tertile_survival.tsv").exists(), "old Low/Mid/High table absent")
    estimable_subgroups = int(subgroup["ok"].astype(bool).sum()) if "ok" in subgroup else 0
    check(records, "statistics", "subgroup_estimable", estimable_subgroups >= 4, f"estimable={estimable_subgroups}")
    failed_core = fit_audit[fit_audit["model"].isin(["R_star_adjusted"]) & ~fit_audit["ok"].astype(bool)]
    check(records, "statistics", "core_model_fit_audit", failed_core.empty, f"failed={len(failed_core)}")
    figure_base = root / "figures" / "Figure_E12_clinical_validation"
    check(records, "figure", "figure_files", figure_base.with_suffix(".png").exists() and figure_base.with_suffix(".pdf").exists(), f"png={figure_base.with_suffix('.png').exists()}; pdf={figure_base.with_suffix('.pdf').exists()}")
    boundary_ok, boundary_detail = figure_boundary(figure_base.with_suffix(".png"))
    check(records, "figure", "figure_boundary", boundary_ok, boundary_detail)
    for report in [
        "experiment_12_protocol_audit.md",
        "experiment_12_summary.md",
        "experiment_12_scientific_review.md",
        "top_journal_figure_design_review.md",
    ]:
        check(records, "structural", f"report_{report}", (root / report).exists(), "exists" if (root / report).exists() else "missing")

    result = pd.DataFrame(records)
    result.to_csv(root / "experiment_12_validation.csv", index=False)
    lines = [
        "# Experiment 12 Validation",
        "",
        "| Category | Check | Passed | Detail |",
        "|---|---|---:|---|",
    ]
    for row in result.itertuples():
        lines.append(f"| {row.category} | {row.check} | {'PASS' if row.passed else 'FAIL'} | {row.detail} |")
    lines.append("")
    lines.append(f"- Overall validation: {'PASS' if result['passed'].all() else 'FAIL'} ({int(result['passed'].sum())}/{len(result)})")
    (root / "experiment_12_validation.md").write_text("\n".join(lines), encoding="utf-8")
    if not result["passed"].all():
        raise SystemExit(1)


if __name__ == "__main__":
    main()

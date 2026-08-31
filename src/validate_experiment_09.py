"""Validate Experiment 9 observation-enrichment simulation outputs."""

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
    parser.add_argument("--config", default="configs/experiment_09.yaml")
    parser.add_argument("--result-root")
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


def figure_boundary(path: Path, min_width: int = 3600, min_height: int = 2600) -> tuple[bool, str]:
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

    required_tables = [
        "truth_states.tsv",
        "truth_selection_candidate_audit.tsv",
        "progression_pilot_distribution.tsv",
        "contrast_dwell_pilot_distribution.tsv",
        "state_recovery_long.tsv",
        "repeat_metrics.tsv",
        "repeat_curves.tsv",
        "representative_state_scores.tsv",
        "experiment_09_summary.tsv",
        "true_theta.tsv",
        "true_edge_list.tsv",
    ]
    missing = [name for name in required_tables if not (tables / name).exists()]
    check(records, "structural", "required_tables", not missing, "OK" if not missing else "; ".join(missing))
    if missing:
        pd.DataFrame(records).to_csv(root / "experiment_09_validation.csv", index=False)
        raise SystemExit(1)

    truth = pd.read_csv(tables / "truth_states.tsv", sep="\t")
    states = pd.read_csv(tables / "state_recovery_long.tsv", sep="\t")
    metrics = pd.read_csv(tables / "repeat_metrics.tsv", sep="\t")
    curves = pd.read_csv(tables / "repeat_curves.tsv", sep="\t")

    high_count = int((truth["omega_class"] == "high_observation").sum())
    low_count = int((truth["omega_class"] == "low_observation").sum())
    check(
        records,
        "protocol",
        "truth_state_counts",
        high_count == int(config["simulation"]["high_observation_states"])
        and low_count == int(config["simulation"]["low_observation_states"]),
        f"high={high_count}; low={low_count}",
    )
    omega_values = sorted(truth["omega_true"].round(6).unique().tolist())
    check(
        records,
        "protocol",
        "truth_omega_values",
        set(omega_values) == {
            round(float(config["simulation"]["high_omega"]), 6),
            round(float(config["simulation"]["low_omega"]), 6),
        },
        str(omega_values),
    )
    contrast_d = truth.loc[truth["omega_class"].eq("low_observation"), "D_true_omega_plus_dwell"].astype(float)
    check(
        records,
        "protocol",
        "contrast_dwell_assignment",
        contrast_d.eq(float(config["simulation"]["contrast_bottleneck_dwell"])).all(),
        contrast_d.tolist().__str__(),
    )

    repeats = int(config["simulation"]["repeats"])
    expected_metric_rows = repeats * 2
    check(
        records,
        "structural",
        "repeat_metric_rows",
        len(metrics) == expected_metric_rows,
        f"rows={len(metrics)}; expected={expected_metric_rows}",
    )
    scenario_set = set(metrics["scenario"])
    check(
        records,
        "structural",
        "scenario_coverage",
        scenario_set == {"omega_only", "omega_plus_dwell"},
        str(sorted(scenario_set)),
    )
    curve_ok = set(curves["method"]) == {"O_star", "occupancy"} and set(curves["scenario"]) == scenario_set
    check(records, "structural", "curve_coverage", curve_ok, f"methods={sorted(set(curves['method']))}")

    stable = states[states["eligible"].astype(bool)].copy()
    identity = np.nan
    if not stable.empty:
        reconstructed = stable["L_v"] / (stable["Lhat_progression"] + float(config["state_scoring"]["epsilon"]))
        identity = float(np.nanmax(np.abs(reconstructed - stable["O_star"])))
    check(
        records,
        "math",
        "O_star_identity",
        np.isfinite(identity) and identity < 1e-10,
        f"max_error={identity:.3g}",
    )
    r_medians = stable.dropna(subset=["R_star"]).groupby(["scenario", "repeat"])["R_star"].median()
    r_ok = bool(np.allclose(r_medians, 1.0, atol=1e-8, rtol=1e-8)) if len(r_medians) else False
    check(
        records,
        "math",
        "R_star_median_normalization",
        r_ok,
        f"range={r_medians.min() if len(r_medians) else 'NA'}-{r_medians.max() if len(r_medians) else 'NA'}",
    )

    primary = metrics[metrics["scenario"].eq("omega_only")].copy()
    finite = primary[["spearman_O_star", "high_omega_auc_O_star", "top3_precision_O_star"]].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    check(records, "performance", "finite_primary_metrics", bool(finite.all()), f"finite={int(finite.sum())}/{len(finite)}")
    med_auc = float(primary["high_omega_auc_O_star"].median())
    med_rho = float(primary["spearman_O_star"].median())
    med_top = float(primary["top3_precision_O_star"].median())
    check(
        records,
        "performance",
        "high_omega_auc",
        med_auc >= float(config["success"]["median_auc_minimum"]),
        f"median={med_auc:.4f}; threshold={config['success']['median_auc_minimum']}",
    )
    check(
        records,
        "performance",
        "spearman_omega_O_star",
        med_rho >= float(config["success"]["median_spearman_minimum"]),
        f"median={med_rho:.4f}; threshold={config['success']['median_spearman_minimum']}",
    )
    check(
        records,
        "performance",
        "top3_precision",
        med_top >= float(config["success"]["median_top3_precision_minimum"]),
        f"median={med_top:.4f}; threshold={config['success']['median_top3_precision_minimum']}",
    )
    med_auc_l = float(primary["high_omega_auc_occupancy"].median())
    check(
        records,
        "performance",
        "O_star_not_worse_than_occupancy",
        med_auc >= med_auc_l,
        f"O*={med_auc:.4f}; occupancy={med_auc_l:.4f}",
    )

    figure_base = root / "figures" / "Figure_E9_observation_enrichment"
    check(
        records,
        "figure",
        "figure_files",
        figure_base.with_suffix(".png").exists() and figure_base.with_suffix(".pdf").exists(),
        f"png={figure_base.with_suffix('.png').exists()}; pdf={figure_base.with_suffix('.pdf').exists()}",
    )
    boundary_ok, boundary_detail = figure_boundary(figure_base.with_suffix(".png"))
    check(records, "figure", "figure_boundary", boundary_ok, boundary_detail)

    for report in [
        "experiment_09_protocol_audit.md",
        "experiment_09_summary.md",
        "experiment_09_scientific_review.md",
        "top_journal_figure_design_review.md",
    ]:
        check(records, "structural", f"report_{report}", (root / report).exists(), "exists" if (root / report).exists() else "missing")

    result = pd.DataFrame(records)
    result.to_csv(root / "experiment_09_validation.csv", index=False)
    lines = [
        "# Experiment 9 Validation",
        "",
        "| Category | Check | Passed | Detail |",
        "|---|---|---:|---|",
    ]
    for _, row in result.iterrows():
        lines.append(
            f"| {row['category']} | {row['check']} | {'PASS' if row['passed'] else 'FAIL'} | {row['detail']} |"
        )
    lines.extend(
        [
            "",
            f"- Overall structural/performance validation: {'PASS' if result['passed'].all() else 'FAIL'} ({int(result['passed'].sum())}/{len(result)})",
        ]
    )
    (root / "experiment_09_validation.md").write_text("\n".join(lines), encoding="utf-8")
    if not result["passed"].all():
        raise SystemExit(1)


if __name__ == "__main__":
    main()

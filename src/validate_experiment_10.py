"""Validate Experiment 10 real-cohort main result outputs."""

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
    parser.add_argument("--config", default="configs/experiment_10.yaml")
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
        "combined_state_scores.tsv",
        "cohort_main_summary.tsv",
        "top_real_cohort_bottleneck_states.tsv",
        "top_real_cohort_observation_enriched_states.tsv",
        "top_state_explanation_table.tsv",
        "top_state_module_matrix.tsv",
    ]
    missing = [name for name in required if not (tables / name).exists()]
    check(records, "structural", "required_tables", not missing, "OK" if not missing else "; ".join(missing))
    if missing:
        pd.DataFrame(records).to_csv(root / "experiment_10_validation.csv", index=False)
        raise SystemExit(1)

    states = pd.read_csv(tables / "combined_state_scores.tsv", sep="\t")
    summary = pd.read_csv(tables / "cohort_main_summary.tsv", sep="\t")
    top = pd.read_csv(tables / "top_real_cohort_bottleneck_states.tsv", sep="\t")
    top_o = pd.read_csv(tables / "top_real_cohort_observation_enriched_states.tsv", sep="\t")
    modules = pd.read_csv(tables / "top_state_module_matrix.tsv", sep="\t")

    expected_datasets = set(config["datasets"])
    check(records, "structural", "dataset_coverage", set(summary["dataset_name"]) == expected_datasets, str(sorted(summary["dataset_name"])))
    check(records, "structural", "state_dataset_coverage", set(states["dataset_name"]) == expected_datasets, str(sorted(states["dataset_name"].unique())))
    top_counts = top.groupby("dataset_name").size().to_dict()
    check(
        records,
        "structural",
        "top_state_counts",
        all(top_counts.get(dataset, 0) == int(config["analysis"]["top_states_per_cohort"]) for dataset in expected_datasets),
        str(top_counts),
    )
    top_o_counts = top_o.groupby("dataset_name").size().to_dict()
    check(
        records,
        "structural",
        "top_O_state_counts",
        all(top_o_counts.get(dataset, 0) == int(config["analysis"]["top_observation_states_per_cohort"]) for dataset in expected_datasets),
        str(top_o_counts),
    )

    stable = states[states["stable"].astype(bool)].copy()
    r_identity = np.nan
    o_identity = np.nan
    if not stable.empty:
        raw_r = stable["L_v"] / (stable["F_hat"] + 1.0e-6)
        normalizers = stable.groupby("dataset_name")[["L_v", "F_hat"]].apply(
            lambda frame: np.median(frame["L_v"] / (frame["F_hat"] + 1.0e-6))
        )
        reconstructed = stable.apply(lambda row: raw_r.loc[row.name] / normalizers.loc[row["dataset_name"]], axis=1)
        r_identity = float(np.nanmax(np.abs(reconstructed - stable["R_star"])))
        reconstructed_o = stable["L_v"] / (stable["Lhat_progression"] + 1.0e-6)
        o_identity = float(np.nanmax(np.abs(reconstructed_o - stable["O_star"])))
    check(records, "math", "R_star_identity", np.isfinite(r_identity) and r_identity < 1e-8, f"max_error={r_identity:.3g}")
    check(records, "math", "O_star_identity", np.isfinite(o_identity) and o_identity < 1e-8, f"max_error={o_identity:.3g}")

    state_key = states.set_index(["dataset_name", "state"])
    top_source_ok = True
    top_details = []
    for row in top.itertuples():
        source = state_key.loc[(row.dataset_name, row.state)]
        ok = bool(source["high_confidence"]) and np.isclose(float(source["R_star"]), float(row.R_star))
        top_source_ok = top_source_ok and ok
        top_details.append(f"{row.short_name}:{row.rank}:{ok}")
    check(records, "scientific", "top_states_high_confidence_source", top_source_ok, "; ".join(top_details[:8]))
    top_module_fraction = float(summary["top_expected_module_fraction"].median())
    check(records, "scientific", "top_module_plausibility", top_module_fraction >= 0.75, f"median={top_module_fraction:.3f}")
    ci_fraction = float(summary["top_R_ci_above_one_fraction"].median())
    check(records, "scientific", "top_ci_above_one_profile", ci_fraction >= 0.75, f"median={ci_fraction:.3f}")
    module_rows_expected = len(config["datasets"]) * len(config["modules"])
    check(records, "structural", "module_matrix_shape", len(modules) == module_rows_expected, f"rows={len(modules)}; expected={module_rows_expected}")

    figure_base = root / "figures" / "Figure_E10_real_cohort_main_results"
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
        "experiment_10_protocol_audit.md",
        "experiment_10_summary.md",
        "experiment_10_scientific_review.md",
        "top_journal_figure_design_review.md",
    ]:
        check(records, "structural", f"report_{report}", (root / report).exists(), "exists" if (root / report).exists() else "missing")

    result = pd.DataFrame(records)
    result.to_csv(root / "experiment_10_validation.csv", index=False)
    lines = [
        "# Experiment 10 Validation",
        "",
        "| Category | Check | Passed | Detail |",
        "|---|---|---:|---|",
    ]
    for row in result.itertuples():
        lines.append(f"| {row.category} | {row.check} | {'PASS' if row.passed else 'FAIL'} | {row.detail} |")
    lines.append("")
    lines.append(f"- Overall validation: {'PASS' if result['passed'].all() else 'FAIL'} ({int(result['passed'].sum())}/{len(result)})")
    (root / "experiment_10_validation.md").write_text("\n".join(lines), encoding="utf-8")
    if not result["passed"].all():
        raise SystemExit(1)


if __name__ == "__main__":
    main()

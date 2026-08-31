"""Validate Experiment 3 cMHN and transition-interface outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Experiment 3.")
    parser.add_argument("--config", default="configs/experiment_03.yaml")
    parser.add_argument(
        "--dataset-config", default="configs/selected_experiment_datasets.yaml"
    )
    return parser.parse_args()


def validate_dataset(dataset: str, result_root: Path, panel_size: int) -> dict:
    root = result_root / dataset
    tables = root / "tables"
    problems: list[str] = []
    warnings: list[str] = []
    required = [
        "cv_likelihood.tsv",
        "theta.tsv",
        "theta_hazard_ratio.tsv",
        "event_baseline_hazard.tsv",
        "event_hazard.tsv",
        "next_event_probability.tsv",
        "genotype_summary.tsv",
        "genotype_transition.tsv",
        "transition_prob.tsv",
        "biological_sanity_checks.tsv",
    ]
    for filename in required:
        if not (tables / filename).exists():
            problems.append(f"missing_{filename}")
    if problems:
        return {
            "dataset_name": dataset,
            "samples": 0,
            "events": 0,
            "observed_genotypes": 0,
            "genotype_edges": 0,
            "state_edges": 0,
            "fit_status": "",
            "max_probability_error": "",
            "all_checks_passed": False,
            "problems": "; ".join(problems),
            "warnings": "",
        }

    metadata = json.loads((root / "model_metadata.json").read_text(encoding="utf-8"))
    theta = pd.read_csv(tables / "theta.tsv", sep="\t", index_col=0)
    genotype = pd.read_csv(tables / "genotype_summary.tsv", sep="\t")
    edges = pd.read_csv(tables / "genotype_transition.tsv", sep="\t")
    state_edges = pd.read_csv(tables / "transition_prob.tsv", sep="\t")
    cv = pd.read_csv(tables / "cv_likelihood.tsv", sep="\t")

    if theta.shape != (panel_size, panel_size):
        problems.append("theta_shape")
    if theta.columns.tolist() != theta.index.astype(str).tolist():
        problems.append("theta_event_order")
    if not np.isfinite(theta.to_numpy(dtype=float)).all():
        problems.append("theta_nonfinite")
    if int(cv["selected"].astype(str).str.lower().eq("true").sum()) != 1:
        problems.append("cv_selected_lambda_count")
    if cv["selected"].astype(str).str.lower().eq("true").iloc[[0, -1]].any():
        warnings.append("selected_lambda_at_grid_boundary")

    sums = edges.groupby("source_genotype")["probability"].sum()
    max_error = float(np.max(np.abs(sums.to_numpy() - 1.0))) if len(sums) else 0.0
    if max_error > 1e-10:
        problems.append("probability_normalization")
    if not edges["probability"].between(0, 1).all():
        problems.append("probability_bounds")
    if not np.isfinite(edges[["hazard", "log_hazard", "probability"]]).all().all():
        problems.append("edge_nonfinite")
    if (edges["hazard"] <= 0).any():
        problems.append("nonpositive_hazard")
    if not (edges["target_event_count"] == edges["source_event_count"] + 1).all():
        problems.append("not_one_event_transition")
    if edges.duplicated(["source_genotype", "event_added"]).any():
        problems.append("duplicate_genotype_edge")

    required_interface = {
        "source_state",
        "target_state",
        "source_genotype",
        "target_genotype",
        "event_added",
        "probability",
    }
    if not required_interface.issubset(state_edges.columns):
        problems.append("transition_interface_fields")
    if not state_edges["stage_transition_rule"].eq(
        "same_stage_genotype_transition"
    ).all():
        problems.append("stage_rule")
    if not state_edges["source_stage"].eq(state_edges["target_stage"]).all():
        problems.append("unexpected_stage_change")

    if int(metadata["fit_status"]) != 0:
        warnings.append(f"optimizer_status_{metadata['fit_status']}")
    if not bool(metadata["theta_finite"]):
        problems.append("metadata_theta_nonfinite")
    if abs(float(metadata["max_probability_sum_error"]) - max_error) > 1e-12:
        problems.append("metadata_probability_error")

    for suffix in [".png", ".pdf"]:
        path = root / "figures" / f"Figure_E3_MHN_interface{suffix}"
        if not path.exists():
            problems.append(f"missing_figure_{suffix[1:]}")
    png = root / "figures" / "Figure_E3_MHN_interface.png"
    if png.exists():
        with Image.open(png) as image:
            if image.size[0] < 2500 or image.size[1] < 1800:
                problems.append("low_resolution_figure")

    return {
        "dataset_name": dataset,
        "samples": int(metadata["samples"]),
        "events": int(metadata["events"]),
        "observed_genotypes": len(genotype),
        "genotype_edges": len(edges),
        "state_edges": len(state_edges),
        "fit_status": int(metadata["fit_status"]),
        "max_probability_error": max_error,
        "all_checks_passed": not problems,
        "problems": "; ".join(problems) if problems else "OK",
        "warnings": "; ".join(warnings) if warnings else "None",
    }


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    selected = yaml.safe_load(Path(args.dataset_config).read_text(encoding="utf-8"))
    datasets = [entry["dataset_name"] for entry in selected["included_datasets"]]
    result_root = Path(config["result_root"]).resolve()
    results = pd.DataFrame(
        [
            validate_dataset(dataset, result_root, int(config["panel_size"]))
            for dataset in datasets
        ]
    )
    combined_problems = []
    combined = (
        result_root
        / "combined_figures"
        / "Figure_E3_MHN_interface_three_cohorts"
    )
    for suffix in [".png", ".pdf"]:
        if not combined.with_suffix(suffix).exists():
            combined_problems.append(f"missing_combined_{suffix[1:]}")
    if combined.with_suffix(".png").exists():
        with Image.open(combined.with_suffix(".png")) as image:
            if image.size[0] < 3000 or image.size[1] < 3000:
                combined_problems.append("low_resolution_combined")

    results.to_csv(result_root / "experiment_03_validation.csv", index=False)
    lines = [
        "# Experiment 3 Validation",
        "",
        "| " + " | ".join(results.columns) + " |",
        "| " + " | ".join(["---"] * len(results.columns)) + " |",
    ]
    for _, row in results.iterrows():
        lines.append(
            "| " + " | ".join(str(row[column]) for column in results.columns) + " |"
        )
    lines.extend(
        [
            "",
            "## Combined Figure",
            "",
            (
                "- Validation: OK"
                if not combined_problems
                else "- Problems: " + "; ".join(combined_problems)
            ),
        ]
    )
    (result_root / "experiment_03_validation.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(results.to_string(index=False))
    print(
        "Combined figure:",
        "OK" if not combined_problems else "; ".join(combined_problems),
    )
    if not results["all_checks_passed"].all() or combined_problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

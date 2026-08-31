"""Validate Experiment 5 outputs and internal score identities."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_05.yaml")
    parser.add_argument(
        "--dataset-config", default="configs/selected_experiment_datasets.yaml"
    )
    return parser.parse_args()


def validate_dataset(dataset: str, config: dict, root: Path) -> dict:
    tables = root / dataset / "tables"
    required = [
        "state_scores.tsv",
        "top_bottleneck_states.tsv",
        "top_bottleneck_states_high_confidence.tsv",
        "fast_passing_states.tsv",
        "high_observation_enrichment.tsv",
        "bootstrap_R_star.tsv",
        "progression_only_sensitivity.tsv",
        "experiment_05_metrics.tsv",
    ]
    problems = [f"missing_{name}" for name in required if not (tables / name).exists()]
    figure = root / dataset / "figures" / "Figure_E5_state_scores"
    for suffix in [".png", ".pdf"]:
        if not figure.with_suffix(suffix).exists():
            problems.append(f"missing_figure_{suffix[1:]}")
    if problems:
        return {"dataset_name": dataset, "all_checks_passed": False, "problems": ";".join(problems)}

    scores = pd.read_csv(tables / "state_scores.tsv", sep="\t")
    metrics = pd.read_csv(tables / "experiment_05_metrics.tsv", sep="\t").iloc[0]
    eligible = scores["eligible_experiment5"].astype(bool)
    epsilon = float(config["thresholds"]["epsilon"])
    identity = scores["L_v"] / (scores["F_hat"] + epsilon)
    if not np.allclose(identity, scores["R_v"], rtol=1e-9, atol=1e-12):
        problems.append("R_identity_failed")
    median_r = float(scores.loc[eligible, "R_star"].median())
    if not np.isclose(median_r, 1.0, rtol=1e-8, atol=1e-8):
        problems.append("R_star_median_not_one")
    o_identity = scores["L_v"] / (scores["Lhat_progression"] + epsilon)
    if not np.allclose(o_identity, scores["O_star"], rtol=1e-9, atol=1e-12):
        problems.append("O_identity_failed")
    if not np.isfinite(scores.loc[eligible, ["R_star", "O_star"]]).all().all():
        problems.append("nonfinite_scores")
    if (scores.loc[eligible, ["R_star", "O_star"]] <= 0).any().any():
        problems.append("nonpositive_scores")
    if scores.loc[eligible, "bootstrap_valid_replicates"].min() < float(
        config["bootstrap"]["minimum_valid_fraction"]
    ) * int(config["bootstrap"]["replicates"]):
        problems.append("insufficient_bootstrap_replicates")
    with Image.open(figure.with_suffix(".png")) as image:
        if image.size[0] < 3500 or image.size[1] < 2200:
            problems.append("low_resolution_figure")
    return {
        "dataset_name": dataset,
        "eligible_states": int(eligible.sum()),
        "median_R_star": median_r,
        "median_all_state_stability": float(
            scores.loc[eligible, "stability"].median()
        ),
        "mean_top_high_confidence_stability": float(
            metrics["top_high_confidence_stability"]
        ),
        "all_checks_passed": not problems,
        "problems": "OK" if not problems else ";".join(problems),
    }


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    selected = yaml.safe_load(Path(args.dataset_config).read_text(encoding="utf-8"))
    datasets = [entry["dataset_name"] for entry in selected["included_datasets"]]
    root = Path(config["result_root"]).resolve()
    records = [validate_dataset(dataset, config, root) for dataset in datasets]
    combined = root / "combined_figures" / "Figure_E5_core_results_three_cohorts"
    combined_problems = []
    for suffix in [".png", ".pdf"]:
        if not combined.with_suffix(suffix).exists():
            combined_problems.append(f"missing_combined_{suffix[1:]}")
    if combined.with_suffix(".png").exists():
        with Image.open(combined.with_suffix(".png")) as image:
            if image.size[0] < 3000 or image.size[1] < 2400:
                combined_problems.append("low_resolution_combined")
    result = pd.DataFrame(records)
    result.to_csv(root / "experiment_05_validation.csv", index=False)
    lines = [
        "# Experiment 5 Validation",
        "",
        "| " + " | ".join(result.columns) + " |",
        "| " + " | ".join(["---"] * len(result.columns)) + " |",
    ]
    for _, row in result.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in result.columns) + " |")
    lines.extend(
        [
            "",
            "## Combined Figure",
            "",
            "- Validation: OK" if not combined_problems else "- Problems: " + "; ".join(combined_problems),
        ]
    )
    (root / "experiment_05_validation.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(result.to_string(index=False))
    print("Combined figure:", "OK" if not combined_problems else "; ".join(combined_problems))
    if not result["all_checks_passed"].all() or combined_problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

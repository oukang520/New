"""Validate Experiment 4 relative-inflow outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Experiment 4.")
    parser.add_argument("--config", default="configs/experiment_04.yaml")
    parser.add_argument(
        "--dataset-config", default="configs/selected_experiment_datasets.yaml"
    )
    return parser.parse_args()


def validate_dataset(dataset: str, config: dict, root: Path) -> dict:
    tables = root / dataset / "tables"
    problems: list[str] = []
    required_rules = [
        config["main_rule"]["name"],
        config["sensitivity_rules"]["stage_bridge"]["name"],
        config["sensitivity_rules"]["two_step"]["name"],
        config["sensitivity_rules"]["smoothed"]["name"],
    ]
    for rule in required_rules:
        if not (tables / f"inflow_table_{rule}.tsv").exists():
            problems.append(f"missing_inflow_{rule}")
        if not (tables / f"predecessor_edges_{rule}.tsv").exists():
            problems.append(f"missing_edges_{rule}")
    if problems:
        return {
            "dataset_name": dataset,
            "analysis_samples": 0,
            "states": 0,
            "main_edges": 0,
            "positive_inflow_states": 0,
            "stable_states": 0,
            "all_checks_passed": False,
            "problems": "; ".join(problems),
        }

    main_rule = config["main_rule"]["name"]
    main = pd.read_csv(tables / f"inflow_table_{main_rule}.tsv", sep="\t")
    edges = pd.read_csv(tables / f"predecessor_edges_{main_rule}.tsv", sep="\t")
    occupancy = pd.read_csv(tables / "state_occupancy_experiment4.tsv", sep="\t")
    sensitivity = pd.read_csv(tables / "inflow_rule_sensitivity.tsv", sep="\t")

    if abs(float(occupancy["L_v"].sum()) - 1.0) > 1e-10:
        problems.append("occupancy_not_normalized")
    if not np.isfinite(main[["L_v", "F_hat"]]).all().all():
        problems.append("nonfinite_inflow")
    if (main[["L_v", "F_hat"]] < 0).any().any():
        problems.append("negative_inflow")
    edge_totals = edges.groupby("target_state")["inflow_contribution"].sum()
    mapped = main.set_index("state")["F_hat"]
    common = edge_totals.index.intersection(mapped.index)
    if len(common) and not np.allclose(
        edge_totals.loc[common], mapped.loc[common], atol=1e-12, rtol=1e-10
    ):
        problems.append("edge_inflow_mismatch")
    if not (
        edges["inflow_contribution"]
        <= edges["source_L"] * edges["edge_probability"] + 1e-14
    ).all():
        problems.append("invalid_edge_contribution")
    if not edges["predecessor_type"].eq("same_stage_one_event").all():
        problems.append("main_rule_contains_nonprimary_edges")
    if not sensitivity["spearman_F_hat"].dropna().between(-1, 1).all():
        problems.append("invalid_spearman")
    if not sensitivity["top_k_overlap"].between(0, 1).all():
        problems.append("invalid_topk_overlap")

    for suffix in [".png", ".pdf"]:
        figure = root / dataset / "figures" / f"Figure_E4_relative_inflow{suffix}"
        if not figure.exists():
            problems.append(f"missing_figure_{suffix[1:]}")
    png = root / dataset / "figures" / "Figure_E4_relative_inflow.png"
    if png.exists():
        with Image.open(png) as image:
            if image.size[0] < 2500 or image.size[1] < 1800:
                problems.append("low_resolution_figure")

    return {
        "dataset_name": dataset,
        "analysis_samples": int(occupancy["N_v"].sum()),
        "states": len(main),
        "main_edges": len(edges),
        "positive_inflow_states": int((main["F_hat"] > 0).sum()),
        "stable_states": int(main["stable_for_experiment5"].astype(bool).sum()),
        "all_checks_passed": not problems,
        "problems": "; ".join(problems) if problems else "OK",
    }


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    selected = yaml.safe_load(Path(args.dataset_config).read_text(encoding="utf-8"))
    datasets = [entry["dataset_name"] for entry in selected["included_datasets"]]
    root = Path(config["result_root"]).resolve()
    result = pd.DataFrame(
        [validate_dataset(dataset, config, root) for dataset in datasets]
    )
    combined_problems = []
    combined_figures = {
        "relative_inflow": (
            "Figure_E4_relative_inflow_three_cohorts",
            (3000, 3000),
        ),
        "rule_sensitivity": (
            "Figure_E4_inflow_rule_sensitivity_three_cohorts",
            (2500, 1200),
        ),
        "dominant_edges": (
            "Figure_E4_dominant_inflow_edges_three_cohorts",
            (2500, 2400),
        ),
    }
    for label, (figure_name, minimum_size) in combined_figures.items():
        combined = root / "combined_figures" / figure_name
        for suffix in [".png", ".pdf"]:
            if not combined.with_suffix(suffix).exists():
                combined_problems.append(f"missing_{label}_{suffix[1:]}")
        if combined.with_suffix(".png").exists():
            with Image.open(combined.with_suffix(".png")) as image:
                if (
                    image.size[0] < minimum_size[0]
                    or image.size[1] < minimum_size[1]
                ):
                    combined_problems.append(f"low_resolution_{label}")
    result.to_csv(root / "experiment_04_validation.csv", index=False)
    lines = [
        "# Experiment 4 Validation",
        "",
        "| " + " | ".join(result.columns) + " |",
        "| " + " | ".join(["---"] * len(result.columns)) + " |",
    ]
    for _, row in result.iterrows():
        lines.append(
            "| " + " | ".join(str(row[column]) for column in result.columns) + " |"
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
    (root / "experiment_04_validation.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(result.to_string(index=False))
    print(
        "Combined figure:",
        "OK" if not combined_problems else "; ".join(combined_problems),
    )
    if not result["all_checks_passed"].all() or combined_problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

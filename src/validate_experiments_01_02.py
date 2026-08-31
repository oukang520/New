"""Validate Rel-ObsTQ-MHN Experiment 1 and 2 outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image


FIGURE_BASES = [
    ("experiment_01_data_preparation", "Figure_E1_QC_overview"),
    ("experiment_01_data_preparation", "Figure_E1_oncoprint"),
    ("experiment_01_data_preparation", "Figure_E1_state_sparsity"),
    ("experiment_02_stage_sensitivity", "Figure_E2_stage_sensitivity"),
    ("experiment_02_stage_sensitivity", "Figure_E2_stage_event_heatmaps"),
]


def validate_dataset(dataset: str, result_root: Path) -> dict:
    root = result_root / dataset
    exp1 = root / "experiment_01_data_preparation"
    exp2 = root / "experiment_02_stage_sensitivity"
    tables1 = exp1 / "tables"
    tables2 = exp2 / "tables"
    problems: list[str] = []

    panels = {}
    for size in [10, 15, 20, 25]:
        panel_path = tables1 / f"event_panel_p{size}.csv"
        matrix_path = tables1 / f"mhn_training_matrix_p{size}.csv"
        if not panel_path.exists() or not matrix_path.exists():
            problems.append(f"missing_p{size}_files")
            continue
        panel = pd.read_csv(panel_path)
        matrix = pd.read_csv(matrix_path)
        panels[size] = panel["event"].astype(str).tolist()
        if len(panel) != size:
            problems.append(f"panel_p{size}_size")
        if list(matrix.columns) != panels[size]:
            problems.append(f"panel_p{size}_column_order")
        if not matrix.isin([0, 1]).all().all():
            problems.append(f"panel_p{size}_nonbinary")
        if any(str(col).lower().endswith("_id") for col in matrix.columns):
            problems.append(f"panel_p{size}_contains_id")

    for smaller, larger in [(10, 15), (15, 20), (20, 25)]:
        if smaller in panels and larger in panels:
            if panels[larger][:smaller] != panels[smaller]:
                problems.append(f"panels_not_nested_{smaller}_{larger}")

    main_panel = pd.read_csv(tables1 / "main_event_panel_p15.csv")
    if (pd.to_numeric(main_panel["frequency"], errors="coerce") < 0.01).any():
        problems.append("main_panel_event_below_1pct")
    if main_panel["likely_passenger_or_size_related"].astype(str).str.lower().eq("true").any():
        problems.append("main_panel_likely_passenger")

    metadata = pd.read_csv(tables1 / "clinical_clean.csv", dtype=str)
    main_matrix = pd.read_csv(tables1 / "mhn_training_matrix_p15.csv")
    state = pd.read_csv(tables1 / "state_table_p15.csv", dtype=str)
    assignments = pd.read_csv(tables2 / "stage_scheme_assignments.csv", dtype=str)
    if not (len(metadata) == len(main_matrix) == len(state) == len(assignments)):
        problems.append("row_alignment")
    if metadata["analysis_id"].duplicated().any():
        problems.append("duplicate_analysis_id")
    if not assignments["mhn_progression_score"].eq("pending_mhn_training").all():
        problems.append("mhn_stage_not_pending")

    pending = pd.read_csv(tables2 / "rstar_stage_sensitivity_pending.csv", dtype=str)
    if not pending["status"].eq("pending_mhn_training_and_rstar").all():
        problems.append("rstar_pending_boundary")

    min_width = 2500
    min_height = 1500
    figure_count = 0
    for experiment_dir, figure_name in FIGURE_BASES:
        png = root / experiment_dir / "figures" / f"{figure_name}.png"
        pdf = root / experiment_dir / "figures" / f"{figure_name}.pdf"
        if not png.exists() or not pdf.exists():
            problems.append(f"missing_figure_{figure_name}")
            continue
        figure_count += 1
        with Image.open(png) as image:
            width, height = image.size
            if width < min_width or height < min_height:
                problems.append(f"low_resolution_{figure_name}")

    validation_path = root / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if not all(validation.values()):
        problems.append("internal_validation_failed")

    stage_metrics = pd.read_csv(tables2 / "stage_scheme_state_metrics.csv")
    return {
        "dataset_name": dataset,
        "analysis_units": len(metadata),
        "main_panel_events": len(main_panel),
        "main_panel_min_frequency": float(
            pd.to_numeric(main_panel["frequency"], errors="coerce").min()
        ),
        "valid_states_clinical": int(
            stage_metrics.loc[
                stage_metrics["scheme"].eq("clinical_stage"), "valid_states"
            ].iloc[0]
        ),
        "figures_checked": figure_count,
        "all_checks_passed": not problems,
        "problems": "; ".join(problems) if problems else "OK",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Experiments 1 and 2.")
    parser.add_argument(
        "--experiment-config", default="configs/experiments_01_02.yaml"
    )
    parser.add_argument(
        "--dataset-config", default="configs/selected_experiment_datasets.yaml"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.experiment_config).open("r", encoding="utf-8") as handle:
        experiment_config = yaml.safe_load(handle)
    with Path(args.dataset_config).open("r", encoding="utf-8") as handle:
        dataset_config = yaml.safe_load(handle)
    result_root = Path(experiment_config["experiment_root"]).resolve()
    datasets = [entry["dataset_name"] for entry in dataset_config["included_datasets"]]
    result = pd.DataFrame(
        [validate_dataset(dataset, result_root) for dataset in datasets]
    )
    combined_problems = []
    for figure_name in [
        "Figure_E1_QC_overview_three_cohorts",
        "Figure_E1_state_sparsity_three_cohorts",
    ]:
        png = result_root / "combined_figures" / f"{figure_name}.png"
        pdf = result_root / "combined_figures" / f"{figure_name}.pdf"
        if not png.exists() or not pdf.exists():
            combined_problems.append(f"missing_{figure_name}")
            continue
        with Image.open(png) as image:
            if image.size[0] < 2500 or image.size[1] < 1500:
                combined_problems.append(f"low_resolution_{figure_name}")
    result.to_csv(result_root / "experiments_01_02_validation.csv", index=False)
    lines = [
        "# Experiments 1–2 Validation",
        "",
        "| "
        + " | ".join(result.columns)
        + " |",
        "| " + " | ".join(["---"] * len(result.columns)) + " |",
    ]
    for _, row in result.iterrows():
        lines.append(
            "| " + " | ".join(str(row[column]) for column in result.columns) + " |"
        )
    (result_root / "experiments_01_02_validation.md").write_text(
        "\n".join(lines)
        + "\n\n## Combined E1 Figures\n\n"
        + (
            "- Validation: OK\n"
            if not combined_problems
            else "- Problems: " + "; ".join(combined_problems) + "\n"
        ),
        encoding="utf-8",
    )
    print(result.to_string(index=False))
    print(
        "Combined figures:",
        "OK" if not combined_problems else "; ".join(combined_problems),
    )


if __name__ == "__main__":
    main()

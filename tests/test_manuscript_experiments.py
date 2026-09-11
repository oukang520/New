"""Regression checks for the scientific contracts extracted from manuscript code."""

import ast
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from sirdwell_experiments import e03, e09, e13, e17_tables

ROOT = Path(__file__).resolve().parents[1]


def test_pooled_genotype_support_is_sum_of_stage_contributions():
    matrix = pd.DataFrame({"A": [0, 0, 0, 1], "B": [0, 0, 0, 0]})
    _, transitions = e03.build_genotype_tables(matrix, np.zeros((2, 2)))
    supported = e03.supported_edges(transitions, 99)
    pooled = supported.loc[
        (supported.source_genotype == "WT") & (supported.event_added == "A"),
        "edge_support",
    ].iloc[0]
    assert pooled == pytest.approx(0.375)
    stages = pd.DataFrame(
        {
            "stage_group": ["primary", "primary", "metastatic", "primary"],
            "genotype_signature": ["WT", "WT", "WT", "A"],
        }
    )
    expanded = e03.expand_state_transitions(
        transitions, stages, ["primary", "metastatic"]
    )
    selected = expanded.loc[
        (expanded.source_genotype == "WT") & (expanded.event_added == "A")
    ]
    np.testing.assert_allclose(
        sorted(selected.source_state_count / 4 * selected.probability), [0.125, 0.25]
    )
    assert pooled == pytest.approx(
        (selected.source_state_count / 4 * selected.probability).sum()
    )


def test_observation_enrichment_renormalizes_weighted_mass():
    observed = e09.weighted_observation_distribution(
        np.array([0.25, 0.75]), np.array([3.0, 1.0])
    )
    np.testing.assert_allclose(observed, [0.5, 0.5])


def test_split_recomputes_occupancy_and_keeps_locked_inflow():
    states = pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p2", "p3"],
            "canonical_state": ["primary::A", "primary::A", "primary::B", "primary::B"],
        }
    )
    scores = pd.DataFrame(
        {
            "canonical_state": ["primary::A", "primary::B"],
            "eligible_experiment5": [True, True],
            "F_hat": [0.1, 0.4],
        }
    )
    config = {
        "analysis": {"minimum_state_count": 1, "minimum_inflow": 1e-8, "epsilon": 1e-6}
    }
    result = e13.compute_split_scores(states, scores, {"p1", "p2"}, config)
    np.testing.assert_allclose(result.L_split, [2 / 3, 1 / 3])
    np.testing.assert_allclose(result.F_hat, scores.F_hat)
    assert result.loc[result.eligible_split, "R_star_split"].median() == pytest.approx(
        1
    )


def test_routes_retain_target_ranking_and_native_states():
    scores = pd.DataFrame(
        {
            "state": ["primary::WT", "primary::A"],
            "N_v": [10, 5],
            "R_star": [0.5, 2.0],
            "log2_R_star": [-1.0, 1.0],
            "eligible_relobstq": [False, True],
            "dominant_predecessor": [None, "primary::WT"],
        }
    )
    routes = e17_tables.route_table(scores, top_paths=1)
    assert set(routes.target_state) == {"primary::A"}
    assert routes.iloc[-1].state == "primary::A"
    assert routes.iloc[-1].R_star == 2


def test_selected_inventory_and_parameter_files_are_complete():
    with (ROOT / "docs/FIGURE_EXPERIMENT_MAP.tsv").open(
        encoding="utf-8", newline=""
    ) as f:
        records = list(csv.DictReader(f, delimiter="\t"))
    assert len(records) == 43
    assert len({r["experiment"] for r in records}) == 14
    assert sum(Path(r["artifact"]).name.startswith("Figure_") for r in records) == 42
    for row in records:
        for config in row["parameters"].split(";"):
            assert isinstance(
                yaml.safe_load((ROOT / config).read_text(encoding="utf-8")), dict
            )


def test_no_rendering_imports_or_file_writes_in_executable_code():
    forbidden = {"matplotlib", "seaborn", "plotly", "bokeh", "figure_style"}
    for folder in [ROOT / "src", ROOT / "experiments"]:
        for file in folder.rglob("*.py"):
            tree = ast.parse(file.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    assert not any(
                        a.name.split(".")[0] in forbidden for a in node.names
                    ), file
                elif isinstance(node, ast.ImportFrom):
                    assert (node.module or "").split(".")[0] not in forbidden, file
                elif isinstance(node, ast.Call) and isinstance(
                    node.func, ast.Attribute
                ):
                    assert node.func.attr not in {"savefig", "imshow", "subplots"}, file

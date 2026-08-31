from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from relobstq_mhn import (  # noqa: E402
    BootstrapConfig,
    ScoreThresholds,
    SimulationConfig,
    aggregate_inflow,
    bootstrap_relative_dwell,
    build_experiment_ready_tables,
    build_dominant_predecessor_path,
    compute_relative_dwell,
    create_sparse_theta,
    event_added,
    genotype_signature,
    genotype_vector,
    implant_dwell_truth,
    same_stage_one_step_edges,
    simulate_cohort_with_audit,
    standardize_stage_group,
)


def synthetic_occupancy() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"state": "primary::WT", "stage": "primary", "genotype": "WT", "event_count": 0, "N_v": 50, "L_v": 0.5},
            {"state": "primary::A", "stage": "primary", "genotype": "A", "event_count": 1, "N_v": 30, "L_v": 0.3},
            {"state": "primary::B", "stage": "primary", "genotype": "B", "event_count": 1, "N_v": 10, "L_v": 0.1},
            {"state": "primary::A+B", "stage": "primary", "genotype": "A+B", "event_count": 2, "N_v": 10, "L_v": 0.1},
        ]
    )


def test_genotype_roundtrip() -> None:
    events = ["A", "B", "C"]
    vector = genotype_vector("C+A", events)
    assert vector.tolist() == [1, 0, 1]
    assert genotype_signature(vector, events) == "A+C"


def test_inflow_and_relative_dwell_identity() -> None:
    events = ["A", "B"]

    def probabilities(genotype: str) -> dict[str, float]:
        if genotype == "WT":
            return {"A": 0.5, "B": 0.5}
        if genotype == "A":
            return {"B": 1.0}
        if genotype == "B":
            return {"A": 1.0}
        return {}

    occupancy = synthetic_occupancy()
    edges = same_stage_one_step_edges(occupancy, events, probabilities)
    inflow = aggregate_inflow(occupancy, edges, minimum_state_count=1, minimum_inflow=1e-12)
    by_state = inflow.set_index("state")

    assert np.isclose(by_state.loc["primary::A", "F_hat"], 0.25)
    assert np.isclose(by_state.loc["primary::B", "F_hat"], 0.25)
    assert np.isclose(by_state.loc["primary::A+B", "F_hat"], 0.4)

    scores, normalizer = compute_relative_dwell(
        inflow,
        ScoreThresholds(minimum_state_count=1, minimum_inflow=1e-12, high_confidence_state_count=1, epsilon=1e-12),
    )
    scored = scores.set_index("state")
    assert np.isclose(normalizer, 0.4)
    assert np.isclose(scored.loc["primary::A", "R_star"], 3.0)
    assert np.isclose(scored.loc["primary::B", "R_star"], 1.0)
    assert np.isclose(scored.loc["primary::A+B", "R_star"], 0.625)


def test_bootstrap_and_topology_helpers() -> None:
    occupancy = synthetic_occupancy()
    edges = pd.DataFrame(
        [
            {"source_state": "primary::WT", "target_state": "primary::A", "edge_probability": 0.5},
            {"source_state": "primary::A", "target_state": "primary::A+B", "edge_probability": 1.0},
        ]
    )
    summary, long = bootstrap_relative_dwell(
        occupancy[["state", "N_v"]],
        edges,
        thresholds=ScoreThresholds(minimum_state_count=1, minimum_inflow=1e-12, high_confidence_state_count=1),
        bootstrap=BootstrapConfig(replicates=10, top_k=2, random_seed=7),
    )
    assert set(summary["state"]) == set(occupancy["state"])
    assert summary["R_star_bootstrap_median"].notna().any()
    assert not long.empty

    score_by_state = {
        "primary::A+B": {"dominant_predecessor": "primary::A"},
        "primary::A": {"dominant_predecessor": "primary::WT"},
        "primary::WT": {"dominant_predecessor": ""},
    }
    assert build_dominant_predecessor_path("primary::A+B", score_by_state) == [
        "primary::WT",
        "primary::A",
        "primary::A+B",
    ]
    assert event_added("primary::A", "primary::A+B") == "B"


def test_data_processing_core_tables() -> None:
    metadata = pd.DataFrame(
        {
            "analysis_id": ["S1", "S2", "S3", "S4"],
            "stage_group": ["Stage I", "Stage IV", "primary", "unknown"],
            "patient_id": ["P1", "P2", "P3", "P4"],
        }
    )
    mutations = pd.DataFrame(
        {
            "analysis_id": ["S1", "S2", "S2", "S3", "S4"],
            "gene": ["TP53", "KRAS", "TP53", "EGFR", "KRAS"],
        }
    )
    tables = build_experiment_ready_tables(
        metadata,
        mutations,
        max_events=3,
        min_event_frequency=0.0,
        min_state_count=1,
    )
    assert set(tables["events"]) == {"TP53", "KRAS", "EGFR"}
    assert tables["event_matrix"].shape == (4, 4)
    assert "state_id" in tables["state_table"]
    assert standardize_stage_group("AJCC stage IV") == "metastatic"


def test_simulation_core_outputs() -> None:
    events = ["E1", "E2", "E3"]
    theta = create_sparse_theta(
        events,
        sparsity=0.0,
        seed=3,
        forced_edges={(1, 0): 1.0, (2, 1): 1.0},
    )
    trajectories, snapshots = simulate_cohort_with_audit(
        theta,
        events,
        {0: 2.0},
        config=SimulationConfig(samples=30, maximum_time=3.0, maximum_events=2, random_seed=9),
    )
    assert len(snapshots) == 30
    assert {"state", "mask", "D_true", "E1", "E2", "E3"}.issubset(snapshots.columns)
    assert {"time_start", "time_end", "event_added"}.issubset(trajectories.columns)
    dwell, truth = implant_dwell_truth(
        snapshots,
        bottleneck_states=1,
        fast_states=1,
        min_bottleneck_count=1,
    )
    assert len(dwell) >= 1
    assert set(truth["truth_class"]).issubset({"bottleneck", "fast"})

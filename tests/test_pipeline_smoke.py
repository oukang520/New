from pathlib import Path

import pandas as pd

from src.build_event_matrix import build_event_matrix
from src.build_state_table import build_state_table


def test_event_matrix_and_state_table_smoke(tmp_path: Path):
    mutations = pd.DataFrame(
        {
            "patient_id": ["P1", "P1", "P2", "P3"],
            "sample_id": ["S1", "S1", "S2", "S3"],
            "gene": ["KRAS", "TP53", "KRAS", "SMAD4"],
            "alteration_type": ["Missense_Mutation", "Nonsense_Mutation", "Missense_Mutation", "Frame_Shift_Del"],
            "alteration_binary": [1, 1, 1, 1],
        }
    )
    clinical = pd.DataFrame(
        {
            "patient_id": ["P1", "P2", "P3", "P4"],
            "sample_id": ["S1", "S2", "S3", "S4"],
            "cohort": ["toy"] * 4,
            "stage_group": ["early", "metastatic", "local_advanced", "unknown"],
        }
    )

    matrix, frequency, metadata, qc = build_event_matrix(
        mutations=mutations,
        clinical=clinical,
        id_level="patient",
        min_frequency=0.0,
        top_k_events=10,
        include_cna=False,
    )

    assert set(["patient_id", "KRAS", "TP53", "SMAD4"]).issubset(matrix.columns)
    assert len(matrix) == 4
    assert qc["sample_count"] == 3 or qc["sample_count"] == 4
    assert not frequency.empty

    state_table, occupancy, state_qc = build_state_table(
        event_matrix=matrix,
        metadata=metadata,
        id_level="patient",
        min_state_count=1,
    )

    assert "genotype_signature" in state_table.columns
    assert "state_id" in state_table.columns
    assert state_qc["sample_count"] == len(state_table)
    assert not occupancy.empty

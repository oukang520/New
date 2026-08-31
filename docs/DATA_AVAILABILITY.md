
# Data availability

The cross-sectional analysis uses AACR Project GENIE v18.0-public-derived LUAD,
COAD, and IDC cohorts. Download GENIE from its authorized provider and create
the files `mhn_training_matrix.csv`, `mhn_row_index_map.csv`, and
`state_table.csv` under `processed/experiment_ready/COHORT/`.

Longitudinal validation uses GLASS, colorectal primary-metastasis triplets, and
MNM-WashU. Provider-specific raw parsing must produce `sample_metadata.tsv` and
`event_matrix.tsv`; the included patient-grouped cross-fitting workflow then
generates leakage-controlled pair predictions. See `docs/RESULT_CONTRACTS.md`.
Raw patient-level records are not redistributed.

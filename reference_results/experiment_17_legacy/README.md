# Selected E17 Reference Results

This directory freezes the aggregate outputs of the project-selected original
Experiment 17 analysis. The source runner is
`experiments/run_longitudinal.py`; the configuration is
`configs/longitudinal.yaml`.

The six TSV tables are byte-identical to the corresponding files in the
historical workspace result directory. The three `fit_metadata_*.json` files
record `frequency_cooccurrence_backbone`. Therefore, these results must not be
described as patient-grouped out-of-fold official-cMHN refits.

Patient-level predictions and provider data are not redistributed. See
`docs/DATA_AVAILABILITY.md` for the input contract and
`E17_PROVENANCE_AND_COMPARISON.md` for the selection decision and limitations.

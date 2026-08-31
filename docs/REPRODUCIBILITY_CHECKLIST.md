
# Reproducibility Checklist

- Python: use 3.11 or 3.12 for the full MHN dependency.
- Main dependency: `mhn==1.2.3`.
- Random seeds: defined in experiment configuration files where applicable.
- Data: not bundled; download from original public providers.
- Minimal tests:
  - `python -m pytest tests/test_relobstq_core.py tests/test_pipeline_smoke.py -q`
  - `python -m pytest src/relobstq_mhn/tests/test_relobstq_core.py -q`
- Main cross-sectional cohorts: AACR_LUAD, AACR_COAD, AACR_IDC.
- Excluded from current main chain: PACA-CA and other low-feasibility datasets
  listed in `configs/selected_experiment_datasets.yaml`.

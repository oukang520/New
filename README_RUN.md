# Rel-ObsTQ-MHN Reproduction Guide

This guide describes the current reproducible analysis chain for the
Rel-ObsTQ-MHN project. It supersedes earlier first-pass data-audit notes.

## Scope

The current main cross-sectional experiment chain uses three AACR Project GENIE
v18.0-public-derived cancer-type cohorts:

- `AACR_LUAD`
- `AACR_COAD`
- `AACR_IDC`

`PACA-CA` and other previously screened datasets are excluded from the current
main experiment chain because their usable state/sample scale or progression
state diversity is insufficient for stable Rel-ObsTQ-MHN real-cohort analysis.

## Environment

Use Python 3.11 or 3.12 for the full MHN-backed workflow.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The official MHN dependency is:

```text
mhn==1.2.3; python_version < "3.13"
```

Current local Python 3.13 can run the light package tests, but it is not the
recommended environment for reproducing the full official `mhn` fitting path.

## Data Contract

Raw downloaded datasets should be placed under `Data/` or the paths specified in
the experiment configuration files. Raw patient-level tables and generated
outputs are not redistributed in the public code release.

Generated files are written to:

- `processed/`
- `reports/`
- `results/`
- `logs/`

## Build Experiment-Ready Datasets

```powershell
python src/build_experiment_ready_datasets.py
python src/validate_experiment_ready.py
```

Expected per-dataset outputs under `processed/experiment_ready/{DATASET}/`:

- `analysis_metadata.csv`
- `mutations_long.csv`
- `event_matrix.csv`
- `mhn_training_matrix.csv`
- `mhn_row_index_map.csv`
- `event_frequency.csv`
- `state_table.csv`
- `state_occupancy.csv`
- `dataset_manifest.json`
- `qc_report.md`

## Minimal Tests

Run these two commands separately because the root and package test directories
contain similarly named test modules.

```powershell
python -m pytest tests/test_relobstq_core.py tests/test_pipeline_smoke.py -q
python -m pytest src/relobstq_mhn/tests/test_relobstq_core.py -q
```

## Main Experiment Chain

Run from the repository root after the required public datasets have been placed
under their configured paths.

```powershell
python src/run_experiments_01_02.py
python src/validate_experiments_01_02.py

python src/run_experiment_03.py
python src/run_experiment_04.py
python src/run_experiment_05.py
python src/run_experiment_06_enhanced.py
python src/run_experiment_06_dwell_gradient.py
python src/run_experiment_07.py
python src/run_experiment_08.py
python src/run_experiment_09.py
python src/run_experiment_10.py
python src/run_experiment_11.py
python src/run_experiment_12.py
python src/run_experiment_13.py
python src/run_experiment_14.py
python src/run_experiment_15.py
python src/run_experiment_16.py
python src/run_experiment_17_longitudinal_public.py
python src/run_experiment_17_longitudinal_extension.py
```

## Method Package

Reusable method code is under `src/relobstq_mhn/`:

- `core/states.py`
- `core/transitions.py`
- `core/scoring.py`
- `core/bootstrap.py`
- `core/topology.py`
- `core/pipeline.py`
- `data/processing.py`
- `simulation/generator.py`

## Manuscript Planning Artifacts

The following files summarize the current code-result-claim chain:

- `MANUSCRIPT_MASTER_PLAN.md`
- `RESULTS_MASTER_TABLE.md`
- `CLAIM_EVIDENCE_MATRIX.md`
- `METHOD_CODE_MAPPING.md`
- `FIGURE_PLAN.md`
- `MANUSCRIPT_TODO.md`

Regenerate them with:

```powershell
python src/build_manuscript_master_plan.py
```

## Public Release Package

Build the clean public-code folder with:

```powershell
python src/build_public_release_package.py
```

The generated folder excludes raw data, processed data, result directories,
figures, logs, caches and binary documents.

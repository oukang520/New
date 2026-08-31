
# Rel-ObsTQ-MHN Reproducible Code

This repository contains the reproducible method and experiment code for the
Rel-ObsTQ-MHN project. The method estimates a state-level relative dwell/stasis
index from cross-sectional genomic cohorts by contrasting observed state
occupancy with progression-expected inflow derived from a Mutual Hazard Network
(MHN) transition backbone.

Prepared for public scientific-code release on 2026-08-31.

## Core Method

For a stage-specific genotype state `v`, the primary score is:

```text
L_v      = N_v / N
F_hat_v  = sum_u L_u * P(u -> v | theta)
R_raw_v  = L_v / (F_hat_v + epsilon)
R*_v     = R_raw_v / median(R_raw among eligible states)
```

`R* > 1` indicates a state that is more frequently observed than expected from
its MHN-derived progression inflow, and is interpreted as a relative dwell/stasis
proxy. It is not an absolute calendar-time estimate.

## Repository Layout

```text
src/relobstq_mhn/     Reusable method package: states, transitions, scoring,
                      bootstrap, topology, data processing and simulation.
src/run_experiment_*  Experiment runners E1-E17.
src/validate_*        Validation and QC scripts for experiment outputs.
configs/              Experiment configuration files.
tests/                Smoke and core unit tests.
docs/                 Method API, figure-style notes and manuscript planning
                      support documents.
external/mhn/         Source note for the official MHN dependency.
```

Raw patient-level data, processed datasets, generated figures and large result
directories are intentionally not included in this public code package.

## Environment

Use Python 3.11 or 3.12 for the full MHN-backed workflow. The official `mhn`
package is pinned as:

```text
mhn==1.2.3; python_version < "3.13"
```

Quick setup:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Conda-style setup:

```powershell
conda env create -f environment.yml
conda activate relobstq-mhn
```

## Data Availability

The main cross-sectional experiments use AACR Project GENIE v18.0-public
subcohorts (LUAD, COAD and IDC) as configured in
`configs/selected_experiment_datasets.yaml`. Public longitudinal validation uses
the cohorts documented in the E17 configuration and scripts. Users must download
public datasets from their original providers and place them under the expected
local paths before running the full pipeline.

No raw AACR/GENIE, ICGC, cBioPortal or other patient-level tables are redistributed
here.

## Minimal Verification

The two test commands should be run separately because the root tests and package
tests intentionally contain similarly named test modules:

```powershell
python -m pytest tests/test_relobstq_core.py tests/test_pipeline_smoke.py -q
python -m pytest src/relobstq_mhn/tests/test_relobstq_core.py -q
```

## Full Reproduction Order

The scripts are designed to be run from the repository root after data have been
placed under the configured paths.

```powershell
python src/build_experiment_ready_datasets.py
python src/validate_experiment_ready.py

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

See `docs/CODE_AVAILABILITY.md`, `docs/REPRODUCIBILITY_CHECKLIST.md` and
`docs/METHOD_CODE_MAPPING.md` for more detail.

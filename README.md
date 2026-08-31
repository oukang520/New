
# Rel-ObsTQ-MHN

Reproducible code for estimating a state-level **relative dwell/stasis proxy**
from cross-sectional genomic cohorts using observed occupancy and an
MHN-derived evolutionary inflow backbone.

Prepared for scientific code release on 2026-08-31.

## Method

For state `v`:

```text
L_v      = N_v / N
F_hat_v  = sum_u L_u * P(u -> v | theta)
R_raw_v  = L_v / (F_hat_v + epsilon)
R*_v     = R_raw_v / median(R_raw among eligible states)
```

`R*` is a relative cross-sectional dwell/accumulation proxy. It is not an
estimate of absolute calendar time.

## Architecture

```text
src/relobstq_mhn/
  core/          state representation, cMHN adapter, inflow, R*/O*, bootstrap
  data/          experiment-ready mutation and state-table construction
  simulation/    audited cMHN-like trajectory generator
  evaluation/    shared metrics and bootstrap intervals
  workflows/     integrated, plotting-free scientific workflows
  io/            configuration and hashed result persistence
experiments/     thin commands; no duplicated method implementation
configs/         five workflow-level configurations
examples/        optional plotting from completed TSV results only
tests/           method and workflow tests
docs/            schemas, experiment mapping, data/code availability
```

The historical `run_experiment_01...17.py` development scripts are deliberately
not included. Their repeated IO, statistics and plotting were consolidated into
the modules above. The E1-E17 evidence mapping is preserved in
`docs/EXPERIMENT_MAPPING.md`.

## Environment

Use Python 3.11 or 3.12 for the official `mhn==1.2.3` backend.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[mhn,test]"
```

## Reproduction

Place provider-authorized data under the paths described in
`docs/DATA_AVAILABILITY.md`, then run:

```powershell
python experiments/prepare_cross_sectional.py
python experiments/run_cross_sectional.py
python experiments/run_secondary.py
python experiments/run_simulation.py
python experiments/prepare_longitudinal.py
python experiments/run_longitudinal.py
```

All scientific outputs are TSV/JSON files under `outputs/`; every output folder
contains SHA-256 hashes. Plotting is optional and separated:

```powershell
python -m pip install -e ".[figures]"
python examples/plot_core_results.py simulation outputs/simulation_dwell_gradient outputs/example_dwell_gradient
```

## Verification

```powershell
python -m pytest -q
```

Raw patient-level data and generated results are not redistributed.

## Provenance boundary

The original frozen audit commit `e4215608...` omitted the declared
`relobstq_mhn.data` package. The Phase-0 successor restores the exact
manifest-matching files and adds hashed input/runtime metadata. Historical
E1-E17 results are not automatically results of this refactored package; see
`RESULT_PROVENANCE_MATRIX.tsv` and `FINAL_REPRODUCIBILITY_REPORT.md` before
using a numerical value in a manuscript.

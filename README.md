
# Rel-ObsTQ-MHN

Reproducible code for estimating a state-level **relative dwell/stasis proxy**
from cross-sectional genomic cohorts using observed occupancy and an
MHN-derived evolutionary inflow backbone.

Final manuscript evidence release audited on 2026-09-02.

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
experiments/     core commands plus the exact legacy E17 longitudinal runner
configs/         workflow and shared figure-style configurations
examples/        optional plotting from completed TSV results only
tests/           method and workflow tests
docs/            schemas, experiment mapping, data/code availability
```

Repeated method code from the historical development scripts was consolidated
into the modules above. By explicit project decision, the exact legacy E17
runner is retained as `experiments/run_longitudinal.py` because its original
selection, scoring, bootstrap and reporting contract is the selected primary
longitudinal analysis. Its configuration explicitly freezes the realized
historical frequency/co-occurrence fallback backend so installing optional MHN
cannot silently change the selected result. The E1-E17 evidence mapping is preserved in
`docs/EXPERIMENT_MAPPING.md`.

## Environment

Use Python 3.11 or 3.12 for the official `mhn==1.2.3` backend.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[all]"
```

## Reproduction

Place provider-authorized data under the paths described in
`docs/DATA_AVAILABILITY.md`, then run:

```powershell
python experiments/prepare_cross_sectional.py
python experiments/run_cross_sectional.py
python experiments/run_secondary.py
python experiments/run_simulation.py
python experiments/run_topology_robustness.py
python experiments/run_longitudinal.py
python experiments/freeze_final_evidence.py
```

Core refactored workflows write hashed TSV/JSON outputs under `outputs/`.
The verified final current-code evidence is committed under
`reference_results/final_manuscript_evidence/`; its 99 files are indexed by
`FINAL_EVIDENCE_INDEX.tsv`.
The selected legacy E17 runner writes its original tables and figures under
`results/experiment_17_longitudinal_public/`; aggregate reference outputs are
also frozen under `reference_results/experiment_17_legacy/`.
The selected E17 output is an external longitudinal consistency analysis, not
a fully out-of-fold official-cMHN validation. Outcome-independent cohort rules,
weak/non-primary cohort results and the strict official-cMHN sensitivity are
retained in the adjacent Experiment 17 reference-result directories.

Optional plotting for the refactored workflows remains separated:

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
E1-E16 outputs are not automatically results of this refactored package. The
manuscript-critical E3-E7 and E10-E16 evidence has now been regenerated and
classified explicitly; E17 uses the restored selected implementation plus
transparent challenge and strict sensitivity analyses. Use
`RESULTS_MASTER_TABLE.md`, `RESULT_PROVENANCE_MATRIX.tsv`, and
`FINAL_REPRODUCIBILITY_REPORT.md` as the sole manuscript evidence controls.

Phase-0 v3 status: **PASS - ready for manuscript writing**, subject to the
permanent interpretation limits in `PHASE0_V3_FINAL_AUDIT.md`.

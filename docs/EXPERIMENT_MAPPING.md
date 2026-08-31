# Experiment-to-workflow mapping

The manuscript experiment numbers describe scientific questions. They are not
software modules. The public code groups them by shared computation so the same
formula is implemented once.

| Original evidence | Public workflow | Primary outputs |
|---|---|---|
| E1 preparation, E3-E5 core | `experiments/prepare_cross_sectional.py`, `experiments/run_cross_sectional.py` | fixed p15 input QC, theta, one-step edges, occupancy, inflow, R* and top states |
| E6 and continuous-gradient supplement | `experiments/run_simulation.py` | truth states, repeat scores, ordering/calibration metrics |
| E7 | reusable simulation functions only; no frozen canonical E7 runner/config | legacy robustness tables only |
| E8 | `workflows/secondary.py::module_enrichment`; not called by the current runner | function available, historical result legacy-only |
| E10 | no current canonical cohort-summary contract | historical result legacy-only |
| E11 | `experiments/run_secondary.py` | information-gain summary |
| E9 | `core/scoring.py::compute_observation_enrichment`; no current experiment contract | function available, historical result legacy-only |
| E12 | `workflows/secondary.py::clinical_association`; no current runner | function available, historical result legacy-only |
| E13 | `workflows/replication.py::compare_score_tables`; no split/refit runner | comparison function only |
| E14 | `workflows/controls.py::backbone_ablation`; not called by current runner | function available, historical result legacy-only |
| E15 | `experiments/run_secondary.py` | matched-decoy and inflow-pairing falsification tables |
| E16 | `workflows/topology.py::topology_route_table` | six table-form evolutionary routes with R* values |
| E17 | `experiments/prepare_longitudinal.py`, then `experiments/run_longitudinal.py` | patient-grouped out-of-fold predictions and cohort metric table |

The old plotting-heavy `run_experiment_*.py` scripts are intentionally absent
from the public package. They were development scripts and duplicated data IO,
statistics, validation and rendering. This refactor retains the numerical
method implementation. It does **not** make every historical E1-E17 number a
reproduced output of v0.2.0. Exact status is maintained in
`RESULT_PROVENANCE_MATRIX.tsv`; missing canonical runners are explicit rather
than represented by callable functions alone.

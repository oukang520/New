# Experiment-to-workflow mapping

The manuscript experiment numbers describe scientific questions. They are not
software modules. The public code groups them by shared computation so the same
formula is implemented once.

| Original evidence | Public workflow | Primary outputs |
|---|---|---|
| E1-E5 | `experiments/run_cross_sectional.py` | input QC, theta, one-step edges, occupancy, inflow, R* and top states |
| E6 and continuous-gradient supplement | `experiments/run_simulation.py` | truth states, repeat scores, ordering/calibration metrics |
| E7 | simulation configuration variants using `workflows/simulation.py` | robustness metric tables |
| E8, E10, E11 | `workflows/secondary.py` and `experiments/run_secondary.py` | biological summaries and information-gain tables |
| E9 | `core/scoring.py::compute_observation_enrichment` | O* state table |
| E12 | `workflows/secondary.py::clinical_association` | Cox coefficients and C-index audit |
| E13 | `workflows/replication.py::compare_score_tables` | split-score correlation and top-state overlap |
| E14-E15 | `workflows/controls.py` | backbone ablation, matched-decoy and inflow-pairing falsification tables |
| E16 | `workflows/topology.py::topology_route_table` | six table-form evolutionary routes with R* values |
| E17 | `experiments/prepare_longitudinal.py`, then `experiments/run_longitudinal.py` | patient-grouped out-of-fold predictions and cohort metric table |

The old plotting-heavy `run_experiment_*.py` scripts are intentionally absent
from the public package. They were development scripts and duplicated data IO,
statistics, validation and rendering. This refactor retains the numerical
method and evidence chain while providing one implementation per operation.

# Method-Code Mapping

| method component | implementation | canonical entry point | frozen output |
|---|---|---|---|
| fixed p15 cohort construction | `src/relobstq_mhn/data/processing.py`, `workflows/preparation.py` | `experiments/prepare_cross_sectional.py` | `P15_INPUT_AUDIT.tsv` |
| state representation and eligibility | `src/relobstq_mhn/core/states.py` | `experiments/run_cross_sectional.py` | `tables/state_occupancy.tsv`, `state_scores.tsv` |
| official cMHN adapter and CV | `src/relobstq_mhn/core/mhn.py` | `experiments/run_cross_sectional.py` | `theta.tsv`, `cv_scores.tsv`, `fit_metadata.json` |
| conditional next-event probabilities | `src/relobstq_mhn/core/mhn.py` | `experiments/run_cross_sectional.py` | `state_edges.tsv` |
| conditional evolutionary inflow | `src/relobstq_mhn/core/inflow.py` | `experiments/run_cross_sectional.py` | `state_scores.tsv` (`F_hat`) |
| relative dwell/stasis score `R*` | `src/relobstq_mhn/core/scoring.py` | `experiments/run_cross_sectional.py` | `state_scores.tsv` (`R_star`) |
| fixed-theta count bootstrap | `src/relobstq_mhn/core/bootstrap.py` | `experiments/run_cross_sectional.py` | `bootstrap_summary.tsv` |
| E4/E10/E11/E14/E15/E16 evidence | `src/relobstq_mhn/workflows/secondary.py` | `experiments/run_secondary.py` | `core_evidence/*/tables/` |
| continuous dwell-gradient generator/evaluation | `src/relobstq_mhn/simulation/`, `workflows/simulation.py` | `experiments/run_simulation.py` | `simulation_dwell_gradient/tables/` |
| topology robustness | `src/relobstq_mhn/workflows/topology.py` | `experiments/run_topology_robustness.py` | `simulation_topology_robustness/tables/` |
| selected external longitudinal consistency | exact retained legacy contract in `experiments/run_longitudinal.py` | `experiments/run_longitudinal.py` | `reference_results/experiment_17_legacy/` |
| hashed result persistence | `src/relobstq_mhn/io/` | all current runners | `result_manifest.tsv`, `run_metadata.json` |
| final evidence verification/freeze | `experiments/freeze_final_evidence.py` | same | `reference_results/final_manuscript_evidence/` |

The reusable method package is plotting-free. `examples/plot_core_results.py`
is the only publication-figure example and consumes completed result tables.
Historical experimental plotting scripts are not part of the reproducible
method core.


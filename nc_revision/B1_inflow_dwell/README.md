# B1: paired inflow and dwell mechanism experiment

Core runner for the completed B1 experiment: baseline, inflow-only, dwell-only,
and joint conditions. It reuses the repository's generator and scoring code.
Only repository discovery differs from the original workspace runner; the
scientific functions, seeds, target selection and scoring are unchanged.

From the repository root, with the project Python environment activated:

```sh
python -B nc_revision/B1_inflow_dwell/run_b1.py init
python -B nc_revision/B1_inflow_dwell/run_b1.py pilot
python -B nc_revision/B1_inflow_dwell/run_b1.py batch
python -B nc_revision/B1_inflow_dwell/run_b1.py summary
```

Use Python 3.12, numpy 1.26.4, pandas 3.0.5 and scipy 1.14.1, with this
repository's dependencies installed. The bounded runner currently uses Windows
taskkill on timeout. Run in a writable checkout; outputs stay beside the runner.

The init command reads the existing frozen simulation_dwell_gradient config
and truth table in reference_results/final_manuscript_evidence. It fixes the
two highest-supported singleton targets (E11 and E10), six repeat seeds
20261001-20261006, N=5000 per condition, and a total budget of 900 seconds.
The pilot is repeat 1 of the same six-repeat plan. Initialization refuses to
overwrite an existing plan; pilot/batch validate hashes and reuse checkpoints.

Inflow-only reweights root branches toward the targets by two while preserving
the root total exit rate. Dwell-only scales target outgoing rates by one half
(D=2), preserving target conditional successor probabilities. Known generating
P is used; every condition recomputes occupancy, inflow, eligibility and median
normalization. Controls are occupancy, uniform inflow and frequency inflow.
Actual trajectory arrivals and waiting draws verify the interventions.

Outputs include KEY_RESULTS.csv, repeat_method_metrics.csv, paired_responses.csv,
mechanism_figure_data.csv, coverage.csv, manifests and per-repeat checkpoints.
Raw simulated trajectories are generated locally and are not bundled here.
The raw occupancy/inflow ratio and normalization are reported separately.
Comparison uses shared eligible target states; missing scores stay unscored.
No fitted-cMHN branch, real patient data or manuscript edits are included.

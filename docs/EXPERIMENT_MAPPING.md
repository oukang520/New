# Experiment-to-workflow mapping

The manuscript experiment numbers describe scientific questions. They are not
software modules. The public code groups them by shared computation so the same
formula is implemented once.

| Original evidence | Public workflow | Primary outputs |
|---|---|---|
| E1 preparation, E3-E5 core | `experiments/prepare_cross_sectional.py`, `experiments/run_cross_sectional.py` | fixed p15 input QC, theta, one-step edges, occupancy, inflow, R* and top states |
| E6 and continuous-gradient supplement | `experiments/run_simulation.py` | truth states, repeat scores, ordering/calibration metrics |
| E7 | `experiments/run_topology_robustness.py` | supplementary oracle-backbone topology/sparsity/dwell-placement robustness tables; explicitly excludes cMHN refit error |
| E10 | `experiments/run_secondary.py` | eligible-state R* landscape and cohort summary |
| E11 | `experiments/run_secondary.py` | information-gain summary |
| E9 | `core/scoring.py::compute_observation_enrichment`; no current experiment contract | function available, historical result legacy-only |
| E13 | `workflows/replication.py::compare_score_tables`; no split/refit runner | comparison function only |
| E14 | `experiments/run_secondary.py` | full-MHN, uniform-inflow, frequency-inflow and occupancy-only denominator ablation |
| E15 | `experiments/run_secondary.py` | matched-decoy and inflow-pairing falsification tables |
| E16 | `experiments/run_secondary.py` via `workflows/topology.py::topology_route_table` | six table-form evolutionary routes with R* values |
| E17 | `experiments/run_longitudinal.py` | selected external longitudinal consistency analysis using the legacy full-cohort frequency/co-occurrence backbone; weak eligible challenge-cohort and strict official-cMHN sensitivity tables are frozen under `reference_results/` |

The plotting-heavy historical scripts are absent except for E17, whose exact
legacy runner is intentionally restored by project decision. Other historical
experiments retain the refactored numerical method implementation. This does
**not** make every historical E1-E16 number a reproduced output of v0.2.0.
Exact status is maintained in
`RESULT_PROVENANCE_MATRIX.tsv`. The manuscript-critical current evidence is
frozen under `reference_results/final_manuscript_evidence/`; E9 and E13 remain
explicitly secondary historical analyses.

# Selected manuscript experiment mapping

The authoritative inventory is `FIGURE_EXPERIMENT_MAP.tsv` (43 PDF items, including
one reference table; 14 experiment numbers). It replaces earlier coverage claims.
The files listed there are identifiers, not distributed graphics.

| Selected panels | Numerical entry point | Configuration |
|---|---|---|
| E1, E2 | `python -m sirdwell_experiments e01_02` | bundled `experiments_01_02.yaml` and `selected_experiment_datasets.yaml` |
| E3 | `python -m sirdwell_experiments e03` | bundled `experiment_03.yaml` |
| E4 | `python -m sirdwell_experiments e04` | bundled `experiment_04.yaml` |
| E5 | `python experiments/run_cross_sectional.py` | `configs/cross_sectional.yaml` |
| E6 | `python experiments/run_simulation.py` | `configs/simulation.yaml`: 5000 samples, 60 repeats |
| E9 | `python -m sirdwell_experiments e09` | bundled resolved `experiment_09.yaml` |
| E10, E11, E14, E15B, E16 | `python experiments/run_secondary.py` | `configs/secondary.yaml` |
| E13 | `python -m sirdwell_experiments e13` | bundled resolved `experiment_13.yaml` |
| E17 | `python experiments/run_longitudinal.py`; `python -m sirdwell_experiments e17_tables` | `configs/longitudinal.yaml`; bundled `longitudinal_tables.yaml` |

E14's simulation comparison uses the E6 simulation tables. E15A is retired.
E7 is outside this 43-item inventory and remains a separately callable numerical
robustness workflow, never substituted for any selected panel.

E3/E4 are historical interface/sensitivity contracts used by the selected panels.
E5/E6/E10/E11/E14/E15B/E16 use the frozen current core workflows. They must not be
silently replaced with earlier experimental implementations. E13 uses a locked
inflow backbone, not a split-wise refitted cMHN. E17 deliberately freezes the
selected frequency/co-occurrence backbone, rather than changing it when mhn is installed.

Each configuration retains its own thresholds and seeds. Differences between
experimental contracts are intentional; no global threshold harmonization is applied.

E13 upstream input: `python -m sirdwell_experiments e05_split_input` reconstructs
the specific Experiment-5 score-table schema used by this panel. Its parameters
are in `src/sirdwell_experiments/parameters/experiment_05_split_input.yaml`; this
supporting input computation is not the current E5 panel workflow.

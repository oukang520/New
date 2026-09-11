# Result contracts

Core `relobstq_mhn` workflows write tab-separated tables under `OUTPUT/tables/`, a resolved
JSON configuration, `run_metadata.json`, and `result_manifest.tsv` containing
file hashes. Run metadata records the command, UTC timestamp, Git commit when
resolvable, Python/package versions, workflow seed/backend fields, and hashes
of supplied input paths.

## Cross-sectional preparation

- `mhn_training_matrix.csv`: pure binary p15 event matrix in fixed event order.
- `mhn_row_index_map.csv`: row-to-analysis-unit mapping and state fields.
- `state_table.csv`: sample-level p15 state assignments.
- `tables/event_panel.tsv`: prespecified model events and selection rule.
- `tables/preparation_qc.tsv`: sample/patient/event counts and contract checks.

## Cross-sectional

- `state_occupancy.tsv`: observed state counts and fractions.
- `state_edges.tsv`: one-event MHN predecessor contributions.
- `state_scores.tsv`: F-hat, R*, eligibility and bootstrap columns.
- `top_relative_dwell_states.tsv`: ranked high-confidence R* states.
- `theta.tsv`, `cv_scores.tsv`, `fit_metadata.json`: MHN fit audit.
- `quality_control.tsv`: schema and score-domain checks.

## Continuous dwell simulation

- `truth_states.tsv`: implanted D values and pilot support.
- `repeat_state_scores.tsv`: recovered R* for every truth state/repeat.
- `repeat_metrics.tsv`: rank recovery, gain, concordance and error.
- `performance_summary.tsv`: median and interquartile range per metric.
- `evaluation_coverage.tsv`: evaluable truth-state coverage for every dwell level.

## Topology robustness simulation

- `canonical_contract.tsv`: labels E7 as a supplementary oracle-backbone test.
- `truth_states.tsv`: implanted continuous dwell levels by condition.
- `condition_metrics.tsv`: repeat-level recovery by topology, sparsity and dwell placement.
- `condition_summary.tsv`: condition-level median and interquartile recovery metrics.

## Cross-sectional core evidence

- `inflow_computability_summary.tsv`: E4 finite-inflow and finite-R* audit.
- `rstar_landscape_states.tsv`, `rstar_landscape_summary.tsv`: E10 state landscape.
- `information_gain_summary.tsv`: E11 occupancy/inflow non-equivalence.
- `denominator_ablation_*.tsv`: E14 denominator specificity.
- `inflow_shuffle_*.tsv`: retained E15B inflow-pairing control; E15A is removed.
- `topology_routes.tsv`: E16 representative dominant-predecessor routes.
- `evidence_contract.tsv`: evidence-unit to output mapping.

## External longitudinal consistency

- Input contract: cBioPortal study exports under
  `Data/longitudinal_public/cbioportal/STUDY/`, including mutation, sample
  clinical and patient clinical tables.
- `tables/dwell_persistence_predictions_all.tsv`: selected-driver longitudinal
  pair predictions from the legacy full-cohort backbone analysis.
- `tables/dwell_persistence_summary_all.tsv`: per-cohort discrimination,
  persistence contrast, dwell-proxy correlation and bootstrap intervals.
- `tables/integrated_longitudinal_metrics_table.tsv`: manuscript-facing GLASS,
  CRC-triplets and MNM-WashU metrics.
- `STUDY/tables/fit_metadata.json`: records whether cMHN or the configured
  frequency/co-occurrence fallback generated the full-cohort backbone.

The frozen aggregate reference tables are under
`reference_results/experiment_17_legacy/`. They report GLASS AUC 0.67,
CRC-triplets AUC 0.65 and MNM-WashU AUC 0.89.

Plots are not result contracts for the refactored workflows. The selected E17
numerical runner retains its original score and metric calculations; graphical output is removed.

## Final evidence freeze

`experiments/freeze_final_evidence.py` verifies every source manifest and clean
Git run, then copies manuscript-facing outputs under
`reference_results/final_manuscript_evidence/`. It also creates p15-input,
runtime/environment, MHN-model-selection and all-file SHA-256 audit tables.


## Selected interface and validation inputs

E1/E2 read `processed/experiment_ready/COHORT/` as listed by the bundled
`selected_experiment_datasets.yaml`. They write the p10/p15/p20/p25 matrices,
event panels, state tables and state-scheme/QC summaries under
`results/experiments_01_02/`. E3 consumes those p15 tables and writes the fitted
theta, CV scores and one-event transitions under `results/experiment_03_mhn_interface/`.
E4 consumes E1/E3 tables and writes each inflow rule and predecessor-edge table
under `results/experiment_04_relative_inflow/`.

The current E5 workflow instead uses `configs/cross_sectional.yaml` and the
prepared inputs described above. For the selected E13 panel only,
`e05_split_input` consumes E3/E4 and writes the exact `eligible_experiment5`
score schema under `results/experiment_05_state_scores/`. E13 joins that table
to `processed/experiment_ready/COHORT/state_table.csv` and writes 50 split
replicates per cohort, representative state scores and summary statistics.
Its locked backbone and thresholds are the resolved panel contract.

E9 writes `state_recovery_long.tsv`, `repeat_metrics.tsv`, `repeat_curves.tsv`
and representative scores from the configured synthetic model. E17's table
extension consumes `dwell_persistence_predictions_all.tsv`, per-study
`state_scores.tsv` and `core_metric_table.tsv` from the numerical E17 runner.
It writes calibration tables, integrated metrics and native route-node tables.

The extracted numerical modules preserve their original CSV/TSV column contracts.
They do not all share the integrated core's metadata format. Full literal I/O
expressions and entry-point parameter files are inventoried in `NUMERICAL_IO.json`.

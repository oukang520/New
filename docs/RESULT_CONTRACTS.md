# Result contracts

Every workflow writes tab-separated tables under `OUTPUT/tables/`, a resolved
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
- `matched_decoy_*.tsv`, `inflow_shuffle_*.tsv`: E15 controls.
- `topology_routes.tsv`: E16 representative dominant-predecessor routes.
- `evidence_contract.tsv`: evidence-unit to output mapping.

## Longitudinal validation

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

Outcome-independent cohort rules and all analyzed weak/negative non-primary
cohorts are under `reference_results/experiment_17_supplement/`. The
patient-grouped official-cMHN sensitivity, including the negative CRC result,
is frozen under `reference_results/experiment_17_sensitivity/`.

Plots are not result contracts for the refactored workflows. The selected E17
legacy runner retains its original table-and-figure generation behavior.

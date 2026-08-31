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

## Longitudinal validation

- Input contract: `sample_metadata.tsv` with `analysis_id`, `patient_id`,
  `collection_time`, and `stage_group`; `event_matrix.tsv` with `analysis_id`
  followed by binary event columns.
- `sample_predictions.tsv`: patient-grouped out-of-fold R* predictions.
- `crossfit_audit.tsv`: training/held-out counts, fitting backend, selected
  lambda/grid boundary, seed, event count and finite theta audit by fold.
- `pair_predictions.tsv`: leakage-controlled predictions and observed outcomes.
- `longitudinal_metrics.tsv`: AUC, AP lift, persistence contrast and dwell-proxy
  correlation with patient-cluster bootstrap intervals when `patient_id` is
  available.

Plots are not result contracts. The optional scripts under `examples/` consume
these tables without recomputing or altering any statistic.

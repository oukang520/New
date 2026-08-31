# Result contracts

Every workflow writes tab-separated tables under `OUTPUT/tables/`, a resolved
JSON configuration, and `result_manifest.tsv` containing file hashes.

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
- `crossfit_audit.tsv`: training/held-out counts and fitting backend by fold.
- `pair_predictions.tsv`: leakage-controlled predictions and observed outcomes.
- `longitudinal_metrics.tsv`: AUC, AP lift, persistence contrast and dwell-proxy correlation with bootstrap intervals.

Plots are not result contracts. The optional scripts under `examples/` consume
these tables without recomputing or altering any statistic.

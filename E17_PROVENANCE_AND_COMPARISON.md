# E17 Provenance and Selection Record

## Selected primary E17

The selected primary analysis is the original E17 implementation, restored
verbatim as `experiments/run_longitudinal.py` with
`configs/longitudinal.yaml`. All three final `fit_metadata.json` files identify
`frequency_cooccurrence_backbone`, not official cMHN. The fixed backbone is
constructed from the full cohort and R* occupancy terms are estimated from
training patients. This is a cohort-level external consistency analysis rather
than a fully out-of-fold model-refitting benchmark.

The release configuration sets `mhn.enabled: false` to freeze the backend that
actually produced these selected outputs. Without this lock, installing the
optional MHN package could change the estimator and no longer reproduce the
selected result.

| cohort | n (persistent/changed) | AUC [95% CI] | AP lift | top-bottom persistence delta [CI] | rho minimum-dwell proxy [CI] |
|---|---:|---:|---:|---:|---:|
| GLASS | 72 (51/21) | 0.67 [0.53, 0.79] | 1.19 | 0.28 [-0.04, 0.47] | 0.19 [-0.03, 0.42] |
| CRC-triplets | 23 (20/3) | 0.65 [0.35, 0.91] | 1.08 | 0.12 [-0.21, 0.45] | 0.18 [-0.18, 0.50] |
| MNM-WashU | 10 (9/1) | 0.89 [0.75, 1.00] | 1.10 | 0.33 [0.00, 0.75] | 0.42 [0.31, 0.81] |

These values are the manuscript-facing E17 results. Frozen aggregate tables and
fit metadata are stored under `reference_results/experiment_17_legacy/`. They
must not be described as fully out-of-fold official-cMHN results.

## Superseded strict sensitivity analysis

A later audit assigned whole patients to five folds and refitted official
`mhn==1.2.3` cMHN in every training fold. It also changed pair eligibility and
score coverage. This is a scientifically useful stress test, but it no longer
matches the selected E17 estimand or implementation and is not the primary
analysis.

| cohort | evaluable (P/C) | AUC [patient-bootstrap CI] | AP lift | top-bottom delta [CI] | rho minimum-dwell proxy [CI] |
|---|---:|---:|---:|---:|---:|
| GLASS | 188 (137/51) | 0.547 [0.455, 0.649] | 1.036 | 0.127 [-0.078, 0.261] | 0.249 [0.091, 0.384] |
| CRC-triplets | 41 (38/3) | 0.228 [0.092, 0.413] | 0.951 | -0.143 [-0.357, 0.000] | -0.246 [-0.424, -0.067] |
| MNM-WashU | 10 (9/1) | 0.556 [0.333, 0.813] | 1.052 | 0.143 [-0.333, 0.429] | 0.063 [-0.278, 0.408] |

These superseded values remain in Git history for provenance. They are not
shipped as current result artifacts and must not be substituted into the
selected E17 table.

## Objective interpretation

- GLASS gives moderate directional discrimination with uncertainty overlapping
  a null persistence contrast.
- CRC is directionally favorable but highly uncertain because only three
  changed pairs are evaluable.
- MNM is strongly directionally favorable but has only one changed pair, so it
  is supporting rather than definitive evidence.
- Across all cohorts, the selected results support compatibility between R* and
  longitudinal persistence; they do not establish calibrated calendar-time
  prediction or universal generalization.

## Manuscript decision

Use the restored original E17 analysis and frozen reference tables as the
selected **external longitudinal consistency analysis**. Disclose the
full-cohort frequency/co-occurrence backbone and the small changed classes.
The stricter patient-grouped official-cMHN refit changes both the estimator and
evaluable-pair population, but it is scientifically relevant sensitivity
evidence and is retained under `reference_results/experiment_17_sensitivity/`.
The eligible weak BRCA-MSK challenge cohort and negative ALP-breast design
pilot are retained under `reference_results/experiment_17_supplement/`.

Status: `RESOLVED_WITH_CAVEAT_AND_TRANSPARENT_SENSITIVITY`.

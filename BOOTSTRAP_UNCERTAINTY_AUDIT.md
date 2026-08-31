# Bootstrap Uncertainty Audit

## Current procedure

`bootstrap_relative_dwell` performs a multinomial resample of observed state counts with the cohort total fixed. For each replicate it recomputes:

- state occupancy `L_v`;
- inflow contributions using resampled source occupancy;
- `F_hat`, the eligible-state median normalizer, and `R*`;
- top-k membership stability and percentile intervals.

The following are fixed across replicates:

- cMHN theta;
- event panel;
- graph topology;
- edge probabilities derived from theta;
- stage and one-step transition rule.

Thus these intervals quantify **sampling uncertainty conditional on the fitted MHN backbone**. They are not full estimator uncertainty and do not include MHN fitting, event-panel selection, preprocessing, or model-selection uncertainty.

## E5 verification

The historical 200-replicate implementation was reproduced exactly for all three cohorts using the historical E4 edges and counts. The current 500-replicate refinement changes mean top-state stability only slightly; see `E5_LEGACY_VS_CURRENT.md`.

## Recommended parameter-uncertainty analysis

The most feasible next layer is subset-refit sensitivity: independently refit cMHN on repeated patient subsets, recompute `R*`, and report common-state rank correlation/top-k overlap. A full nested bootstrap is substantially more expensive, especially for COAD. Until run, this is `RECOMMENDED_FUTURE_ANALYSIS`, not evidence.

Status: `RESOLVED_WITH_CAVEAT`.

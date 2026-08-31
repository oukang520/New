# E17 Provenance and Comparison

## Legacy E17

Historical outputs came from `src/run_experiment_17_longitudinal_public.py` and `configs/experiment_17_longitudinal_public.yaml`. All three final `fit_metadata.json` files identify `frequency_cooccurrence_backbone`, not official cMHN. Theta/backbone construction used the full cohort before score cross-fitting, so held-out patients could influence the fixed backbone.

| cohort | n (persistent/changed) | AUC [95% CI] | AP lift | top-bottom persistence delta [CI] | rho minimum-dwell proxy [CI] |
|---|---:|---:|---:|---:|---:|
| GLASS | 72 (51/21) | 0.67 [0.53, 0.79] | 1.19 | 0.28 [-0.04, 0.47] | 0.19 [-0.03, 0.42] |
| CRC-triplets | 23 (20/3) | 0.65 [0.35, 0.91] | 1.08 | 0.12 [-0.21, 0.45] | 0.18 [-0.18, 0.50] |
| MNM-WashU | 10 (9/1) | 0.89 [0.75, 1.00] | 1.10 | 0.33 [0.00, 0.75] | 0.42 [0.31, 0.81] |

These values are `E17_LEGACY` and must not be described as OOF official-cMHN results.

## Current patient-grouped OOF official-cMHN

The current workflow assigns whole patients to five folds, fits official `mhn==1.2.3` cMHN only on other patients, computes training-fold occupancy/R*, and predicts held-out samples. Event-loss pairs are excluded according to the prespecified config. Every fold reports training/held-out patients and samples, selected lambda, fit backend, seed, event count, theta shape, and boundary-grid status.

Input conversion changed column names/order only and is hashed in `outputs/phase0_reaudit_2026/E17_input_conversion_audit.json`.

| cohort | evaluable (P/C) | AUC [patient-bootstrap CI] | AP lift | top-bottom delta [CI] | rho minimum-dwell proxy [CI] |
|---|---:|---:|---:|---:|---:|
| GLASS | 188 (137/51) | 0.547 [0.455, 0.649] | 1.036 | 0.127 [-0.078, 0.261] | 0.249 [0.091, 0.384] |
| CRC-triplets | 41 (38/3) | 0.228 [0.092, 0.413] | 0.951 | -0.143 [-0.357, 0.000] | -0.246 [-0.424, -0.067] |
| MNM-WashU | 10 (9/1) | 0.556 [0.333, 0.813] | 1.052 | 0.143 [-0.333, 0.429] | 0.063 [-0.278, 0.408] |

Current files: `outputs/phase0_reaudit_2026/E17_CURRENT_OOF_CMHN/`.

## Score-source audit

Frozen-code counts before the label correction were:

| cohort | exact state | genotype fallback | not evaluable |
|---|---:|---:|---:|
| GLASS | 441 | 16 | 236 |
| CRC-triplets | 81 | 0 | 57 |
| MNM-WashU | 50 | 3 | 83 |

The fallback was labeled `genotype_stage_median` but actually pooled the genotype across stages. The repaired label is `genotype_median_across_stages`; point estimates are unchanged.

## Objective interpretation

- GLASS gives weak-to-moderate direct support, especially for the rank correlation with the conservative dwell proxy.
- CRC is a clear contradiction under leakage-controlled official-cMHN, not merely a null result.
- MNM is too class-imbalanced to support a strong discrimination claim.
- The larger current evaluable counts arise from OOF score coverage and differ from legacy exact-state-only selection; the versions are not numerically interchangeable.

## Manuscript decision

Use `E17_CURRENT_OOF_CMHN` if E17 is retained, because its leakage control and official backend are scientifically stronger. Report the result as **mixed longitudinal evidence**, not uniform validation. The favorable legacy metrics may appear only in provenance/sensitivity material with the fallback/full-cohort-backbone limitation stated explicitly.

Status: `RESOLVED_WITH_CAVEAT`.

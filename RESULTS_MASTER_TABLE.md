# Results Master Table

Canonical manuscript-facing results frozen on 2026-09-01. Values in this file
must be traceable through `RESULT_PROVENANCE_MATRIX.tsv`; historical values do
not replace current values unless explicitly marked secondary.

## Real cross-sectional cohorts

| evidence | metric | AACR_LUAD | AACR_COAD | AACR_IDC |
|---|---|---:|---:|---:|
| E3 | samples | 27,148 | 12,693 | 10,964 |
| E3 | observed states | 2,270 | 2,036 | 752 |
| E3 | selected lambda | 0.0003684 | 0.0007878 | 0.0002736 |
| E3 | selected multiplier | 10 (boundary) | 10 (boundary) | 3 |
| E4 | positive-inflow states | 2,107 | 1,729 | 708 |
| E4 | eligible finite `R*` states | 416 | 282 | 167 |
| E4 | one-step edges | 5,782 | 4,399 | 1,707 |
| E5 | high-confidence states | 233 | 147 | 111 |
| E5 | observed top-10 high-confidence stability, 500 bootstraps | 0.6704 | 0.6758 | 0.7192 |
| E10 | maximum `R*` | 21.290 | 7.337 | 5.725 |
| E10 | states with `R* > 2` | 63 | 51 | 34 |
| E10 | states with `R* < 0.5` | 68 | 54 | 26 |
| E11 | Spearman `R*` vs occupancy | 0.092 | 0.174 | 0.222 |
| E11 | Spearman `R*` vs inflow | -0.457 | -0.486 | -0.414 |
| E11 | top-10 overlap with occupancy / inflow | 1 / 0 | 3 / 0 | 5 / 2 |
| E14 | top-10 retention: uniform / frequency / occupancy | 0.30 / 0.20 / 0.10 | 0.40 / 0.40 / 0.30 | 0.50 / 0.30 / 0.40 |
| E15A | fraction above matched-decoy q90 | 1.00 | 1.00 | 1.00 |
| E15A | median log2 `R*` advantage | 1.532 | 1.466 | 0.751 |
| E15B | median shuffled top-10 overlap | 0.10 | 0.30 | 0.40 |
| E15B | median overlap loss | 0.90 | 0.70 | 0.60 |
| E16 | representative routes | 6 | 6 | 6 |

E5 stability is the mean high-confidence top-bootstrap stability for the ten
highest observed `R*` states satisfying the high-confidence count threshold.
The bootstrap resamples state counts while keeping theta and edge probabilities
fixed. Historical 200-replicate values remain only in `E5_LEGACY_VS_CURRENT.md`.

## Continuous simulated dwell truth

E6-gradient uses 5,000 trajectories per repeat, 60 repeats, and five implanted
relative-dwell levels. The backbone is the generating theta.

| metric | `R*` median | occupancy median | paired Wilcoxon p |
|---|---:|---:|---:|
| Spearman correlation | 0.765 | 0.530 | 8.15e-12 |
| Kendall correlation | 0.623 | 0.411 | 8.15e-12 |
| pairwise concordance | 0.842 | 0.727 | 8.15e-12 |
| calibration slope | 0.688 | 0.479 | 8.15e-12 |
| median absolute log2 error (lower is better) | 0.881 | 1.164 | 1.06e-10 |
| correctly ordered adjacent levels (of 4) | 4 | 3 | 5.28e-10 |

The lowest dwell level had 163/300 evaluable state instances; all five levels
were represented in all 60 repeats. Results therefore support continuous
relative ordering, not only an easy binary bottleneck distinction.

## Topology robustness

E7 contains 36 topology/sparsity/bottleneck-placement conditions and 360
repeat-level evaluations. Mean evaluable-state fraction was 0.837 (range
0.40-1.00). Across repeats, median Spearman correlation was 0.934 for `R*` and
0.874 for occupancy; median gain was 0.049. Gain was positive in 235/360
repeats, nonnegative in 260/360, and positive for 25/36 condition medians.

Interpretation: supportive but modest, heterogeneous robustness. This is not a
claim of universal superiority and does not include cMHN refit error.

## External longitudinal consistency

### Selected analysis

| cohort | evaluable n (persistent/changed) | AUC (95% CI) | AP lift | high-low persistence difference (95% CI) | rho with minimum-dwell proxy (95% CI) |
|---|---:|---:|---:|---:|---:|
| GLASS | 72 (51/21) | 0.67 (0.53, 0.79) | 1.19 | 0.28 (-0.04, 0.47) | 0.19 (-0.03, 0.42) |
| CRC-triplets | 23 (20/3) | 0.65 (0.35, 0.91) | 1.08 | 0.12 (-0.21, 0.45) | 0.18 (-0.18, 0.50) |
| MNM-WashU | 10 (9/1) | 0.89 (0.75, 1.00) | 1.10 | 0.33 (0.00, 0.75) | 0.42 (0.31, 0.81) |

### Outcome-independent challenge and strict sensitivity

| analysis | cohort | n (P/C) | AUC | AP lift | high-low difference | dwell-proxy rho |
|---|---|---:|---:|---:|---:|---:|
| eligible challenge | BRCA-MSK | 47 (31/16) | 0.49 (0.33, 0.69) | 1.01 | 0.06 (-0.29, 0.42) | 0.08 (-0.18, 0.32) |
| ineligible design pilot | ALP-breast | 13 (8/5) | 0.225 | 0.949 | -0.40 | -0.427 |
| strict official-cMHN | GLASS | 188 (137/51) | 0.547 | 1.036 | 0.127 | 0.249 |
| strict official-cMHN | CRC-triplets | 41 (38/3) | 0.228 | 0.951 | -0.143 | -0.246 |
| strict official-cMHN | MNM-WashU | 10 (9/1) | 0.556 | 1.052 | 0.143 | 0.063 |

The selected analysis is favorable but not decisive. It uses a fixed
full-cohort fallback backbone and has small changed classes in CRC and MNM.
Weak, negative, and strict sensitivity results remain part of the evidence.

## Secondary historical analyses

E9 O-star and E13 fixed-backbone split stability may be used only as
supplementary context under
their provenance labels. They do not establish dwell-time truth and are not
needed for the primary method claim.

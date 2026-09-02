# Final Reproducibility Report v3

Audit date: 2026-09-02

## Decision

**PASS - READY FOR MANUSCRIPT WRITING**

The Phase-0 blockers identified in the v2 audit are closed. The final
cross-sectional evidence was regenerated from the exact p15 inputs through
official `mhn==1.2.3`, fitted theta, conditional evolutionary inflow, and
`R*`. The core current-code experiments and two simulation contracts were
then regenerated and frozen with file-level hashes.

This decision approves manuscript writing from the frozen evidence. It does
not remove the interpretation limits below.

## Frozen evidence

- Evidence root: `reference_results/final_manuscript_evidence/`
- Index: `FINAL_EVIDENCE_INDEX.tsv` (99 verified files)
- Source runs: all record `git dirty=false`
- Runtime: Python 3.12.13, `mhn==1.2.3`, NumPy 1.26.4, pandas 3.0.5
- Cohorts: AACR_LUAD, AACR_COAD, AACR_IDC
- Event panels: 15 events per cohort
- Model selection: five-fold cross-validation with the one-standard-error rule
- Bootstrap: 500 multinomial count resamples conditional on fitted theta

## Current cross-sectional closure

| cohort | samples | observed states | eligible `R*` states | selected lambda | multiplier | boundary |
|---|---:|---:|---:|---:|---:|---|
| AACR_LUAD | 27,148 | 2,270 | 416 | 0.0003683513 | 10 | yes |
| AACR_COAD | 12,693 | 2,036 | 282 | 0.0007878358 | 10 | yes |
| AACR_IDC | 10,964 | 752 | 167 | 0.0002736228 | 3 | no |

All theta matrices were finite, all genotype alignments had zero mismatches,
and the three p15 matrices matched their frozen SHA-256 values. The LUAD and
COAD selected penalties lie at the tested search-grid boundary; this is
reported as a model-selection sensitivity limitation, not hidden.

## Regenerated manuscript evidence

The freeze contains current-code outputs for inflow computability (E4),
conditional-bootstrap stability (E5), the real-cohort `R*` landscape (E10),
component distinctness (E11), denominator ablation (E14), matched-decoy and
inflow-shuffle controls (E15A/B), and six representative dominant-predecessor
routes per cohort (E16). Continuous dwell-gradient simulation (E6-gradient)
and topology/sparsity/placement robustness (E7) have canonical configs,
runners, contracts, metadata, and manifests.

E6-gradient supports the intended continuous target: median Spearman
correlation was 0.765 for `R*` versus 0.530 for occupancy, with better pairwise
ordering, calibration, and absolute error in paired analyses. E7 was favorable
but deliberately graded as supplementary: `R*` exceeded occupancy in 235/360
repeats and 25/36 condition medians, so the advantage is modest and
condition-dependent rather than universal.

## E17 disposition

Experiment 17 is named **external longitudinal consistency analysis**. Its
selected three-cohort estimator uses a frozen full-cohort
frequency/co-occurrence fallback backbone and training-patient occupancy; it
is not a patient-grouped out-of-fold official-cMHN validation and does not
estimate calendar time.

The frozen E17 reference result contains only GLASS, CRC-triplets and
MNM-WashU under this selected analysis contract. These results must be
presented as supportive consistency evidence, not decisive external validation
or evidence of estimator robustness.

## Permanent interpretation limits

1. `R*` is a cohort-relative dwell/stasis proxy, not absolute dwell time.
2. `F_hat` is conditional relative evolutionary inflow; it excludes absolute
   exit rates and stage transitions.
3. The predecessor graph contains observed, same-stage, one-event additions.
4. E5 quantifies sampling uncertainty conditional on fixed fitted theta.
5. E6-gradient and E7 use an oracle generating-theta backbone and do not include
   cMHN refit error.
6. E13 is fixed-backbone internal sampling stability, not independent replication.
7. E16 routes are representative dominant-predecessor routes, not phylogenies.
8. E17 is sample-size-limited external consistency evidence under a fallback,
   full-cohort backbone and does not establish estimator-robust validation.

## Final status

The method, code, current cross-sectional evidence, simulation truth evidence,
selected longitudinal evidence, and manuscript control documents now form one
traceable evidence chain. No unresolved Phase-0 evidence blocker remains.

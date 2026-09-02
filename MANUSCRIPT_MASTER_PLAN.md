# Manuscript Master Plan

## Central message

Cross-sectional state abundance alone conflates evolutionary supply and state
accumulation. Rel-ObsTQ-MHN combines observed occupancy with an MHN-derived
conditional inflow backbone to estimate a normalized, cohort-relative
dwell/stasis proxy (`R*`).

## Results story

1. **Define and audit the estimator.** Introduce `L`, conditional `F_hat`, raw
   occupancy-to-inflow ratio, and median-normalized `R*`; lock the boundary that
   the output is relative rather than calendar time.
2. **Establish real-data computability.** Fit official cMHN models to LUAD,
   COAD, and IDC p15 cohorts and show finite, supported state scores.
3. **Demonstrate nonredundancy.** Use E11 and E14 to show that `R*` rankings are
   not equivalent to occupancy, inflow, or interchangeable denominators.
4. **Validate the core innovation against truth.** Use E6-gradient to show
   recovery of a continuous implanted dwell ordering beyond occupancy.
5. **Test robustness and falsification.** Present E7 as heterogeneous
   robustness, then E15 controls as evidence that structured inflow pairing is
   necessary for the observed high-state rankings.
6. **Return to real evolutionary context.** Use E16 representative routes to
   make the state-level signal interpretable without calling routes phylogenies.
7. **Close with longitudinal evidence.** Report favorable selected E17 results
   together with the null BRCA challenge cohort and strict mixed sensitivity;
   call the section external longitudinal consistency, not definitive validation.

## Section sequence

### Introduction

- Cross-sectional recurrence mixes arrival frequency and persistence.
- Existing progression models estimate event dependencies but do not directly
  expose a state-level relative dwell/stasis score.
- State the contribution and its deliberately relative interpretation.

### Methods

- Cohorts and fixed event panels.
- Official cMHN fitting and one-standard-error model selection.
- Observed same-stage one-step graph and conditional inflow.
- `R*` definition, eligibility, normalization, and conditional bootstrap.
- Simulation generator and truth metrics.
- Longitudinal pair construction, result-independent cohort rules, selected
  fallback estimator, and strict sensitivity estimator.
- Statistical analysis and reproducibility controls.

### Results

Use the seven-step story above. Every numerical statement must be copied from
`RESULTS_MASTER_TABLE.md` and checked against frozen TSVs.

### Discussion

- Emphasize relative dwell/stasis and supply-adjusted state accumulation.
- Discuss topology/model dependence, conditional bootstrap, grid-boundary
  penalties, cross-sectional identifiability, longitudinal heterogeneity, and
  the need for larger repeated-genomic cohorts.
- Do not claim absolute time, causality, patient-specific phylogeny, or universal
  longitudinal prediction.

## Evidence policy

Primary claims require grade A/A- evidence or a clearly qualified B result.
E9 and E13 are supplementary and cannot carry the central claim.
Weak and negative sensitivity evidence is reported rather than selected away.

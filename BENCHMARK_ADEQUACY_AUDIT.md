# Benchmark Adequacy Audit

## Target task

The method estimates a state-level relative dwell/stasis proxy from cross-sectional state occupancy and an evolutionary transition backbone. This is not ordinary mutation-order inference, survival prediction, phylogeny reconstruction, or absolute waiting-time estimation.

## Existing controls in this repository

| comparator | classification | what it tests |
|---|---|---|
| occupancy-only | baseline/component control | whether prevalence alone explains recovery/ranking |
| MHN inflow-only | component control | whether transition accessibility alone explains ranking |
| uniform denominator | ablation | whether denominator structure matters |
| frequency denominator | ablation | whether a simpler frequency surrogate suffices |
| shuffled state-inflow pairing | falsification/negative control | whether correct state-specific pairing matters |

These are scientifically relevant controls but are not external competing methods for the identical target quantity.

## External-method boundary

Methods such as oncotrees, HyperTraPS, TiMEx, and time-aware hazard models generally differ in input assumptions and output target. They should be added only if a method accepts comparable cross-sectional genomic states and emits a directly evaluable state-level relative residence quantity. Mutation order or edge recovery alone is not an equivalent benchmark.

The repository contains no audited implementation of an external method that satisfies all four comparability criteria: same scientific target, compatible inputs, comparable output quantity, and the same evaluation target. Adding an unrelated algorithm would create a decorative benchmark rather than a fair one.

## Decision

The current simulation comparisons and ablations are adequate for component validation, but the manuscript should call them baselines/ablations/falsifications, not competing state-of-the-art dwell estimators. A targeted literature review remains advisable before submission to confirm whether a newer directly comparable method exists.

Status: `RESOLVED_WITH_CAVEAT`.

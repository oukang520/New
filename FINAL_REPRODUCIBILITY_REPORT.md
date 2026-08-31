# Final Reproducibility Report

## A. Repository state

- Original frozen audit commit: `e4215608dd394581da19e1ae0d8a0206c4d33798`.
- Phase-0 repaired code/audit baseline commit: `f430286c8a9c7175dbb0191705531c9eaf09c34f`.
- Audit environments: Python 3.11.16 and 3.12.13, official `mhn==1.2.3`, Windows x86-64.
- Final pre-commit tests: 11 passed, one expected small-sample Wilcoxon warning.
- Original frozen commit self-contained: **NO**.
- Repaired package self-contained excluding restricted data: **YES**.

## B. P0 issues

The missing preprocessing package, canonical p15 preparation, environment verification, result metadata, E5 bootstrap conflict, theoretical quantity naming, E17 backend/leakage conflict, E17 bootstrap unit, clinical result retention, and topology route definition were addressed. Exact evidence and caveats are listed in `PHASE0_REPRODUCIBILITY_AUDIT.md`.

## C. Result provenance

- **Reproduced:** E1 p15 matrices; E5 historical bootstrap metrics; E17 current OOF official-cMHN workflow.
- **Legacy with strengthened provenance:** E3-E4, E6, E7-E16 historical numbers as classified row-by-row in the matrix.
- **Changed:** E17 numerical results after official fold-specific cMHN and patient-grouped leakage control.
- **Not claimed reproduced:** any historical experiment for which the current package has only a reusable function or no canonical runner/config.

## D. Changed numerical results

| experiment | old | new | reason |
|---|---|---|---|
| E5 | 200 bootstrap replicates | 500-replicate stability in `E5_LEGACY_VS_CURRENT.md` | reproducibility refinement plus documented threshold/seed defaults |
| E17 GLASS AUC | 0.67 | 0.547 | full-cohort fallback backbone replaced by patient-grouped OOF official cMHN |
| E17 CRC AUC | 0.65 | 0.228 | same; direction reverses |
| E17 MNM AUC | 0.89 | 0.556 | same; only one changed pair remains highly unstable |

No result was adjusted to match a desired conclusion.

## E. E17 decision

Legacy E17 used a frequency/co-occurrence fallback backbone and allowed the full cohort to influence theta/backbone construction. Current E17 refits official cMHN inside each patient-grouped training fold. GLASS supports a modest positive association, CRC contradicts it, and MNM is underpowered. The manuscript should use current results and state **mixed, limited longitudinal support**.

## F. Reproducibility layers

| layer | status | note |
|---|---|---|
| authorized provider raw to harmonized cohort tables | REQUIRES_DATA | provider-specific extraction not distributable |
| harmonized cross-sectional tables to exact p15 input | RESOLVED | exact SHA-256 match for all three cohorts |
| p15 input to official cMHN | RESOLVED_WITH_CAVEAT | code/backend verified; full costly E3 refit not repeated in Phase-0 |
| theta/occupancy to F-hat/R* | RESOLVED | tested canonical implementation |
| conditional bootstrap | RESOLVED_WITH_CAVEAT | fixed-backbone uncertainty only |
| continuous simulation | RESOLVED_WITH_CAVEAT | current oracle-backbone runner exists; end-to-end refit extension remains recommended |
| longitudinal OOF evaluation | RESOLVED_WITH_CAVEAT | fully rerun; mixed outcome and small changed classes |
| all historical E1-E17 manuscript tables | UNRESOLVED | several remain legacy-only without canonical runners |

## G. Manuscript caveats that must remain

1. `R*` is cohort-relative and not absolute calendar dwell time.
2. `F_hat` uses conditional next-event probabilities and excludes absolute exit rates and stage transitions.
3. E5 bootstrap intervals are conditional on fitted theta.
4. E8 supports plausibility, not significant pathway enrichment.
5. E12 is association, is directionally heterogeneous, and adds little C-index.
6. E13-fixed-backbone is internal sampling stability, not independent full-pipeline replication.
7. E16 routes are representative dominant-predecessor routes, not phylogenies.
8. E17 provides mixed and sample-size-limited direct longitudinal evidence.

## H. Readiness decision

**NOT READY FOR MANUSCRIPT WRITING**

Minimum remaining blockers:

1. Update manuscript claims/tables to the current mixed E17 result rather than the favorable legacy fallback result.
2. Decide which legacy-only experiments are truly manuscript-critical, then provide canonical runner/config/result contracts for those retained rows or explicitly move them to provenance-labeled historical supplements.
3. Perform one clean end-to-end current cross-sectional run, especially the costly COAD official-cMHN fit, and freeze its manifests before using v0.2.0-generated E3-E16 numbers.

The package is now suitable for continued reproducibility work, but declaring all manuscript evidence regenerated would be inaccurate.

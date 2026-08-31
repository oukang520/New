# Final Reproducibility Report

## A. Repository state

- Original frozen audit commit: `e4215608dd394581da19e1ae0d8a0206c4d33798`.
- Phase-0 repaired code/audit baseline commit: `f430286c8a9c7175dbb0191705531c9eaf09c34f`.
- Selected original E17 restoration commit: `1a03c8183719286d5f822963d8653cd71e230814`.
- Audit environments: Python 3.11.16 and 3.12.13, official `mhn==1.2.3`, Windows x86-64.
- Final selected-E17 rollback tests: 14 passed, one expected small-sample Wilcoxon warning.
- Original frozen commit self-contained: **NO**.
- Repaired package self-contained excluding restricted data: **YES**.

## B. P0 issues

The missing preprocessing package, canonical p15 preparation, environment verification, result metadata, E5 bootstrap conflict, theoretical quantity naming, E17 backend/leakage conflict, E17 bootstrap unit, clinical result retention, and topology route definition were addressed. Exact evidence and caveats are listed in `PHASE0_REPRODUCIBILITY_AUDIT.md`.

## C. Result provenance

- **Reproduced or restored exactly:** E1 p15 matrices; E5 historical bootstrap metrics; selected original E17 code and aggregate outputs.
- **Legacy with strengthened provenance:** E3-E4, E6, E7-E16 historical numbers as classified row-by-row in the matrix.
- **Superseded sensitivity:** the later fold-specific official-cMHN E17 audit changed the estimator and evaluation population and is retained only in Git history.
- **Not claimed reproduced:** any historical experiment for which the current package has only a reusable function or no canonical runner/config.

## D. Changed numerical results

| experiment | selected result | sensitivity result | disposition |
|---|---|---|---|
| E5 | 200 bootstrap replicates | 500-replicate stability in `E5_LEGACY_VS_CURRENT.md` | retain both with labels |
| E17 GLASS AUC | 0.67 | 0.547 | selected original restored; strict audit superseded |
| E17 CRC AUC | 0.65 | 0.228 | selected original restored; strict audit superseded |
| E17 MNM AUC | 0.89 | 0.556 | selected original restored; strict audit superseded |

No result was adjusted to match a desired conclusion.

## E. E17 decision

The project selected the original E17 estimand and implementation as primary.
It uses a full-cohort frequency/co-occurrence backbone with training-patient
occupancy scoring and gives directionally favorable AUCs in GLASS (0.67), CRC
(0.65) and MNM (0.89). The manuscript may describe this as supportive external
longitudinal consistency, while explicitly disclosing the fixed full-cohort
backbone and small changed classes. It must not call E17 an out-of-fold official
cMHN refit or a calibrated calendar-time prediction test.

## F. Reproducibility layers

| layer | status | note |
|---|---|---|
| authorized provider raw to harmonized cohort tables | REQUIRES_DATA | provider-specific extraction not distributable |
| harmonized cross-sectional tables to exact p15 input | RESOLVED | exact SHA-256 match for all three cohorts |
| p15 input to official cMHN | RESOLVED_WITH_CAVEAT | code/backend verified; full costly E3 refit not repeated in Phase-0 |
| theta/occupancy to F-hat/R* | RESOLVED | tested canonical implementation |
| conditional bootstrap | RESOLVED_WITH_CAVEAT | fixed-backbone uncertainty only |
| continuous simulation | RESOLVED_WITH_CAVEAT | current oracle-backbone runner exists; end-to-end refit extension remains recommended |
| selected longitudinal evaluation | RESOLVED_WITH_CAVEAT | exact legacy runner and aggregate results restored; fixed full-cohort fallback backbone and small changed classes disclosed |
| all historical E1-E17 manuscript tables | UNRESOLVED | several remain legacy-only without canonical runners |

## G. Manuscript caveats that must remain

1. `R*` is cohort-relative and not absolute calendar dwell time.
2. `F_hat` uses conditional next-event probabilities and excludes absolute exit rates and stage transitions.
3. E5 bootstrap intervals are conditional on fitted theta.
4. E8 supports plausibility, not significant pathway enrichment.
5. E12 is association, is directionally heterogeneous, and adds little C-index.
6. E13-fixed-backbone is internal sampling stability, not independent full-pipeline replication.
7. E16 routes are representative dominant-predecessor routes, not phylogenies.
8. E17 provides directionally supportive but sample-size-limited longitudinal consistency evidence using a fixed full-cohort fallback backbone.

## H. Readiness decision

**NOT READY FOR MANUSCRIPT WRITING**

Minimum remaining blockers:

1. Keep E17 wording within external consistency and disclose its fallback/full-cohort backbone; do not present it as fully OOF official-cMHN validation.
2. Decide which legacy-only experiments are truly manuscript-critical, then provide canonical runner/config/result contracts for those retained rows or explicitly move them to provenance-labeled historical supplements.
3. Perform one clean end-to-end current cross-sectional run, especially the costly COAD official-cMHN fit, and freeze its manifests before using v0.2.0-generated E3-E16 numbers.

The package is now suitable for continued reproducibility work, but declaring all manuscript evidence regenerated would be inaccurate.

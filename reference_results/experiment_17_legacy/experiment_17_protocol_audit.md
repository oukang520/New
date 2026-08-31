# Experiment 17 Protocol Audit

Purpose: validate the Rel-ObsTQ-MHN relative dwell-time idea on public real longitudinal or quasi-longitudinal cohorts.

Cohort evidence tiers:
- Primary validation cohorts: difg_glass, coadread_mskcc.
- Supplementary integrated cohorts: mnm_washu_2016.

Design focus:
1. Build driver-event state spaces from cBioPortal public processed data.
2. Fit a fixed-penalty cMHN backbone when feasible; otherwise use an audited frequency/co-occurrence backbone.
3. Compute state-level R* as observed occupancy divided by MHN-derived expected inflow.
4. Predict relative dwell length from held-out-training R* and validate it against same-patient longitudinal genotype persistence.

Empirical dwell proxy:
A pair is labeled persistent when the selected driver-event genotype is unchanged between adjacent ordered samples from the same patient. This is a conservative observable proxy for long relative dwell; it is not a direct measurement of continuous residence time.

A stricter time-length proxy is also recorded: the minimum observed dwell interval equals the adjacent observation interval when the selected-driver genotype is persistent and 0 when it has changed. This gives a conservative lower bound rather than a direct residence-time measurement.

Excluded cohort audit:
| study_id | short_name | reason |
| --- | --- | --- |
| nsclc_tracerx_2017 | TRACERx | only 5-6 evaluable adjacent pairs under audited driver panels; underpowered for dwell-persistence accuracy. |
| msk_chord_2024 | MSK-CHORD | pan-cancer ordered pairs are dominated by changed genotypes and lack a usable persistent class for dwell-persistence validation. |
| all_phase2_target_2018_pub | TARGET-ALL | relapse flags do not provide usable ordered adjacent sample pairs for this validation. |
| prad_su2c_2019 | SU2C-PRAD | only 2 evaluable adjacent pairs; usable as a smoke test but not as scientific validation. |
| brca_aurora_2023 | AURORA | downloaded through the cBioPortal API and audited; primary-metastatic design is strong, but held-out R* scoring leaves only 4 evaluable retained pairs under the full driver-state model. |
| breast_alpelisib_2020 | ALP-breast | downloaded and audited; longitudinal samples are dominated by cfDNA pre/on/post-treatment profiles, and strict held-out R* validation is directionally negative. |
| brca_mbcproject_2022 | MBCProject | downloaded and audited; serial biopsy days are available, but tissue/liquid biopsy mixture and sparse held-out state coverage produce no reliable support for R* dwell validation. |
| brca_dldccc_2022 | TNBC-DLDCCC | downloaded and audited; only 11 monotone-QC retained adjacent pairs and only 1 changed genotype pair, so it cannot support dwell-discrimination metrics. |
| skcm_broad_brafresist_2012 | BRAF-resist | downloaded and audited; biologically relevant pre/post resistance cohort, but only 14 monotone-QC retained adjacent pairs after removing apparent driver-loss pairs. |
| crc_hta8_htan_2024 | HTAN-CRC | downloaded and audited; primary/metastatic design is relevant, but only 6 monotone-QC retained adjacent tumor pairs, likely due strong multi-region sampling differences. |
| lung_smc_2016 | SMC-lung | downloaded and audited; only 6 retained adjacent pairs and no changed selected-driver genotype pairs, so discrimination is not estimable. |
| pcnsl_msk_2024 | PCNSL | downloaded and audited; CSF/biopsy ordering is quasi-longitudinal and only 1 changed retained pair, so it is not reliable for main validation. |
| egc_msk_2017 | EGC-MSK | downloaded and audited; primary/metastatic biology is relevant but only 14 retained adjacent pairs and 3 changed pairs, underpowered for main validation. |
| nepc_wcm_2016 | NEPC-WCM | downloaded and audited; only 6 retained adjacent prostate-metastatic pairs, insufficient for stable dwell validation. |
| bm_nsclc_mskcc_2023 | NSCLC-BM | downloaded and audited; many patients have multiple samples, but same-patient samples are usually within the same specimen category rather than ordered PT-to-metastasis pairs. |
| aml_target_2018_pub | AML-TARGET | downloaded and audited; mutation records are dominated by diagnostic -03/-09 samples and do not provide reliable recurrent mutation pairs. |
| lipo_msk_2026 | LIPO-MSK | downloaded and audited; strict monotone longitudinal scoring retained only 3 evaluable pairs and all belonged to the persistent class. |
| breast_msk_2018 | BRCA-MSK | downloaded and audited; 47 evaluable pairs were available, but core metrics were mixed rather than clearly supportive, so it is retained as extension evidence rather than a main Experiment 17 cohort. |

Shared figure rules used:
1. Keep each main figure aligned with one primary scientific question.
2. Prefer one annotated matrix over repeated small panels for factorial robustness experiments.
3. Use color for the primary scalar and a second compact encoding only when it directly supports the same claim.
4. Put auxiliary diagnostics in reports or supplement-style tables unless they are needed to interpret the main claim.
5. Export vector PDF and 600-dpi PNG for every final figure.
6. Run a boundary/cropping check after rendering every final figure.
7. Preserve data transformations, sample sizes, exclusions, missing-value handling and uncertainty definitions in the paired report.
8. Use position on a common scale as the default quantitative encoding; reserve area, node size and color intensity for secondary signals.
9. Do not connect missing observations or convert non-evaluable values to zero unless zero is scientifically meaningful.
10. Use shared axes and near-square panels when cross-cohort comparison is the point of the panel.
11. Avoid distant legends when direct labels or compact in-panel keys can explain the encoding without stealing space.

Reference design patterns used:
- Color policy: extract_layout_and_annotation_logic_only; do_not_extract_colors.
1. embedded micro legends: Place compact scale keys or legends inside the figure's unused top/right area when direct numeric labels already carry exact values. Suitable for: dense matrices, line panels and compact benchmark summaries.
2. direct in panel annotation: Annotate the main comparison, effect direction or exceptional condition directly inside the plotting area instead of relying on long legends. Suitable for: single-message panels and mechanism-linked measurements.
3. boxed context insets: Add small boxed mechanism or variable-definition insets only when they reduce ambiguity about what a metric means. Suitable for: mechanistic figures and conceptual method panels.
4. shared axis small multiples: Use aligned small multiples with shared axes when separate panels represent different variables or experimental conditions. Suitable for: multi-condition benchmarks and comparative validation.
5. summary with raw points: Overlay raw points on bars or summary intervals when replicate-level evidence is part of the claim. Suitable for: replicate experiments and uncertainty-aware summaries.
6. phase bands and arrows: Use shaded bands, arrows or bracketed windows to mark process stages when the x-axis has an ordered temporal or procedural meaning. Suitable for: time-series, trajectory and intervention-stage figures.
7. uncertainty bands: Pair fitted trends with transparent uncertainty bands when modeling a continuous relationship. Suitable for: regression and sensitivity analyses.
8. compact panel letters: Keep panel letters small, bold and outside the data region; avoid extra title text when axes already explain the content. Suitable for: multi-panel composite figures.
9. metric table panel: Replace cluttered bar/point summaries with a compact table when the exact value and criterion are the main evidence. Suitable for: validation matrices, audit metrics and uncertainty summaries.
10. ranked forest panel: Use horizontal intervals with a zero/reference line for effect estimates so direction, uncertainty and cohort identity are read on one common scale. Suitable for: effect-size comparison, correlation summaries and cross-cohort validation.
11. compact trajectory strip: Use short aligned topology or trajectory strips with direct terminal labels when route order matters more than full network geometry. Suitable for: real-state progression paths and relative dwell-time route illustrations.
# CLAIM_EVIDENCE_MATRIX



目标是把论文中的每一句主要 claim 约束在已有证据边界内，避免把“相对停留时间代理”写成“真实绝对时间”。



| claim | evidence | strength | safe_wording | caveat |
| --- | --- | --- | --- | --- |
| Rel-ObsTQ-MHN defines a state-level relative dwell-time index from cross-sectional genomic data. | Formula and implementation in core scoring/transitions/pipeline; E3-E5 real cohort execution. | Strong for computational definition and implementation. | estimates a relative dwell/stasis proxy; does not measure absolute calendar time. | Requires a progression transition model and sufficient state support. |
| R* is not equivalent to raw occupancy or MHN inflow alone. | E11 top-overlap, rank-gain and correlation results; E14 denominator ablation; E15 falsification. | Strong. | R* adds state-specific information beyond prevalence and model inflow. | Information gain is shown on selected three AACR cohorts. |
| R* can recover known relative dwell structure in controlled simulations. | E6 enhanced bottleneck recovery and E6 continuous dwell-gradient. | Strong. | R* outperforms occupancy in controlled settings with known dwell truth. | Simulations still simplify real tumor evolution and use known/inferred topology assumptions. |
| R* remains robust under changes in topology, sparsity and bottleneck placement. | E7 balanced topology grid: R* global Spearman 0.482 vs occupancy 0.255. | Moderate to strong. | R* retains directional advantage across stress conditions. | The gain is condition-dependent and not a universal perfect recovery guarantee. |
| Real AACR high-R* states are biologically plausible in LUAD, COAD and IDC. | E8 and E10 top states/modules; E16 route overlays. | Moderate to strong as biological plausibility evidence. | high-R* states are enriched for recognizable tumor-type-relevant driver contexts. | Module p-values in E8 are not strong enough to claim formal pathway enrichment. |
| R* has clinical relevance. | E12 subgroup Cox and survival profiles. | Supportive but secondary. | R* is associated with clinical outcome in several cohort/subgroup settings. | IDC shows mixed direction; survival is not a direct dwell-time label. |
| R* generalizes to held-out/internal splits. | E13 patient split replication: median rho around 0.87-0.90, top10 enrichment >5x. | Strong for internal stability. | R* ranking is stable under patient resampling/splitting. | Backbone was fixed from full E5 outputs according to current review; independent split refit would be stronger. |
| R* predictions are directionally consistent with real longitudinal persistence. | E17 GLASS, CRC-triplets, MNM-WashU integrated metrics. | Moderate and direct, with sample-size caveat. | external longitudinal cohorts provide directional support for high-R* states being more persistent. | Small evaluable pairs, particularly CRC/MNM changed pairs; fallback backbone in E17 should be disclosed. |
| O* identifies cross-sectional enrichment not expected from progression-only models. | E9 O* simulation. | Strong for auxiliary O* positive control. | O* is a companion residual for observation enrichment, not the main R* dwell claim. | O* should not be over-positioned as the primary novelty. |

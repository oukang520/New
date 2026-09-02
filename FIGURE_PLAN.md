# Manuscript Figure Plan

The final paper should preserve the real cross-sectional to simulation to real
longitudinal evidence chain. Panels are rendered from frozen tables, not by
recomputing methods inside plotting code.

## Main figures

| figure | scientific job | evidence | recommended visual contract |
|---|---|---|---|
| Figure 1 | define inputs, conditional inflow, and normalized `R*` | method + E1/E4 | compact workflow plus one real state example |
| Figure 2 | show real-cohort `R*` landscapes and component distinctness | E10/E11 | three aligned cohort panels; rank/interval plots; compact metric table |
| Figure 3 | test continuous relative-dwell truth | E6-gradient | truth-recovery curves plus paired metric summary |
| Figure 4 | test denominator necessity and falsification | E14/E15A/E15B | ablation rank-retention and negative-control summaries |
| Figure 5 | place `R*` on representative real evolutionary routes | E16 | six routes per cohort; clearly label as representative |
| Figure 6 | assess external longitudinal consistency | E17 | selected cohort estimates with CIs; explicitly label the selected fallback-backbone contract |

## Supplementary figures and tables

| item | evidence | reason |
|---|---|---|
| Supplementary Figure S1 | E3/E4/E5 QC and bootstrap | implementation and sampling stability |
| Supplementary Figure S2 | E7 | robustness is important but modest/heterogeneous |
| Supplementary Figure S3 | E13 | internal stability evidence |
| Supplementary Table S1 | panels, hashes, CV and theta metadata | reproducibility |

## Shared rendering rules

- Use the frozen public palette and a color-blind-safe semantic mapping.
- Final assembled text must remain at least 7 pt; panel labels 9-10 pt bold.
- Prefer near-square single panels where scientifically appropriate.
- Export vector PDF/SVG and at least 600 dpi raster previews.
- Do not place A/B/C/D titles inside standalone source panels.
- Do not encode a numeric quantity only by area or color.
- Check clipping, overlap, empty cohorts, legend duplication, and axis consistency
  at final assembled size.

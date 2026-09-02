# Claim-Evidence Matrix

## Primary claims

| claim | direct evidence | grade | permitted wording | prohibited overclaim |
|---|---|---|---|---|
| `R*` is computable from occupancy and MHN-derived conditional inflow in real cohorts | E3-E5 | A | finite, cohort-relative state-level proxy in three AACR cohorts | absolute dwell time or calendar duration |
| `R*` recovers continuous relative dwell ordering beyond occupancy in controlled simulation | E6-gradient | A | better rank, pairwise, calibration and error metrics under the generating backbone | end-to-end cMHN estimator accuracy |
| `R*` is not a relabeling of occupancy or inflow | E11, E14 | A | low top-state overlap and changed rankings under denominator ablations | statistical independence of all components |
| high `R*` states depend on biologically structured inflow pairing | E15A/B | A- | matched-decoy advantage and loss under shuffled inflow pairing | causal biological mechanism |
| continuous-dwell recovery is reasonably robust to topology design | E7 | B+ | mostly favorable, modest, condition-dependent robustness | universal superiority |
| selected real longitudinal observations are directionally consistent with relative persistence | E17 selected | B | supportive external longitudinal consistency | fully independent OOF validation or calibrated time prediction |

## Supporting claims

| claim | evidence | grade | required qualification |
|---|---|---|---|
| high and low relative-dwell states coexist in real cohorts | E10 | A- | descriptive cohort-relative landscape |
| representative routes can contextualize state-level `R*` | E16 | B+ | dominant-predecessor routes, not phylogenies |
| rankings are internally stable under patient splits with a fixed backbone | E13 | B | not an independent pipeline refit |

## Counterevidence and uncertainty

| evidence | implication for the manuscript |
|---|---|
| E7 is favorable in 235/360 repeats, not all repeats | state that robustness is heterogeneous |
| LUAD and COAD lambda selections are grid-boundary values | disclose and avoid implying penalty insensitivity |
| E5 fixes theta during bootstrap | call intervals conditional sampling uncertainty |

## Overall conclusion

The full evidence chain supports the method's central innovation as a
**relative state-level dwell/stasis proxy** that carries information beyond
occupancy alone. Evidence is sufficient for a methods manuscript when the
limits above are preserved. It does not support absolute time estimation or a
claim of uniformly strong longitudinal prediction.

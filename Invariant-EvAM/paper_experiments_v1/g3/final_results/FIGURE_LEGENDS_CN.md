# G3 当前单图图注

图件采用 Arial 字体，PDF/SVG 为矢量单图；导出设置为 600 dpi。必要条件、坐标单位、图例及有效统计连线保留在图内。各图的完整统计定义与适用范围见以下说明及[实验报告](REPORT_CN.md)。

## G3_01

[PDF](../final_figures/G3_01.pdf) · [SVG](../final_figures/G3_01.svg) · [作图数据](plot_data/G3_01.tsv)

本图的定义及统计说明见上述实验报告与作图数据；图内保留必要标签。

## G3_05

[PDF](../final_figures/G3_05.pdf) · [SVG](../final_figures/G3_05.svg) · [作图数据](plot_data/G3_05.tsv)

本图的定义及统计说明见上述实验报告与作图数据；图内保留必要标签。

## G3_06

[PDF](../final_figures/G3_06.pdf) · [SVG](../final_figures/G3_06.svg) · [作图数据](plot_data/G3_06.tsv)

完整题注与说明（保留原统计措辞）：

- 200 datasets per row; retained time / measurement endpoints; no within-group filtering.
- Common bandwidth within each block; reflected KDE; density is not uncertainty.

## G3_07

[PDF](../final_figures/G3_07.pdf) · [SVG](../final_figures/G3_07.svg) · [作图数据](plot_data/G3_07.tsv)

本图的定义及统计说明见上述实验报告与作图数据；图内保留必要标签。

## G3_R2_bias_0

[PDF](../final_figures/G3_R2_bias_0.pdf) · [SVG](../final_figures/G3_R2_bias_0.svg) · [作图数据](plot_data/G3_R2_bias_0.tsv)

完整题注与说明（保留原统计措辞）：

- G3 | Declared selection-bias bound · effect +0
- 500 datasets per point; Wilson intervals. Insets show the cost in correct answers.
- Guarantee applies only when the allowed bound contains the true bias.
- No calibration in this panel; log-OR bias is not a generator-coefficient bound.

## G3_R2_bias_0p8

[PDF](../final_figures/G3_R2_bias_0p8.pdf) · [SVG](../final_figures/G3_R2_bias_0p8.svg) · [作图数据](plot_data/G3_R2_bias_0p8.tsv)

完整题注与说明（保留原统计措辞）：

- G3 | Declared selection-bias bound · effect +0.8
- 500 datasets per point; Wilson intervals. Insets show the cost in correct answers.
- Guarantee applies only when the allowed bound contains the true bias.
- No calibration in this panel; log-OR bias is not a generator-coefficient bound.

## G3_R2_bias_neg0p8

[PDF](../final_figures/G3_R2_bias_neg0p8.pdf) · [SVG](../final_figures/G3_R2_bias_neg0p8.svg) · [作图数据](plot_data/G3_R2_bias_neg0p8.tsv)

完整题注与说明（保留原统计措辞）：

- G3 | Declared selection-bias bound · effect -0.8
- 500 datasets per point; Wilson intervals. Insets show the cost in correct answers.
- Guarantee applies only when the allowed bound contains the true bias.
- No calibration in this panel; log-OR bias is not a generator-coefficient bound.

## G3_R3_joint_cost1000

[PDF](../final_figures/G3_R3_joint_cost1000.pdf) · [SVG](../final_figures/G3_R3_joint_cost1000.svg) · [作图数据](plot_data/G3_R3_joint_cost1000.tsv)

完整题注与说明（保留原统计措辞）：

- G3 | Two-phase calibration · total cost 1,000
- Main: equal-weight mean of 6 nonzero-effect cells; 500 datasets per cell.
- Caps: 1.96 Monte Carlo SE. Insets: all 9 effect/bias cells at 50% allocation.
- Magnitude intervals concern source log-OR; they are not generator intervals.

## G3_R3_joint_cost4000

[PDF](../final_figures/G3_R3_joint_cost4000.pdf) · [SVG](../final_figures/G3_R3_joint_cost4000.svg) · [作图数据](plot_data/G3_R3_joint_cost4000.tsv)

完整题注与说明（保留原统计措辞）：

- G3 | Two-phase calibration · total cost 4,000
- Main: equal-weight mean of 6 nonzero-effect cells; 500 datasets per cell.
- Caps: 1.96 Monte Carlo SE. Insets: all 9 effect/bias cells at 50% allocation.
- Magnitude intervals concern source log-OR; they are not generator intervals.

## G3_R3_reference_anchor

[PDF](../final_figures/G3_R3_reference_anchor.pdf) · [SVG](../final_figures/G3_R3_reference_anchor.svg) · [作图数据](plot_data/G3_R3_reference_anchor.tsv)

完整题注与说明（保留原统计措辞）：

- G3 | Reference-target calibration
- 500 new datasets per scenario; Wilson intervals; actual cost 4,000 in every arm.
- Original reference interaction target; biology bound 0.5, pair-selection bound 1.
- No-anchor unresolved sets remain visible; coverage and failures in source tables.
- vs reference X within scenario; Holm | ns >= .05; * < .05; ** < .01; *** < .001

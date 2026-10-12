# G2 当前单图图注

图件采用 Arial 字体，PDF/SVG 为矢量单图；导出设置为 600 dpi。必要条件、坐标单位、图例及有效统计连线保留在图内。各图的完整统计定义与适用范围见以下说明及[实验报告](REPORT_CN.md)。

## G2_01

[PDF](../final_figures/G2_01.pdf) · [SVG](../final_figures/G2_01.svg) · [作图数据](plot_data/G2_01.tsv)

完整题注与说明（保留原统计措辞）：

- 500 replicates per cell; all 12 conditions retained.
- Answerability and conditional error

## G2_02

[PDF](../final_figures/G2_02.pdf) · [SVG](../final_figures/G2_02.svg) · [作图数据](plot_data/G2_02.tsv)

同一三事件合同下，比较完整数据经典似然剖面（连续端点、固定多起点、约束/无约束似然一致性检查）与闭合基因对子系统的条件置信外包。前9行是3种观测设计×3个总样本量，每行每法200条，4种真效应各50条；最后2行是新种子独立确认，总样本量16,000，每行每法100条，4种效应各25条。两方法使用相同数据配对，主要与新增确认不混合分母。左、中为互斥穷尽的组成：有效且覆盖且有限、有效且覆盖但无界、有效但不覆盖、数值一致性检查未通过；所有记录保留。95%竖虚线对照的是前两覆盖部分的累计长度。右侧为正确方向作答比例，实心圆经典剖面、空心方块条件外包，误差棒为Wilson 95%区间，分母包含零效应、拒答和数值无效记录。无界覆盖不当作有用的有限答案；条件方法在弱观测下舍弃第三基因信息，其拒答不等于完整数据不可识别。零正确作答时，Wilson区间下限与点估计同为0，区间只向右延伸；端帽不是组间显著性括号。此图展示状态组成，不套用其他柱形端点的星号。主参考R28（e8ef4c02b84393cf67c59f091dda8bcf.jpg）；保留共同行序、两组组成、右侧点图与独立图例，并增加置信区间和95%基准。

## G2_03

[PDF](../final_figures/G2_03.pdf) · [SVG](../final_figures/G2_03.svg) · [作图数据](plot_data/G2_03.tsv)

完整题注与说明（保留原统计措辞）：

- 12 conditions; 500 paired datasets per condition; labels: true interaction.
- Crossbars: paired 95% Monte Carlo intervals. All six zero gains retained.
- Top: condition counts and descriptive KDE; no additional replicate pooling.
- Linear axes retain zero / negative gains; faint diagonal denotes equal gains.

## G2_04

[PDF](../final_figures/G2_04.pdf) · [SVG](../final_figures/G2_04.svg) · [作图数据](plot_data/G2_04.tsv)

完整题注与说明（保留原统计措辞）：

- True interaction below each group; 200 paired replicates per effect; no numerical failures.
- Whiskers: Wilson 95% Monte Carlo intervals. Point estimates agree; interval procedures differ.
- vs official within effect/endpoint; Holm | ns >= .05; * < .05; ** < .01; *** < .001

## G2_05

[PDF](../final_figures/G2_05.pdf) · [SVG](../final_figures/G2_05.svg) · [作图数据](plot_data/G2_05.tsv)

完整题注与说明（保留原统计措辞）：

- Fixed equal mixture of all four effects; 2,000 replicates per bar.
- Whiskers: Wilson 95% Monte Carlo intervals. Failures remain in denominator.
- vs M within design; Holm | ns >= .05; * < .05; ** < .01; *** < .001

## G2_06

[PDF](../final_figures/G2_06.pdf) · [SVG](../final_figures/G2_06.svg) · [作图数据](plot_data/G2_06.tsv)

完整题注与说明（保留原统计措辞）：

- All five frozen confidence levels; Wilson 95% Monte Carlo intervals.
- Separate panels: zero calls at every level; conditional error undefined.

## G2_07

[PDF](../final_figures/G2_07.pdf) · [SVG](../final_figures/G2_07.svg) · [作图数据](plot_data/G2_07.tsv)

完整题注与说明（保留原统计措辞）：

- Black diamonds / crossbars: means / ±1 SD.
- Maximum absolute difference = 1.33e−9.
- All 800 paired estimates; no subsampling or jitter.

## G2_R2_sign_strong

[PDF](../final_figures/G2_R2_sign_strong.pdf) · [SVG](../final_figures/G2_R2_sign_strong.svg) · [作图数据](plot_data/G2_R2_sign_strong.tsv)

完整题注与说明（保留原统计措辞）：

- G2 | Exact sign inference · strong observation
- 500 paired datasets per effect; Wilson intervals. No raw 0/1 dot cloud.
- Classical exact testing is a component; no new-test or finite-magnitude claim.
- vs stratified exact within effect; Holm | ns >= .05; * < .05; ** < .01; *** < .001
- Stratified exact: null wrong calls 13/500 (2.6%).

## G2_R2_sign_weak

[PDF](../final_figures/G2_R2_sign_weak.pdf) · [SVG](../final_figures/G2_R2_sign_weak.svg) · [作图数据](plot_data/G2_R2_sign_weak.tsv)

完整题注与说明（保留原统计措辞）：

- G2 | Exact sign inference · weak observation
- 500 paired datasets per effect; Wilson intervals. No raw 0/1 dot cloud.
- Classical exact testing is a component; no new-test or finite-magnitude claim.
- vs stratified exact within effect; Holm | ns >= .05; * < .05; ** < .01; *** < .001
- Stratified exact: null wrong calls 26/500 (5.2%).

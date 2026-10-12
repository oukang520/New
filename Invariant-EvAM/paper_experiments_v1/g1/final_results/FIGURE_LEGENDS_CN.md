# G1 当前单图图注

图件采用 Arial 字体，PDF/SVG 为矢量单图；导出设置为 600 dpi。必要条件、坐标单位、图例及有效统计连线保留在图内。各图的完整统计定义与适用范围见以下说明及[实验报告](REPORT_CN.md)。

## G1A

[PDF](../final_figures/G1A.pdf) · [SVG](../final_figures/G1A.svg) · [作图数据](plot_data/G1A.tsv)

本图的定义及统计说明见上述实验报告与作图数据；图内保留必要标签。

## G1B

[PDF](../final_figures/G1B.pdf) · [SVG](../final_figures/G1B.svg) · [作图数据](plot_data/G1B.tsv)

在固定10,000次基因测量预算下，比较重复观察与互补观察的目标投影Fisher信息。3/4/6/8事件，链、稠密DAG及全双向网络，普通/稀有/较强交互，每单元10个独立参数抽样，总360条。主图30个正信息单元的中位数，每点代表10条；颜色为拓扑。粉色虚线表示信息相等，不是拟合回归。顶部和右侧为这些单元中位数在log10尺度上的描述性KDE（带宽因子0.4），不表示置信区间。嵌图使用全部300条正信息配对的log10(互补/重复)；箱体为四分位距，中位线为中位数，须为5–95百分位，极端值保留在数据中而未单独显示。全双向3/4事件的60条目标均不可识别，不能放入对数轴，因此在图内明确单列；没有用伪小正数代替零信息。六种设计、全部360条的数值及秩容差标记见experiment_data/G1_R8_geometry.tsv，其中46/2160项设计的秩对容差不稳定。结论限定正则局部总体几何，信息比不等于已实测的有限样本精度倍数。主参考R02（0a20f3a1a3899805f1516b4ead2df34e.jpg）；相对原参考采用单元中位数减少散点，增益箱线代替残差箱线，不复制参考图显著性。

## G1C

[PDF](../final_figures/G1C.pdf) · [SVG](../final_figures/G1C.svg) · [作图数据](plot_data/G1C.tsv)

本图的定义及统计说明见上述实验报告与作图数据；图内保留必要标签。

## G1D

[PDF](../final_figures/G1D.pdf) · [SVG](../final_figures/G1D.svg) · [作图数据](plot_data/G1D.tsv)

本图的定义及统计说明见上述实验报告与作图数据；图内保留必要标签。

## G1E

[PDF](../final_figures/G1E.pdf) · [SVG](../final_figures/G1E.svg) · [作图数据](plot_data/G1E.tsv)

本图的定义及统计说明见上述实验报告与作图数据；图内保留必要标签。

## G1_R2_recovery_frequent

[PDF](../final_figures/G1_R2_recovery_frequent.pdf) · [SVG](../final_figures/G1_R2_recovery_frequent.svg) · [作图数据](plot_data/G1_R2_recovery_frequent.tsv)

完整题注与说明（保留原统计措辞）：

- G1 | Fixed-cost recovery · frequent events
- Four independent parameter draws; 50 datasets per draw and design.
- Open points: draw means. Bars: mean; whiskers: minimum–maximum across draws.
- All outcomes retained, including numerical failures and boundary fits.
- vs heterogeneous; 4 parameter draws; Holm | ns >= .05; * < .05; ** < .01; *** < .001

## G1_R2_recovery_ordinary

[PDF](../final_figures/G1_R2_recovery_ordinary.pdf) · [SVG](../final_figures/G1_R2_recovery_ordinary.svg) · [作图数据](plot_data/G1_R2_recovery_ordinary.tsv)

完整题注与说明（保留原统计措辞）：

- G1 | Fixed-cost recovery · ordinary events
- Four independent parameter draws; 50 datasets per draw and design.
- Open points: draw means. Bars: mean; whiskers: minimum–maximum across draws.
- All outcomes retained, including numerical failures and boundary fits.
- vs heterogeneous; 4 parameter draws; Holm | ns >= .05; * < .05; ** < .01; *** < .001

## G1_R2_recovery_rare

[PDF](../final_figures/G1_R2_recovery_rare.pdf) · [SVG](../final_figures/G1_R2_recovery_rare.svg) · [作图数据](plot_data/G1_R2_recovery_rare.tsv)

完整题注与说明（保留原统计措辞）：

- G1 | Fixed-cost recovery · rare events
- Four independent parameter draws; 50 datasets per draw and design.
- Open points: draw means. Bars: mean; whiskers: minimum–maximum across draws.
- All outcomes retained, including numerical failures and boundary fits.
- vs heterogeneous; 4 parameter draws; Holm | ns >= .05; * < .05; ** < .01; *** < .001

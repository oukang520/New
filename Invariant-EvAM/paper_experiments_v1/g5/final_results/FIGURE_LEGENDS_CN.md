# G5 当前单图图注

图件采用 Arial 字体，PDF/SVG 为矢量单图；导出设置为 600 dpi。必要条件、坐标单位、图例及有效统计连线保留在图内。各图的完整统计定义与适用范围见以下说明及[实验报告](REPORT_CN.md)。

## G5_05_design_scenario_0

[PDF](../final_figures/G5_05_design_scenario_0.pdf) · [SVG](../final_figures/G5_05_design_scenario_0.svg) · [作图数据](plot_data/G5_05_design_scenario_0.tsv)

完整题注与说明（保留原统计措辞）：

- Independent simulation | Correct selection family
- Four-alias coverage at 50%: 95.5%; wrong / called: 2 / 34.
- 200 datasets per point; mean ± 1.96 MCSE. Insets: Wilson intervals.
- 0% is the exact no-acquisition contrast. Dashed boundary: remeasurement capacity.
- Four aliases have identical actions; no extra replicates. All outcome counts in source tables.

## G5_05_design_scenario_1

[PDF](../final_figures/G5_05_design_scenario_1.pdf) · [SVG](../final_figures/G5_05_design_scenario_1.svg) · [作图数据](plot_data/G5_05_design_scenario_1.tsv)

完整题注与说明（保留原统计措辞）：

- Independent simulation | Common omitted pair selection
- Four-alias coverage at 50%: 78.0%; wrong / called: 16 / 66.
- 200 datasets per point; mean ± 1.96 MCSE. Insets: Wilson intervals.
- 0% is the exact no-acquisition contrast. Dashed boundary: remeasurement capacity.
- Four aliases have identical actions; no extra replicates. All outcome counts in source tables.

## G5_05_design_scenario_2

[PDF](../final_figures/G5_05_design_scenario_2.pdf) · [SVG](../final_figures/G5_05_design_scenario_2.svg) · [作图数据](plot_data/G5_05_design_scenario_2.tsv)

完整题注与说明（保留原统计措辞）：

- Independent simulation | Biology shift (+0.2)
- Four-alias coverage at 50%: 95.5%; wrong / called: 3 / 30.
- 200 datasets per point; mean ± 1.96 MCSE. Insets: Wilson intervals.
- 0% is the exact no-acquisition contrast. Dashed boundary: remeasurement capacity.
- Four aliases have identical actions; no extra replicates. All outcome counts in source tables.

## G5_07_direct_uncertainty_cost

[PDF](../final_figures/G5_07_direct_uncertainty_cost.pdf) · [SVG](../final_figures/G5_07_direct_uncertainty_cost.svg) · [作图数据](plot_data/G5_07_direct_uncertainty_cost.tsv)

完整题注与说明（保留原统计措辞）：

- Direct records | honest answers, interval limits and acquisition cost
- All six strategies × three budget caps; no natural parameter truth is assumed.
- Repeated allocations of one held-out patient pool; signs are model-conditional, not validated biology.
- Unknown and unbounded answers remain visible; remeasurement does not create new patients.

## G5_R2_costmatched_COAD

[PDF](../final_figures/G5_R2_costmatched_COAD.pdf) · [SVG](../final_figures/G5_R2_costmatched_COAD.svg) · [作图数据](plot_data/G5_R2_costmatched_COAD.tsv)

完整题注与说明（保留原统计措辞）：

- G5 | Equal actual acquisition cost · COAD
- 50 allocation seeds; mean ± 1.96 seed-MCSE, conditional on the same patient pool.
- Each replicate uses a common feasible actual cost, chosen before reveal.
- Four action-equivalent strategies shown once; no independent algorithm advantage.

## G5_R2_costmatched_LUAD

[PDF](../final_figures/G5_R2_costmatched_LUAD.pdf) · [SVG](../final_figures/G5_R2_costmatched_LUAD.svg) · [作图数据](plot_data/G5_R2_costmatched_LUAD.tsv)

完整题注与说明（保留原统计措辞）：

- G5 | Equal actual acquisition cost · LUAD
- 50 allocation seeds; mean ± 1.96 seed-MCSE, conditional on the same patient pool.
- Each replicate uses a common feasible actual cost, chosen before reveal.
- Four action-equivalent strategies shown once; no independent algorithm advantage.

## G5_R3_anchor_called

[PDF](../final_figures/G5_R3_anchor_called.pdf) · [SVG](../final_figures/G5_R3_anchor_called.svg) · [作图数据](plot_data/G5_R3_anchor_called.tsv)

完整题注与说明（保留原统计措辞）：

- G5 | Source anchoring at equal acquisition cost
- 500 new draws per scenario; matched added cost 1,500; Wilson intervals.
- 350 source records versus 350 selected records; same 100 completed baseline records.
- Expanded nuisance family; original cost-74 results remain separately reported.
- vs more selected patients within scenario; Holm | ns >= .05; * < .05; ** < .01; *** < .001

## G5_R3_anchor_covered

[PDF](../final_figures/G5_R3_anchor_covered.pdf) · [SVG](../final_figures/G5_R3_anchor_covered.svg) · [作图数据](plot_data/G5_R3_anchor_covered.tsv)

完整题注与说明（保留原统计措辞）：

- G5 | Source anchoring at equal acquisition cost
- 500 new draws per scenario; matched added cost 1,500; Wilson intervals.
- 350 source records versus 350 selected records; same 100 completed baseline records.
- Expanded nuisance family; original cost-74 results remain separately reported.
- vs more selected patients within scenario; Holm | ns >= .05; * < .05; ** < .01; *** < .001

## G5_R3_natural_transfer

[PDF](../final_figures/G5_R3_natural_transfer.pdf) · [SVG](../final_figures/G5_R3_natural_transfer.svg) · [作图数据](plot_data/G5_R3_natural_transfer.tsv)

完整题注与说明（保留原统计措辞）：

- G5 | Natural-panel predictive transfer
- Paired patient-score differences; caps: 1.96 patient SE, conditional on fitted models.
- Transfer prior strength selected using training panels only. CTMC predictions are legacy.
- Same previously explored evaluation records; not an external replication.
- paired gain vs 0; conditional patient test; Holm | ns >= .05; * < .05; ** < .01; *** < .001

## G5_R6_correct_answers

[PDF](../final_figures/G5_R6_correct_answers.pdf) · [SVG](../final_figures/G5_R6_correct_answers.svg) · [作图数据](plot_data/G5_R6_correct_answers.tsv)

完整题注与说明（保留原统计措辞）：

- 150 independent datasets / condition; dots: three fixed, disjoint blocks of 50 datasets.
- Bars: correct-answer rates; whiskers: Wilson 95% intervals; brackets: paired McNemar, 40-test Holm.
- Grey: outside the fitted selection family. 此图为D4独立合同的补充验证；逐条件覆盖率、错误率和有限率见REPORT_CN.md的D4表及experiment_data/policy_summary.tsv，不与R8预算或正确率直接比较。

## G5_R8_diagnostic_actions

[PDF](../final_figures/G5_R8_diagnostic_actions.pdf) · [SVG](../final_figures/G5_R8_diagnostic_actions.svg) · [作图数据](plot_data/G5_R8_diagnostic_actions.tsv)

目标c-opt利用独立先导样本，在冻结的可行采集行动及相同新增成本6,000下选择动作，随后由独立确认数据评价。前4行为预设信息缺口，每行60条确认记录，共240条；后2行为起始样本量匹配的独立检查，各30条，两个筛选环境各2,000条基础确认样本与500条先导样本，仅改变筛选强度。左侧是动作选择比例：同类扩样、补联合测量、新互补环境、新来源、参考边际；中间为正确方向、未知、错误方向的结论构成；右侧为有限区间率及Wilson 95%区间。全部记录包含零效应；未观察到错误方向不代表真实错误率为零。成本约定为新来源完整记录6单位、筛选完整记录2单位、单基因或补测1单位，不是真实医院成本估计。置信构造仅保证声明的闭合子系统与选择家族条件；不据此推广到自然队列。主参考R28（e8ef4c02b84393cf67c59f091dda8bcf.jpg）；保留两种组成和侧列点图，6行代替参考截图的20类，增加Wilson区间。

## G5_R8_diagnostic_value

[PDF](../final_figures/G5_R8_diagnostic_value.pdf) · [SVG](../final_figures/G5_R8_diagnostic_value.svg) · [作图数据](plot_data/G5_R8_diagnostic_value.tsv)

比较目标c-opt、始终同类扩样及开发期冻结的最佳固定行动（新来源）。4种预设情景，每情景60条同基础确认记录的配对结果，新增预算均6,000。柱高为置信区间与[-4,4]交集长度的均值；越低越好。该有界评分不是完整区间宽度，无界区间没有从数据或分母删除，实际上下界和有限率另见原始数据与动作图。误差棒为10,000次配对重抽样所得的均值95%百分位区间。每柱空心点为固定种子、不依赖结果均匀选出的12/60条显示样本；三策略共享被选重复编号，全部60条进入均值、区间和检验。括号分别连接目标c-opt与同类扩样、最佳固定行动的具体柱中心，共8个双侧配对均值符号翻转检验，49,999次翻转、+1修正、Holm校正；零假设需配对差符号可交换。ns≥0.05，*<0.05，**<0.01，***<0.001；ns不表示等价。全部检验与显示点编号见配套表。该图保留与同类扩样相同的情景，不按结果删除比较。主参考R21（b7f665c0644dd5d35226c0b3e29df4f0.jpg）；保留窄柱、黑边、原始空心点、误差棒与顶部图例，按用户要求用完整两柱括号替代参考图的竖排P值。

## G5_BIO_01_clone_tracking

手术组织标记信息与术后非主干克隆追踪的公开计数回放。四个冻结策略完整保留：Clonal=CLONAL_ONLY、Diverse=BRANCH_DIVERSE经典强对照、Random=RANDOM_MARKERS、Adaptive=INFORMATION_ADAPTIVE。主曲线为 ALL197_CT_DNA 确认集、INTRINSIC_QC 合同，在名义标记预算20、50、100下的患者平均 wholePSP参照分支得分；圆点和同色线为均值，黑描边圆点与黑端帽为患者级2000次 bootstrap95%区间。虚线50只标记预登记主预算，不是处理阶段；曲线连接三个实际预算点，不是时间轨迹或平滑拟合。右侧数值为K100均值，解决接近终点的重叠；没有移动点的真实坐标或添加未注册比较星号。

分母必须按合同解释：74名确认患者中52名有术后血浆；71名有分支评分身份，其中51名有血浆、20名缺术后样本按冻结合同贡献0分，而不是71名全部实测。3名未知树保留MRD/容量资料但不冒充0分支真值。14名有wholePSP实际检出的非主干参照分支。K50左侧插图仅展示这14名whole-reference-positive患者的条件恢复率（全部14个真实患者点）：Clonal0%、Diverse83.93%、Random26.79%、Adaptive45.83%。右侧插图为39名具有明确真实复发事件和日期的患者，31名有血浆、8名缺样本按已冻结合同为未检出；复发前检出依次22/39、23/39、20/39、23/39。图中指标是含资料可用性的回放合同得分，不是对缺样本患者进行了真实测量。

标记选择只使用手术时可得组织信息和primary-only参考，不读取术后血浆或未来复发作选择。该实验使用已有完整患者特异面板的处理后UMI计数和误差/QC注释，未重新测序或FASTQ重跑，不证明自然演化theta、耐药机制、治疗获益或可部署小面板的独立误差校准。Adaptive没有超过固定Diverse强基线，整体MRD和复发前检出也不显示其独有优势。

重要边界保留：K50 Clonal有10/74标记容量不足；名义预算相同不等于每患者实际可用标记数完全相等。10个独立确认正常来源中，Adaptive/Diverse平均回放阳性率8.89%，Random10%，Clonal0；非恶性淋巴细胞体细胞突变混杂集中于CRUK0498，病例与错误均保留。因此分支覆盖增加不能直接解释为可靠临床MRD或患者获益。14个参照阳性和39个复发患者与本项目TRACERx组织资料存在患者重叠，不是新增独立临床队列。

参考R04的单行顶部图例、两真实指标插图、彩色线、黑描边圆点、黑色端帽；不同终点保留独立明确分母，不复制无实验意义的阶段阴影或星号。Arial、规定六色内配色、600dpi PDF+SVG，全部源患者与作图数据保留。

## G5_BIO_02_future_clone_tracking

真正后期组织支持的谱系目标追踪。确认集74名患者中12名具有独立的复发/进展后组织；其中10名有可评价的20个主肿瘤非主干谱系标记目标，并均有术后血浆和真实复发日期。另外2名有配对组织但无可评价非主干参照，保留未知/无参照身份，不作为阴性。未来目标必须有至少3个分支自有变异在真实未来区域获得支持；未来组织只用于该新增次级评分，不改变手术时已冻结的面板选择。左组为术后任一血浆的未来目标恢复率，右组只计真实复发前的血浆证据。每患者先计算自身目标恢复比例，再对完整10名患者求均值；不是20个目标或多份血浆作为独立样本。

四个名义K50策略完整保留。术后/复发前患者均值分别为Clonal0%/0%、Diverse63.33%/33.33%、Random20%/6.67%、Adaptive23.33%/6.67%。空心橙色WholePSP整面板参照为63.33%/33.33%，属于额外信息参照，未与K50策略等成本；数值相同不能宣称节省75%费用或验证一款真实重新测序的小面板。3个未来目标低于原PSP多标记容量，仍在分母保留，不按能否被PSP检测剔除。

窄分组柱、黑描边、空心患者点和黑色端帽参照R21。均值与2000次患者bootstrap95%区间均由完整10名患者决定；为减少散点密度，每组仅显示由固定患者ID SHA256规则选出的同6名患者，所有10名患者和显示标记均在作图数据中保留。这是新增描述性生物桥接终点，没有注册比较检验，因此无星号或P值。此结果支持利用经典分支多样性获取策略选择可追踪的真实后期谱系目标；不支持Adaptive优于Diverse、不验证CTMC theta、迁移方向或治疗获益，也不改变正常来源体细胞突变混杂和处理后UMI回放的临床可靠性边界。Arial、规定六色、600dpi PDF+SVG。

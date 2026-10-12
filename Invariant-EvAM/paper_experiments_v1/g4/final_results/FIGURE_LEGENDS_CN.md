# G4 当前单图图注

图件采用 Arial 字体，PDF/SVG 为矢量单图；导出设置为 600 dpi。必要条件、坐标单位、图例及有效统计连线保留在图内。各图的完整统计定义与适用范围见以下说明及[实验报告](REPORT_CN.md)。

## G4_01

[PDF](../final_figures/G4_01.pdf) · [SVG](../final_figures/G4_01.svg) · [作图数据](plot_data/G4_01.tsv)

完整题注与说明（保留原统计措辞）：

- Direct-record | patient flow and information boundary

## G4_07

[PDF](../final_figures/G4_07.pdf) · [SVG](../final_figures/G4_07.svg) · [作图数据](plot_data/G4_07.tsv)

完整题注与说明（保留原统计措辞）：

- Model-anchored | source-bootstrap profile noncoverage
- 10 source fits × 10 replicates; dots = source-fit noncoverage; bars = mean ± SD; dashed = 5%.
- vs K0 within source bound; 10 source fits; Holm | ns >= .05; * < .05; ** < .01; *** < .001

## G4_R2_joint_audit_COAD

[PDF](../final_figures/G4_R2_joint_audit_COAD.pdf) · [SVG](../final_figures/G4_R2_joint_audit_COAD.svg) · [作图数据](plot_data/G4_R2_joint_audit_COAD.tsv)

完整题注与说明（保留原统计措辞）：

- G4 | Joint information at equal assay cost · COAD
- 500 random audits per cost; mean width ± 1.96 MCSE; inset Wilson intervals.
- Target: finite-cohort reported joint frequency, not a natural evolution parameter.
- * Independence is an assumption ablation. More unpaired patients do not reveal linkage.

## G4_R2_joint_audit_LUAD

[PDF](../final_figures/G4_R2_joint_audit_LUAD.pdf) · [SVG](../final_figures/G4_R2_joint_audit_LUAD.svg) · [作图数据](plot_data/G4_R2_joint_audit_LUAD.tsv)

完整题注与说明（保留原统计措辞）：

- G4 | Joint information at equal assay cost · LUAD
- 500 random audits per cost; mean width ± 1.96 MCSE; inset Wilson intervals.
- Target: finite-cohort reported joint frequency, not a natural evolution parameter.
- * Independence is an assumption ablation. More unpaired patients do not reveal linkage.

## G4_R4_BC_patient_bootstrap

[PDF](../final_figures/G4_R4_BC_patient_bootstrap.pdf) · [SVG](../final_figures/G4_R4_BC_patient_bootstrap.svg) · [作图数据](plot_data/G4_R4_BC_patient_bootstrap.tsv)

完整题注与说明（保留原统计措辞）：

- G4B–C | Patient-resampling uncertainty
- 200 role-wise patient bootstraps per cell; complete three-design blocks shown.
- Every fit is rerun; dots: fixed every-tenth replicate; box/density: all valid blocks.
- Distributions combine patient resampling and one fresh perturbation per bootstrap.

## G4_R4_B_gain_COAD

[PDF](../final_figures/G4_R4_B_gain_COAD.pdf) · [SVG](../final_figures/G4_R4_B_gain_COAD.svg) · [作图数据](plot_data/G4_R4_B_gain_COAD.tsv)

完整题注与说明（保留原统计措辞）：

- G4B | Equal-cost bridge measurements · COAD
- Both arms: 638 gene assays; bridge n=80 versus routine expansion n=106.
- Completed paired seeds: 39/41/50 of 50/50/50. Caps: 1.96 seed-MCSE.
- Open points: fixed batches of 10 seeds; all completed seeds contribute.
- paired gain vs 0; conditional seed test; Holm | ns >= .05; * < .05; ** < .01; *** < .001

## G4_R4_B_gain_LUAD

[PDF](../final_figures/G4_R4_B_gain_LUAD.pdf) · [SVG](../final_figures/G4_R4_B_gain_LUAD.svg) · [作图数据](plot_data/G4_R4_B_gain_LUAD.tsv)

完整题注与说明（保留原统计措辞）：

- G4B | Equal-cost bridge measurements · LUAD
- Both arms: 638 gene assays; bridge n=80 versus routine expansion n=106.
- Completed paired seeds: 50/50/50 of 50/50/50. Caps: 1.96 seed-MCSE.
- Open points: fixed batches of 10 seeds; all completed seeds contribute.
- paired gain vs 0; conditional seed test; Holm | ns >= .05; * < .05; ** < .01; *** < .001

## G4_R4_C_calibration_COAD

[PDF](../final_figures/G4_R4_C_calibration_COAD.pdf) · [SVG](../final_figures/G4_R4_C_calibration_COAD.svg) · [作图数据](plot_data/G4_R4_C_calibration_COAD.tsv)

完整题注与说明（保留原统计措辞）：

- G4C | Hidden-event probability calibration · COAD
- Five probability bins; paired seeds 39/41/50; caps: 1.96 seed-MCSE.
- Base cost 560; each audit method pays 78 extra assays (total 638).
- Marginal-only can lower ECE while worsening proper scores; see score-control figure.
- vs base CTMC; conditional seed test; Holm | ns >= .05; * < .05; ** < .01; *** < .001

## G4_R4_C_calibration_LUAD

[PDF](../final_figures/G4_R4_C_calibration_LUAD.pdf) · [SVG](../final_figures/G4_R4_C_calibration_LUAD.svg) · [作图数据](plot_data/G4_R4_C_calibration_LUAD.tsv)

完整题注与说明（保留原统计措辞）：

- G4C | Hidden-event probability calibration · LUAD
- Five probability bins; paired seeds 50/50/50; caps: 1.96 seed-MCSE.
- Base cost 560; each audit method pays 78 extra assays (total 638).
- Marginal-only can lower ECE while worsening proper scores; see score-control figure.
- vs base CTMC; conditional seed test; Holm | ns >= .05; * < .05; ** < .01; *** < .001

## G4_R4_C_proper_score_controls

[PDF](../final_figures/G4_R4_C_proper_score_controls.pdf) · [SVG](../final_figures/G4_R4_C_proper_score_controls.svg) · [作图数据](plot_data/G4_R4_C_proper_score_controls.tsv)

完整题注与说明（保留原统计措辞）：

- G4C | Proper-score controls across all three designs
- Positive values mean better predictions; zero is the base CTMC (cost 560).
- Every candidate costs 638. Caps: seed-MCSE; open points: fixed batches of 10.
- Equal mean of all three designs; incomplete paired-seed blocks are reported in ledger.
- paired gain vs 0; complete seed blocks; Holm | ns >= .05; * < .05; ** < .01; *** < .001

## G4_R5_external_transfer

[PDF](../final_figures/G4_R5_external_transfer.pdf) · [SVG](../final_figures/G4_R5_external_transfer.svg) · [作图数据](plot_data/G4_R5_external_transfer.tsv)

完整题注与说明（保留原统计措辞）：

- G4 | Frozen GENIE models on external TCGA records
- Positive = candidate improves on named comparator; four retained B / baseline comparisons.
- Caps: 95% paired patient/seed bootstrap; points: 5 fixed draws of 2,000.
- GENIE fits frozen; no TCGA tuning. Recorded variants, not verified callable wild type.

## G4_R5_nested_patient_means

[PDF](../final_figures/G4_R5_nested_patient_means.pdf) · [SVG](../final_figures/G4_R5_nested_patient_means.svg) · [作图数据](plot_data/G4_R5_nested_patient_means.tsv)

完整题注与说明（保留原统计措辞）：

- G4 | Patient uncertainty of repeated-experiment means
- B: bridge vs more. 50 patient resamples × 8 seeds × 3 designs.
- Each value averages feasible seeds, then gives designs equal weight.
- Box/density: all valid outer blocks; dots: every fifth block. Capacity failures reported.

## G4_finite_calibration_ablation

[PDF](../final_figures/G4_finite_calibration_ablation.pdf) · [SVG](../final_figures/G4_finite_calibration_ablation.svg) · [作图数据](plot_data/G4_finite_calibration_ablation.tsv)

完整题注与说明（保留原统计措辞）：

- G4 | Value of finite source calibration
- 50 paired perturbations per cell; caps: 1.96 seed-MCSE; dots: fixed batches of 10.
- The same selected records; XS adds 80 source records (320 gene assays).
- Additional-information ablation, not equal-cost algorithm superiority.
- paired gain vs 0; conditional seed test; Holm | ns >= .05; * < .05; ** < .01; *** < .001

## G4_R6_attribution

[PDF](../final_figures/G4_R6_attribution.pdf) · [SVG](../final_figures/G4_R6_attribution.svg) · [作图数据](plot_data/G4_R6_attribution.tsv)

完整题注与说明（保留原统计措辞）：

- Total = gain after deassociation + pairing-dependent contribution (identical draws).
- Bars: equal-design means; intervals: paired patient bootstrap; dots: all three design means.
- Elbow labels: gain vs 0, 20-test Holm family. GENIE 449 / 287; TCGA 566 / 534 patients.

## G4_E1_multicohort_value

[PDF](../final_figures/G4_E1_multicohort_value.pdf) · [SVG](../final_figures/G4_E1_multicohort_value.svg) · [作图数据](plot_data/G4_E1_multicohort_value.tsv)

新来源中探索患者选择9个队列后，在其分开的1383名确认患者上评估冻结GENIE模型；不依据确认结果重筛。柱为三个预定观察合同等权的MORE−ALIGNED隐藏记录log-score增益，正值表示同成本补测改善预测。空心柱、3个固定合同均值点、患者与原冻结拟合抽样的成对95%bootstrap区间；点不代表额外患者或独立队列。原R4成本为初始560、追加78次gene assays；所有原COMPLETE拟合参与，容量失败与非收敛信息保存在研究资格记录。标注为对零的患者成对均值符号翻转检验（差值符号对称零假设），9项Holm族；直引线指向所检验单柱，不存在第二根比较柱。来源资格、4事件记录定义、原始成对损失及冻结选择见本轮报告。参考R29：空心柱、实际点、端帽与统计连接成分保留，9组代替3组；使用用户六色内的蓝与绿。

## G4_E1_attribution_LUAD

[PDF](../final_figures/G4_E1_attribution_LUAD.pdf) · [SVG](../final_figures/G4_E1_attribution_LUAD.svg) · [作图数据](plot_data/G4_E1_attribution_LUAD.tsv)

同一批探索预选的新确认患者，展示total=MORE−ALIGNED、deassociated=MORE−DEASSOCIATED、pairing=DEASSOCIATED−ALIGNED、vs pooled=POOLED−ALIGNED、vs two-step=TWO_STEP−ALIGNED。保持原测量成本、注册观察合同及拟合冻结；去关联拟合保持已测三基因边际、隐藏基因边际和质量。柱与3个固定合同均值点，区间为患者及原拟合抽样的成对95%区间，三个点不作独立患者检验。本图次级成分只展示区间，不增加未经定义的星号族。total逐点与bootstrap中精确等于deassociated+pairing；不同合同、不同队列的配对贡献可以正负不同。参考R21窄分组柱、空心点、端帽、顶部单行图例；显示全部5项强基线及归因，正负坐标完整。

## G4_E1_attribution_COAD

[PDF](../final_figures/G4_E1_attribution_COAD.pdf) · [SVG](../final_figures/G4_E1_attribution_COAD.svg) · [作图数据](plot_data/G4_E1_attribution_COAD.tsv)

同一批探索预选的新确认患者，展示total=MORE−ALIGNED、deassociated=MORE−DEASSOCIATED、pairing=DEASSOCIATED−ALIGNED、vs pooled=POOLED−ALIGNED、vs two-step=TWO_STEP−ALIGNED。保持原测量成本、注册观察合同及拟合冻结；去关联拟合保持已测三基因边际、隐藏基因边际和质量。柱与3个固定合同均值点，区间为患者及原拟合抽样的成对95%区间，三个点不作独立患者检验。本图次级成分只展示区间，不增加未经定义的星号族。total逐点与bootstrap中精确等于deassociated+pairing；不同合同、不同队列的配对贡献可以正负不同。参考R21窄分组柱、空心点、端帽、顶部单行图例；显示全部5项强基线及归因，正负坐标完整。

## G4_BIO_01_later_lineage

真实后期谱系标记的补测预算支持实验。ALL_NSCLC 确认集中14名患者具有真正复发/进展后的样本，其中9名有可评价的主肿瘤非主干标记簇参照；另外5名维持未知/无可评价参照，不冒充生物阴性，也不列入恢复率分母。显示开发数据冻结的 BEST_FIXED 经典采集策略，在额外预算10、14、20和新区获取开销2的预定合同下，实际后期样本支持的主肿瘤非主干谱系标记恢复率。三个均值为54.44%、54.44%、77.78%；空心柱、全部9个患者点、患者级2000次 bootstrap95%区间。每患者五次初始区域旋转先平均；九个点是九名患者，而不是额外区域、簇或重复。

原区窄测2、完整WES8和新区域开销均为冻结的假设成本单位，不是实际货币价格或前瞻性新采集。区域按冻结随机顺序开放；后期样本身份仅用于独立评分，未用于选择动作。标记簇至少有3个分支自有变异满足实际测量支持；该指标不证明迁移方向、克隆适应度或原CTMC theta。此图支持固定经典补测预算增长的生物桥接，不支持诊断策略超过最佳固定方案；没有为未注册预算比较添加星号。参考R29的空心柱、实际患者点、端帽及简洁轴线，使用规定配色、Arial、600dpi PDF+SVG。

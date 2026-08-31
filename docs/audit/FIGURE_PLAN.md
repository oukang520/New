# FIGURE_PLAN



## Main Figure Strategy



主文需要保留“真实横断面队列 -> 模拟真值 -> 真实纵向队列”的全链条。建议不要再把核心主图过度压缩成表格；表格服务于核心指标，图服务于机制与直观证据。



| figure | role | recommended_panels | source | message |
| --- | --- | --- | --- | --- |
| Figure 1 | 方法框架 | input state table -> MHN one-step graph -> F_hat denominator -> R* relative dwell map | core method code plus schematic, not a result-only plot | 把创新点定义清楚：R* = observed occupancy / progression-expected inflow, normalized. |
| Figure 2 | 真实横断面主队列输入与主输出 | E1/E2 data readiness, E4 inflow feasibility table/plot, E5/E10 high-R* summary | results/experiments_01_02; results/experiment_04_relative_inflow; results/experiment_05_state_scores; results/experiment_10_real_cohort_main | 三癌种真实队列可被统一处理并产生稳定的 state-level R* 结果。 |
| Figure 3 | 模拟真值验证 | E6 enhanced bottleneck table, E6 dwell-gradient compact statistical plot, E7 robustness panel | results/experiment_06_bottleneck_recovery_enhanced; results/experiment_06_dwell_gradient; results/experiment_07_topology_robustness_balanced | 在已知真值下 R* 能恢复相对停留时间强弱，且不是只在过易二分类中有效。 |
| Figure 4 | 真实生物学可解释性与非冗余性 | E8 biological convergence, E11 information gain, E14 denominator ablation, E15 falsification | results/experiment_08_biological_convergence; results/experiment_11_information_gain; results/experiment_14_ablation_backbone; results/experiment_15_uncertainty_negative_controls | 高 R* 状态既有生物学可解释性，也不是 raw occupancy/MHN-only 的简单重命名。 |
| Figure 5 | 真实数据拓扑直观展示 | E16 three-cohort real topology routes with R* overlay | results/experiment_16_real_topology | 从真实输入到方法处理再到带 R* 的进化路线，直接展示创新点。 |
| Figure 6 | 真实纵向外部验证 | E17 core metrics table + top-bottom persistence + minimum dwell proxy correlation | results/experiment_17_longitudinal_public | 高 R* 状态在纵向配对中更倾向保持，形成真实纵向方向性支持。 |



## Supplementary Strategy



| item | content | reason |
| --- | --- | --- |
| Supplementary Figure S1-S2 | E1/E2 detailed QC, oncoprint, sparsity and stage sensitivity. | 数据准备必要但不应抢占主文篇幅。 |
| Supplementary Figure S3 | E3 MHN interface details and one-step edge examples. | 说明接口正确性。 |
| Supplementary Table S1-SN | All per-state scores, edges, split repeats, bootstrap intervals, longitudinal pair-level outputs. | 保证可审查和可复现。 |



## Current Figure File Inventory



| result_root | png_total | pdf_total | single_png | figure_png |
| --- | --- | --- | --- | --- |
| results/experiment_03_mhn_interface | 30 | 30 | 9 | 12 |
| results/experiment_04_relative_inflow | 20 | 20 | 3 | 12 |
| results/experiment_05_state_scores | 17 | 17 | 2 | 12 |
| results/experiment_06_bottleneck_recovery_enhanced | 6 | 6 | 2 | 4 |
| results/experiment_06_dwell_gradient | 6 | 6 | 2 | 4 |
| results/experiment_07_topology_robustness | 1 | 1 | 0 | 1 |
| results/experiment_07_topology_robustness_balanced | 3 | 3 | 2 | 1 |
| results/experiment_08_biological_convergence | 11 | 11 | 2 | 9 |
| results/experiment_09_observation_enrichment | 6 | 6 | 2 | 4 |
| results/experiment_10_real_cohort_main | 8 | 8 | 2 | 6 |
| results/experiment_11_information_gain | 10 | 10 | 2 | 8 |
| results/experiment_12_clinical_validation | 6 | 6 | 2 | 4 |
| results/experiment_13_cross_cohort_replication | 7 | 7 | 2 | 5 |
| results/experiment_14_ablation_backbone | 6 | 6 | 2 | 4 |
| results/experiment_15_uncertainty_negative_controls | 4 | 4 | 2 | 2 |
| results/experiment_16_real_topology | 5 | 5 | 2 | 3 |
| results/experiment_17_longitudinal_public | 20 | 20 | 3 | 15 |
| results/experiments_01_02 | 64 | 64 | 11 | 45 |
| results/experiments_01_02 | 64 | 64 | 11 | 45 |



## Style Rules



- 使用当前公共配色模板：`#B5AED5`, `#B2E6FD`, `#B8D2CC`, `#E8B2A7`, `#FEEBB9`；辅助灰黑用中性灰，不新增竞争性主色。

- 主文单图优先接近正方形或紧凑矩形；组图后正文可读字号不低于期刊常见 6-8 pt 的最低线。

- 不在单图中保留 A/B/C/D panel title；panel 字母由后期组图统一添加。

- 表格用于精确指标和 CI；图用于机制、趋势、拓扑、分布。

- 所有图导出时保留足够边距，尤其网络节点、CI 端帽、右侧图例和顶部热力色带。

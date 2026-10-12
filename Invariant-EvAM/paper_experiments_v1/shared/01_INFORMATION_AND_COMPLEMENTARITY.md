# 实验执行书 1｜哪些队列组合增加进化信息

**项目** Invariant-EvAM  
**组编号** G1  
**角色** 你是负责生成模型、局部识别几何和观察设计的 Codex 科研代理。执行本书中的代码开发、实验、失败诊断和交付，不只复述方案。  
**任务类型** PROTOCOL ONLY；参数和重复数是本轮拟定执行约定，历史数值不是必须复现的目标。

## 0. 先读与执行边界

先读取同目录 `00_README_AND_SHARED_CONTRACT.md`。然后定位现有项目中的 `FINAL_INVARIANT_EVAM_REPORT_CN.md` 第 3–5、7、12 节，`INVARIANT_EVAM_PHASE2_REPORT_CN.md` 的模型、A–F、H 部分，以及 `PHASE2_THEORY.md`。核对实际代码、冻结配置和原始结果。缺少文件时记录缺失，不能假装读过报告链接指向的代码。

历史依据：报告曾在稀疏 3–5 event 模型中发现互补设计的 projected-rank rescue，同时明确 exact-shift 是参数几何构造、regular-local 不是 global identification。相关结果可作为回归测试，不能替代本轮的非精心构造 operator 验证 [S1 §4–7；S2 模型、A；S3]。

本组只负责**信息是否增加、增加什么、上限在哪里**。大规模 profile coverage 交给 G2；biology/observation 失配交给 G3；患者级直接扰动交给 G4；天然应用交给 G5。不要在本组重复生产四套相似的性能图。

允许在本组内连续自迭代，不需人工批准。非核心实现失败可以替换；没有发现增益也是可交付的科学结果，不能挑选成功设计代替整个设计库。

## 1. 核心问题、假设与反驳条件

核心问题：在独立患者总数或测量预算相同的条件下，新增一个环境提供了哪些单环境没有的、关于目标 interaction 的约束？

| 编号 | 待检验主张 | 可反驳它的结果 |
|---|---|---|
| G1-H1 | 部分预设的、可解释的 K1 observation families 会减少目标混淆方向 | 所有此类候选只重复原有信息，或增益仅在真值定制设计中存在 |
| G1-H2 | 环境数量与信息互补性不同 | 相同 operator 的重复竟被实现报告为新增结构 rank，提示实现或参数比较错误 |
| G1-H3 | Panel overlap 必须涉及目标所需信息，不能用整体基因重叠代替 | 只有总 overlap 数量与结果相关，未找到目标信息路径；需缩小机制解释 |
| G1-H4 | 观察操作不能创造源横断面分布本身不存在的信息 | 某实现声称纯状态选择/遮盖能区分本来相同的源分布，必须查生成流程或数据泄漏 |

不预设所有 interaction 都能识别，也不预设增益等于完整生成器恢复。观察机制差异只在其机制及知识条件下解释。

## 2. 输入与输出依赖

输入：现有 CTMC、operator、Jacobian 模块；事件与参数定义；本地真实 panel 定义仅用于建立候选覆盖结构，不读取最终算法效果挑选 panel。

新写入 `paper_experiments_v1/g1/`。本组负责发布下列 shared 资产，供 G2–G5 消费：

```text
shared/MODEL_CONTRACT.json
shared/event_dictionary.tsv
shared/parameter_dictionary.tsv
shared/target_registry.tsv
shared/operator_library.json
shared/design_registry.tsv
shared/numerical_precision_contract.json
```

每个资产有 schema、protocol version、SHA256。所有模型都显式标记 `topology_assumed`、`time_contract`、`source_initial_state`、`operator_knowledge`。不能把某个稀疏候选拓扑默认为所有癌症真实拓扑。

## 3. 生成模型与参数库

### 3.1 模型层级

1. `D3_SPARSE5`：复用已验证的 3-event、3 baselines + 2 interactions 模型，保持旧方向与索引。
2. `D3_FULL9`：3 baselines + 6 个有向 pairwise effects，包含反向竞争解释。
3. `D4_SPARSE10`：复用既有四事件、10 参数模型，但明确六条 interaction 的先验排除结构。
4. `D4_FULL16`：4 baselines + 12 directed pairwise effects，用于信息容量和拓扑依赖。
5. `D5_SPARSE10`：复用五事件稀疏模型进行扩展。
6. `D6_D7_EXTENSION`：只在数值资源允许时做稀疏扩展；不影响本组核心完成。

事件过程在状态空间不可逆，不要求基因间效应图是 DAG；禁止仅因事件只能累积，就自动禁止双向调节效应。

### 3.2 默认参数设计

- 固定时点 `t=1`，初态全零；float64。
- baseline 开发抽样区间 `[-1.8,-0.2]`；interaction magnitude 候选 `[0,0.15,0.5,1.0]`，混合正负与零。
- 另加少量稀有/高频压力场景，不把其排除后只展示中等频率。
- 每个核心模型预先生成 20 个开发参数点；确认参数库 60 个独立参数点。D5 默认确认 30 点。
- 这些范围是方法检验范围，不声称已校准为真实临床效应。若从旧配置继承不同范围，先做兼容说明并冻结，不能按性能修改。
- 开发与确认参数种子独立。采用来源状态支持或数值可计算性筛选时预设规则、记录全部失败；不能按 rank rescue 或模型胜负筛选。

## 4. 观察机制库

### 4.1 选择机制

主分析采用 `p_e(x) ∝ p_θ(x)exp(F_e(x)^Tφ_e)`。候选 F 在模拟开始前冻结：

- 单事件 indicator；
- 目标模块 burden `sum(x)/d`；
- 两个预设事件的 indicator 向量；
- 预设事件对的共同存在 indicator，作为较复杂选择压力场景。

默认强度包括 0、0.5、1.0，以及更强的 1.5 扩展。F 的尺度统一并保存。自由 φ 下的特征缩放不计入机制 diversity。

`EXACT_SHIFT_ANALYTIC` 单独标记为解析参照，允许定向构造盲方向，但不得并入“现实候选中成功比例”的分母。

### 4.2 测量机制

- full target panel；
- 冗余 panel；
- 不同但具有关键重叠的 panel；
- 有意移除关键联合状态信息的 failure panel。

小 d 可枚举非空 masks，按预设 panel 宽度/成本约束生成组合；大 d 从真实候选覆盖抽取。组合登记时只能读 mask、特征定义和成本，不能读确认参数、rank 或最终误差。

默认每个核心模型保留 24 个候选组合，覆盖 full/full、full/partial、partial/partial 与冗余组合；如可行组合少于 24，全部纳入。设计库使用中性 ID，`realized_complementarity` 是评价结果，不在事后重命名为预先设计成功。

### 4.3 知识条件

- K0：完整映射和权重已知。
- K1：panel 已知、选择特征族已知，φ 未知。
- K2 留给 G3；本组仅做标签故障的单元/回归检查。

已知权重、全覆盖、处处正权重的单环境，是信息上限参照，不应被误归为必然结构不识别。

## 5. 核心实验 G1A｜固定预算的四类比较

### 要做什么

对每个冻结参数点，以相同源过程比较 single、duplicated、candidate heterogeneous、cross-view failure。主要 N 为 `[400,1000,4000]`，E 为 `[1,2,4]`，分配主分析均衡，扩展 `[0.8,0.2]`。

人口层面用精确 p 和导数计算 H、rank、kernel 与目标 null leakage。有限样本只做解释性的 100 次重复/选定 cell，展示信息与估计波动的联系；不在本组声称正式 95% coverage。

### 预算规则

- 主比较固定独立入组患者数 N；记录有偏选择所需 screened 数。
- 次比较固定测量预算 `sum_e n_e * c(A_e) + calibration_cost`。
- 缺乏真实价格时，`c(A)` 用预设“患者×被测事件数”单位，仅称 proxy budget，不写真实临床成本。
- 平衡 panel 宽度和患者数后重新比较。不能把观察更多目标事件造成的收益全部写成 heterogeneity。

### 必须产生

`G1A_population_geometry.tsv`、`G1A_budget_matched.tsv`、少量可解释例子的参数路径与可观测分布。

## 6. 核心实验 G1B｜非真值定制的互补性

### 要做什么

在 G1 的开发期只用机制定义冻结候选库。确认期遍历所有候选与独立参数点。不能根据确认结果删掉“没有互补”的组合。

报告每种机制组合下：新增 rank 比例、目标 null leakage 变化、目标有效信息、数值不稳定比例、关键状态覆盖与 positivity。相同设计在不同参数点的结果单独保留。

给出两个明确分母：

1. 全部设计资格通过的候选组合；
2. 其中数值评估完成的候选组合。

数值失败和稀有状态不能消失于图表。按开发几何选出的典型强/弱/不识别场景可以供 G2 校准，但只称“条件代表场景”，不用于估计一般成功率。

### 不能做什么

不能在确认真值处搜索最佳 operator，随后报告其恢复效果为主结果。此类结果只能另设 `ORACLE_DESIGN_UPPER_BOUND`。

## 7. 核心实验 G1C｜assay 与选择信息的拆解

### 2×知识条件设计

分别比较：仅 panel 变化、仅选择变化、二者同时变化、二者都不变；每类覆盖 K0 与 K1。

关键正控制：在 full-panel、K0、w(x)>0 时，用精确概率直接计算 `p(x) ∝ p_e(x)/w_e(x)`，验证单环境已恢复源状态分布。这不代表 θ 必然可识别，随后仍检查 `θ→p` 的内在歧义。

关键负控制：同样全覆盖但 φ 未知时，计算 nuisance 引入的混淆；增加具有不同已知特征族的环境后，检验其是否实际解除目标歧义。

当 panel 覆盖不完整时，用边缘化 likelihood；不能对未观察 x 直接应用完整状态的反权重公式。

输出区分：

- 源分布可恢复性；
- 从源分布到生成器的可识别性；
- 有限样本精度收益。

三者分别命名，不能把第二项未成立的结果写成完整进化历史恢复。

## 8. 核心实验 G1D｜源信息上限与拓扑依赖

### 8.1 解析上限

验证并记录如下本模型直接推论：若 `p_θ1(x)=p_θ2(x)`，对相同的、仅依赖状态的 selection 与 measurement 操作，所有环境的观测分布仍相同。这适用于本组“固定源状态分布后选择/测量”流程，不覆盖真实提供新观察时间、轨迹或干预信息的不同实验。

完整观测 d 个二元事件最多有 `2^d-1` 个自由概率。D3_FULL9 有 9 参数，但源概率只有 7 个自由度；D4_FULL16 有 16 参数、15 个概率自由度。纯状态后处理的多环境不能越过这一局部维数上限。新增独立 nuisance 只会增加待区分量，不能创造源过程信息。

### 8.2 数值反例与参数路径

对稀疏/完整参数化逐一比较。在找到 rank deficiency 后，优先构造保留同一源分布的局部兼容路径或优化到相近分布的不同参数。保存最大概率差、KL、参数距离与目标变化。

多起点未发现 alias 仅记为诊断；找到明确 alias 可作为反例。有限数值精度不称全局证明。

### 8.3 functional 与拓扑约束

在增加反向边或多余候选边后重复原目标的检测。若原本可识别只依赖固定的排除结构，报告该依赖。不要靠 L1/L2 唯一解恢复识别声明。目标梯度与单点 kernel 正交时，仍沿兼容路径检查目标变化，避免一阶条件被误写为全局恒定。

## 9. 数值实现与验证要求

先实现或核对：

```text
build_generator(theta, topology)
state_distribution(theta, time_contract)
apply_selection_and_assay(p, operator)
observable_jacobians(theta, nuisance, operator)
profiled_information(jacobians, sample_weights)
functional_identifiability(target, information)
trace_compatible_path(target, model, operator)
```

Q 的行/列约定必须唯一。测试 off-diagonal 非负、守恒、分布归一化、无事件丢失转移。将解析/自动导数与多个步长的中心差分比较，保存最大绝对误差和相对误差；差分误差不能直接当识别零阈值。

投影使用 SVD/QR，避免直接求病态矩阵逆。nuisance 共享时对完整联合导数投影。参数尺度固定后比较奇异值，跨参数化仅作辅助，不将坐标相关的 σ_min 包装为坐标不变的新分数。

必测案例：

- independent 退化结果；
- duplicated operator 不新增行空间；
- K0 反权重概率恢复；
- source alias 经相同操作后仍 alias；
- missing 不等于 wild type；
- 同样预算不重复计算 N；
- 全参数不识别但个别 functional 可分；
- 数值容差改变时状态标记正确。

## 10. 默认实验配置

以下是供实现的配置，不是假设已经存在的 CLI：

```yaml
group: G1
protocol_version: "1.0"
seed_master: 20260923
models_core: [D3_SPARSE5, D3_FULL9, D4_SPARSE10, D4_FULL16]
models_extension: [D5_SPARSE10, D6_D7_EXTENSION]
parameter_draws:
  development: 20
  confirmation: 60
  d5_confirmation: 30
candidate_designs_per_model: 24
operator_knowledge: [K0_KNOWN, K1_STRUCTURED]
n_total: [400, 1000, 4000]
environments: [1, 2, 4]
finite_sample_illustrations:
  replicates: 100
  coverage_claim_allowed: false
rank_rtol: 1.0e-8
rank_rtol_sensitivity: [1.0e-6, 1.0e-7, 1.0e-9, 1.0e-10]
selection_rule_uses_confirmation_truth: false
budget_analyses: [fixed_enrolled_patients, fixed_measurement_proxy]
```

不要直接把该配置展开成全部组合的有限样本重复。人口几何广覆盖；有限样本只运行预注册 sentinel cells，G2 承担正式性能网格。

## 11. 作图数据与交付

输出 G1A–G1E 独立素材，最终拼版由用户完成：

- G1A：单观察中的混淆示例与兼容参数路径。
- G1B：四类观察设计、信息增量与预算。
- G1C：非真值定制设计库的成功/失败分布。
- G1D：仅 assay、仅 selection、二者结合的信息分解。
- G1E：稀疏/完整模型的信息上限、拓扑依赖。

每个 panel 保存 `data.tsv`、`plot.py`、`caption_notes.md`，记录主张、分母、知识条件和样本预算。报告必须说明多少结果来自 exact-shift，多少来自真实可解释 selection。

最终文件：

```text
g1/reports/G1_FINAL_REPORT_CN.md
g1/reports/G1_THEORY_AND_BOUNDARIES.md
g1/reports/G1_DESIGN_ELIGIBILITY.md
g1/reports/RUN_COMMANDS.md
g1/results/G1_population_geometry.tsv
g1/results/G1_design_yield.tsv
g1/results/G1_source_information_cap.tsv
g1/results/G1_target_map.tsv
g1/results/G1_counterexamples.jsonl
g1/figure_data/G1A ... G1E
```

## 12. 完成判据与自主决策

工程完成要求：模型/导数校验通过；四个核心实验已运行或有明确无法完成原因；完整设计分母保留；共享接口可被 G2 直接读取；原始结果可重跑。

科学判据：信息增加必须来自预设的有效条件，而非增加总测量、泄漏真值或误用正则化。若现实候选未带来增益，保留该结论；可以探索新机制族，但必须另标探索、冻结新确认种子，不能替换原失败。

若 full generator 无法识别而目标 functional 仍有信息，继续发布 functional 结果。若只有解析构造有效而现实候选无效，明确创新受到限制，不能以“理论已证明”覆盖实证缺口。

**现在执行。首先定位项目并核对旧结果，建立新 shared 合同；然后连续完成 G1A–G1D。不要在每个内部步骤后等待用户确认。**

# Invariant-EvAM 五组实验执行包

**版本** 1.0 · 2026-09-23  
**用途** 交给 Codex 的研究执行规范。五组是同一篇论文的证据分工，不是五个需要逐次请示的研究阶段。  
**状态** PROTOCOL ONLY。本执行包没有运行新实验，没有验证本地研究代码，也不把历史报告中的结论当作本轮验收答案。

## 1. 文件与执行方式

| 文件 | 负责的问题 | 主要交付 |
|---|---|---|
| `01_INFORMATION_AND_COMPLEMENTARITY.md` | 哪些队列组合增加进化信息 | 观察机制库、信息边界、设计资格与目标识别图谱 |
| `02_RELIABLE_INFERENCE_AND_BASELINES.md` | 新增信息能否转化为可靠推断 | 强基线、正式校准、作答收益与错误风险 |
| `03_UNKNOWN_OBSERVATION_AND_BIOLOGY.md` | 观察未知与生物差异如何限制结论 | 混淆实验、校准资料收益、联合敏感性与失配边界 |
| `04_REAL_PATIENT_CONTROLLED_VALIDATION.md` | 在原始患者记录上是否仍然有效 | 数据分母审计、直接遮盖/选择、隐藏记录评价 |
| `05_NATURAL_COHORT_AND_DATA_DESIGN.md` | 如何用于真实队列解释与数据补充 | 天然应用、留出检验、关系分级、预算约束设计 |

推荐把整个目录放到现有 Invariant-EvAM 项目内。先让 Codex 读取本文件，再读取对应执行书。默认顺序 G1 → G2 → G3 → G4 → G5；G4 的数据审计可以与 G1 并行，G5 的元数据审计可以提前进行。完成一个内部任务后继续执行，不等待人工批准。不能跳过依赖输出的核验。

可直接给 Codex 的总启动指令：

> 读取 `00_README_AND_SHARED_CONTRACT.md` 和五份实验执行书，定位并审计现有 Invariant-EvAM 项目。按照共享协议连续完成五组实验，不把工作拆成需要我批准的阶段。保留原始项目和冻结结果，在新命名空间内实现、执行、诊断并迭代。先完成依赖项，再消费其已冻结输出。非核心工程或数据失败可以替换并记录；核心反例必须保留，不能通过换数据消除。全部实验从真实运行产物生成报告与组图素材，不生成虚构结果，不强求本方法胜出。若实际资源中断，保存可续跑清单和未完成项目，不宣布已闭环。

## 2. 资料来源与证据标签

本包直接继承用户确认的最终五组重设计，依据下列已提供报告。报告仅说明此前报告过的结果；运行日志、原始结果、配置与代码仍需在用户本机核验。

- **S1** `FINAL_INVARIANT_EVAM_REPORT_CN.md`。重点为第 3–5 节的方法/理论边界，第 6、8、9 节的数据与验证，第 10–13 节的基线和限制。
- **S2** `INVARIANT_EVAM_PHASE2_REPORT_CN.md`。重点为模型与 estimand、exact-shift 的定位、局部识别和有限重复校准。
- **S3** `PHASE2_THEORY.md`。提供正则局部理论、functional 的一阶条件和偏移不可分性。
- **S4** `PHASE1_PROJECT_AUDIT.md`、`PHASE1_NOVELTY_AUDIT.md`。定位本地资源及与 IQE/oMHN 的边界。
- **S5** 用户最后确认的查重后五组方案。决定本包分工，不回退到早期 Phase 3 清单。

所有实验条目都属于**新增执行约定**，其中参数、重复数、软件接口是拟定的默认配置，不是文献结论或已经实现的功能。继承实验必须标记 `LEGACY_REPORTED`，核验真实产物后改为 `LEGACY_VERIFIED`。本轮新结果标记 `NEW_RUN`。已被多轮查看的数据标为 `PREVIOUSLY_EXPLORED`，不得因重新随机切分改称独立外部确认。

用于方法边界核对的主要外部来源：

- **R1** Schill et al. *Correcting for Observation Bias in Cancer Progression Modeling*. Journal of Computational Biology, 2024, 31(10):927–945. DOI `10.1089/cmb.2024.0666`。观察偏差建模是直接先行工作。
- **R2** Morris, White, Crowther. *Using simulation studies to evaluate statistical methods*. Statistics in Medicine, 2019. DOI `10.1002/sim.8086`；作者预印本 `arXiv:1712.03198`。用于 ADEMP 与 Monte Carlo 误差报告。
- **R3** Raue et al. *Structural and practical identifiability analysis of partially observed dynamical models by exploiting the profile likelihood*. Bioinformatics, 2009. DOI `10.1093/bioinformatics/btp358`。profile/实验规划不是本项目首创。

执行时优先复用本地已核查文献表。新增文献引用前读取原始论文或官方实现，记录版本；不得把一般 rank、profile、biased-sample fusion 或实验设计思想改称独立原创。来源之外的推导标记 `DERIVED_FOR_THIS_PROTOCOL`。

## 3. 固定科学问题和推断对象

固定问题：多个异质观察环境在什么条件下增加关于同一潜在癌症积累过程的信息，哪些 interaction/functionals 可以回答，哪些只能部分回答或不能回答。

主模型使用不可逆二元事件状态 `x ∈ {0,1}^d`，只允许未发生事件被获得：

\[
\lambda_i(x)=\exp\{\theta_{ii}+\sum_{j\ne i}x_j\theta_{i\leftarrow j}\},\quad x_i=0.
\]

全零初始状态、固定时间 `t=1` 是主分析的建模约定，不代表真实癌症的绝对时间单位。观察时间不同的基线必须进入独立的条件匹配子实验。若同时估计整体速率与时间尺度，先处理尺度任意性，不能把任意归一化解释为绝对进化时间。

主要目标 `I_a(θ)=a^Tθ`，优先为预设 interaction log-hazard、两条 interaction 的差；`exp(θ_{i←j})` 是该模型内的获得率倍数。图中 `j→i` 不表示因果机制、克隆祖先关系或时间上的必然先后。事件含义、索引顺序、θ 的行列方向必须写入 `event_dictionary.tsv` 和 `parameter_dictionary.tsv`，跨组不得变动。

环境生物差异统一写为 `θ_e=θ*+δ_e, Q_e=Q(θ_e)`。共享对象选择参考总体参数，或使用固定目标权重 `sum_e w_e δ_e=0` 的参数中心；必须注明这不是平均生成器。主分析默认参考总体参数，并在组 3 配置中显式指定 reference。增加/删除环境不应静默改变 estimand。天然分析保留 reference 环境的验证与真正留出 reference 的分析分开。

## 4. 观察生成流程必须明确

主分析为先选择、后测量：

\[
p_e(y)=\sum_x A_e(y\mid x)\frac{p_{\theta_e}(x)w_e(x;\phi_e)}{Z_e},\qquad Z_e=\sum_xp_{\theta_e}(x)w_e(x;\phi_e).
\]

`A_e` 必须对每个潜在状态正规化。缺失事件由求和边缘化处理，不能填 0。正值权重 `w` 是相对选择权重，不自动等同于入组概率。模拟拒绝抽样时显式使用 `s_e(x)=c_e w_e(x)≤1`，记录候选人数、入组人数和校准成本。

若使用 `w_e=exp(F_e^T φ_e)`，截距在条件入组分布中通常被归一化消去，不能声称仅凭入组患者估计了总体入组率。特征函数和截距处理必须统一。单纯缩放特征且让 φ 自由调整可能只是重参数化，不作为实质鲁棒性。

若采用随进化发生的临床观察事件或竞争风险过程，单独建立其生成器和 likelihood，不将其当作固定时点后的权重操作。来自不同观察时间条件的原始 KL/RMSE 不得混排。

知识等级统一：

- `K0_KNOWN`：测量映射和选择权重已知，作为参考上限。
- `K1_STRUCTURED`：测量映射已知，选择只知道特征族；参数需要联合估计或有限校准资料。此等级是主要方法场景。
- `K2_MISSPECIFIED`：拟合遗漏特征或使用错误映射；作为适用边界，不混入机制正确的平均成绩。

给所有合理基线提供同样的可用信息。故意不使用某类信息的方法标记为消融或信息受限对照。

## 5. 识别、区间与敏感性不得混用

设 `p_e` 是实际可观测类别的概率向量，`Jθ,e` 和 `Jφ,e` 是对应导数。`ρ_e=n_e/N`，独立环境 nuisance 情形的每样本投影 Jacobian 为：

\[
H=\operatorname{stack}_e\left[\sqrt{\rho_e}(I-P_e)\operatorname{diag}(p_e^{-1/2})J_{\theta,e}\right],
\quad \mathcal I_{\mathrm{eff}}=N H^T H.
\]

`P_e` 投影到 whitened nuisance 列空间。不要先乘 `sqrt(n_e)` 又再乘 `N`，造成样本量重复加权。存在共享 nuisance、生物偏移约束、校准 likelihood 或同患者重复观测时，从**完整联合 likelihood**建立整体 nuisance 投影，不能机械逐环境相减。

默认 float64、SVD 相对容差 `1e-8`，同时检查 `1e-6,1e-7,1e-9,1e-10`。这些是数值规则，不是识别定理。结论随容差不稳定时标为 `NUMERICAL_RANK_UNRESOLVED`。只去掉理论上不可能的观测类别，不能去掉有限样本中计数为 0、但模型概率为正的类别。

functional 的一阶检查是 `a` 是否与 `ker(H)` 正交；默认同时保存归一化 null leakage 与其容差敏感性。单点正交不自动等于全局 functional 不变，需局部兼容路径和多个近最优解诊断。整体参数秩亏不自动表示每条 interaction 不识别。

区间标签至少分为：`REGULAR_LR_VALIDATED`、`LOCAL_PROFILE_EXPLORATORY`、`CONSERVATIVE_REFERENCE`、`SENSITIVITY_ENVELOPE`、`BOUND_TRUNCATED`。结构不识别、有限样本弱信息、数值求解失败、模型失配必须有不同状态。区间触边界不假装有限；无限宽度不截断成好看的均值。

## 6. 共享数据与代码命名空间

定位本机实际项目，不能假定聊天中的 `/mnt/data` 或旧报告中的 `/Users/...` 路径在用户本机仍存在。只审计项目目录与用户授权的数据目录，不扫描无关私人文件。优先复用本地合法数据；下载仅限公开或已授权资源，不绕过登录、不自动购买云资源、不上传患者级数据。

原始报告和旧结果只读。建议新建：

```text
paper_experiments_v1/
  shared/                 # 冻结协议、字典、设计库、split、基线合同
  g1/ g2/ g3/ g4/ g5/      # 各组 configs/src/tests/runs/results/reports/figure_data
  manifests/
  reviewer_audit/
```

建议的统一实验接口由 Codex 实现或映射到已有实现：`audit`、`freeze`、`run --resume`、`summarize`、`export-figure-data`、`verify`。这是待实现接口，不宣称项目已经存在这些命令。每组交付实际测试通过的 `RUN_COMMANDS.md`，不能只交伪命令。

跨组依赖如下：

- G1 发布 `MODEL_CONTRACT.json`、`operator_library.json`、`target_registry.tsv`、`design_registry.tsv`。
- G2 消费上述文件，发布 `method_registry.tsv`、`inference_contract.json`、`calibration_summary.tsv`。
- G3 发布 `sensitivity_contract.json`、`calibration_budget_registry.tsv`、`misspecification_library.json`。
- G4 发布去标识化 `DATA_AUDIT.md`、`split_manifest`、仅供评估器读取的 reference、冻结 benchmark。
- G5 消费 G1–G4 已验证接口；设计选择只读 development，评价只读被隔离的 holdout。

若依赖尚未产生，可自主完成最小依赖实现；其结果仍记入所属组，避免重复生产多个不一致版本。多 agent 并行使用独立目录/分支，单一维护者更新 shared；不能多个 agent 覆盖同一冻结配置。

## 7. 冻结、随机化和计算预算

种子主值建议为 `20260923`，通过 `SeedSequence([master, group, purpose, cell, parameter_draw, replicate])` 派生。purpose 对应开发、正式评价、校准、bootstrap；不得共用 RNG 流。方法比较使用相同生成数据形成配对比较。真值文件、隐藏基因型、选择机制真实参数只供 generator/scorer 访问，估计器不得读取。

小规模开发只检查实现、可计算性和机制覆盖，不能调到本方法取胜。冻结后失败的 cell 不更换随机种子、不删除。若方法有实质改动，创建新的 protocol/method version，并用新的评价随机流；旧失败保留。

配置中的 core 表为预定核心实验，extension 为扩展，不对全部变量作巨大笛卡尔积。先进行运行成本探测并记录，不向用户承诺完成时长。硬件预算只允许改变并行、缓存、非核心网格与重复精度，不能改变科学主张。若核心重复数因资源不足未完成，标记 `INCOMPLETE_CALIBRATION`，不以 8 次或 20 次结果代替 500 次的验收。

处理非核心故障可以修复、替换或绕过，但都留痕。不能用“仍在探索”隐藏停滞；连续失败应换算法/求解方法并解释。任务中断写可续跑状态，不声称在无人调用时继续后台科研。

## 8. 统计评价和失败记录

每个实验按 aims、DGM、estimand、methods、performance measures 明确配置 [R2]。覆盖率或错误率报告分子、分母和二项区间；Monte Carlo 标准误 `sqrt(p_hat*(1-p_hat)/R)`。零次错误须给上界，不写错误概率为零。500 次且 p≈0.95 时标准误约 0.0097，这是精度设计计算。

不同算法在同一 replicate 上配对比较；参数点、重复模拟、bootstrap 患者三个层级不可混算。真实记录的多次遮盖不是新增患者。bootstrap 使用患者作为独立单位，重复测量必须保留聚类。

同时报告总调用数、拟合成功数、区间完成数、条件覆盖率、有效作答率和失败率。求解失败不计为科学 unknown，不伪造覆盖；报告失败对端到端工作流的影响。方法返回无限区间时保留无限值和比例。所有有限平均宽度都明确条件分母。

主验收没有“必须提高 20%”“必须 rank 10/10”“必须有四条稳定边”。科学状态可以是支持、条件支持、不支持或未解决。工程完成与科学结论方向分离。

## 9. 最低结果接口

每条目标结果至少含：

```text
protocol_version, group_id, experiment_id, cell_id, method_id,
model_contract_hash, operator_id, operator_knowledge, topology_id,
target_id, evidence_type, truth_available, n_unique_patients,
n_observed, n_screened, n_calibration, budget_basis, budget_value,
seed, parameter_draw_id, replicate_id, split_id, estimate,
ci_lower, ci_upper, interval_type, coverage_defined, covers_truth,
structural_status, null_leakage, rank, sigma_min_per_sample,
sign_call, computational_status, compatibility_status,
boundary_hit, runtime_seconds, peak_memory_mb, reason
```

缺失用 `null/NA` 并说明；不能以 0 代替不可用的真值。原始指标建议 Parquet/TSV，元数据 JSON。另存 `FAILURE_LEDGER.tsv`、`PROTOCOL_DEVIATIONS.md`、`CLAIM_EVIDENCE.tsv`。

最终 scientific status 至少区分 `SUPPORTED_UNDER_CONDITIONS`、`NOT_SUPPORTED`、`UNRESOLVED`；run status 区分 `COMPLETE`、`PARTIAL`、`BLOCKED`。没有通过科学假设不妨碍如实完成研究；没有运行则不能称完整验证。

## 10. 组图素材和最后交付

每组自行生成**独立小图与作图数据**，不拼成最终五张版图。每个 panel 有唯一 ID（G1A 等），保存数据、脚本、配置、适用条件和一句话证据，不预填“胜出”标题。必要时导出矢量 SVG/PDF；文字保持可编辑、Arial 优先、白底。这里是给 Codex 的输出要求，本执行包本身不包含新实验图。

所有报告使用中文；公式与代码术语统一。主文只保留核心结果，详细实现与完整参数表进入补充。不要重写英文论文，不用期刊评分代替科学验收。

最后交付五份 `Gx_FINAL_REPORT_CN.md`、五套 figure_data、完整原始结果、可复现命令、软件环境与哈希、共同 claim 表和一份 `FIVE_GROUP_INTEGRATION_REPORT_CN.md`。集成报告明确新增了什么，哪些旧结论仍成立，哪些被削弱，哪些未完成。

## 11. 配套模板

`START_CODEX.txt` 可直接复制给 Codex。`templates/` 提供空白 run/claim/failure/data/figure 表、建议默认配置、结果 schema 与源报告哈希。模板没有填入任何实验结果，也没有创建伪造的研究状态。

# METHOD_CODE_MAPPING



本表给出方法概念、数学含义、实现位置和实验使用位置。代码位置以当前仓库审计为准。



| concept | definition | code | experiment_use | audit |
| --- | --- | --- | --- | --- |
| State/genotype representation | stage::gene1+gene2 state plus canonical binary event vector. | src/relobstq_mhn/core/states.py:122 build_state_occupancy | E1/E2 data tables; E4-E5 state scoring; E16 route display; E17 longitudinal states. | Confirmed from code. |
| MHN one-step event addition probability | Softmax over absent event log-rates derived from theta diagonal and context terms. | src/relobstq_mhn/core/transitions.py:21 softmax_addition_probabilities | E3 interface; E4 inflow; E5/E10/E16 scoring and display. | Confirmed from code. |
| Same-stage one-step predecessor graph | Edges from u to v when v adds exactly one event and stage is unchanged. | src/relobstq_mhn/core/transitions.py:49 same_stage_one_step_edges | E4 inflow rule, E16 real topology paths. | Confirmed from code. |
| Expected inflow F_hat | Sum over predecessor occupancy times edge-addition probability. | src/relobstq_mhn/core/transitions.py:106 aggregate_inflow | E4, E5, E10, E11, E14, E15, E16. | Confirmed from code. |
| Relative dwell index R* | R_raw=L/(F_hat+epsilon), median-normalized among eligible states. | src/relobstq_mhn/core/scoring.py:21 compute_relative_dwell | All core method validation experiments E5-E8, E10-E17. | Primary novelty; confirmed from code. |
| Observation enrichment O* | Observed occupancy over progression-only expected occupancy. | src/relobstq_mhn/core/scoring.py:65 compute_observation_enrichment | E5/E9/E10 as auxiliary analysis. | Confirmed; secondary metric. |
| Bootstrap uncertainty | Multinomial resampling of state counts and recomputation of F_hat/R*. | src/relobstq_mhn/core/bootstrap.py:24 bootstrap_relative_dwell | E5, E6 variants, E17 intervals. | Confirmed from code. |
| Integrated scoring pipeline | State occupancy + theta + event set -> edges, F_hat, R*, optional bootstrap. | src/relobstq_mhn/core/pipeline.py:20 score_states_from_mhn | Reusable method package for experiments. | Confirmed from code. |
| Dominant predecessor topology route | Backtrack through dominant predecessor to display a representative inflow-supported path. | src/relobstq_mhn/core/topology.py:22 build_dominant_predecessor_path | E16 real topology and E17 route figures. | Confirmed; descriptive route, not full phylogenetic tree. |
| Experiment-ready data construction | Metadata, event matrix, state table and MHN matrix construction. | src/relobstq_mhn/data/processing.py:220 build_experiment_ready_tables | Data processing package; raw-to-analysis interface. | Confirmed from code. |
| Simulation generator | cMHN-like trajectory simulation with state dwell multipliers and audit outputs. | src/relobstq_mhn/simulation/generator.py:183 simulate_cohort_with_audit | E6, E6-gradient, E7, E9. | Confirmed; should be described as controlled positive-control simulator. |



## Minimal Computational Contract



- 输入：样本级二值事件矩阵、stage/metastasis 分组、状态表、MHN theta 或可替代的转移概率 backbone。

- 输出：`state`, `N_v`, `L_v`, `F_hat`, `R_raw`, `R_star`, `log2_R_star`, eligibility/high-confidence flags, predecessor/edge tables, bootstrap intervals where requested.

- 必须披露：事件筛选、最小状态数、F_hat eligibility、是否使用真实 MHN backend、是否使用 fallback backbone。

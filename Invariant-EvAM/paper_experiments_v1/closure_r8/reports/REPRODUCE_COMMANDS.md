# 实际执行的复现命令

工作目录：`/Users/oukang/Desktop/投稿资料/fangxiang2/Invariant-EvAM`。

开发和评价使用不同种子命名空间。冻结配置中的代码哈希必须匹配。脚本会跳过已有逐样本JSON；完整从零复现请复制代码、配置与依赖至新工作目录，不要删除当前原始结果。评价命令也可用于重建聚合表。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python paper_experiments_v1/closure_r8/src/run_g1.py development --workers 3
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python paper_experiments_v1/closure_r8/src/run_g1.py evaluation --workers 3
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python paper_experiments_v1/closure_r8/src/run_g2.py development --workers 3
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python paper_experiments_v1/closure_r8/src/run_g2.py evaluation --workers 3
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python paper_experiments_v1/closure_r8/src/run_g5.py development --workers 3
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python paper_experiments_v1/closure_r8/src/run_g5.py evaluation --workers 3
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python paper_experiments_v1/closure_r8/src/check_exact.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python paper_experiments_v1/closure_r8/src/refine_profile.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python paper_experiments_v1/closure_r8/src/g5_matched_size.py
```

`refine_profile.py`在原1800个评价记录生成后完成连续端点重算；逐条输出在g2/revision_r8/profile_refinement。它保留旧粗网格结果作数值对照，不混写旧结果。

G5同样本量敏感性输出在g5/revision_r8/matched_size。主要原始输出分别位于g1/g2/g5的revision_r8/runs。

经典对照的进一步数值修复与新种子确认（均已执行）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python paper_experiments_v1/closure_r8/src/refine_profile_v2.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python paper_experiments_v1/closure_r8/src/g2_fresh_confirmation.py
```

V2对全部1800条记录统一使用预设对称多起点；若约束拟合优于无约束拟合，重新求无约束极值后再解端点，最多4轮。V1原始诊断保留于profile_refinement，V2位于profile_refinement_v2。新增独立确认使用种子命名空间3，200条记录位于fresh_confirmation，不混入原1800条分母。

完成计算后的汇总及数值诊断命令：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python paper_experiments_v1/closure_r8/src/audit_solver.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python paper_experiments_v1/closure_r8/src/summarize.py
```

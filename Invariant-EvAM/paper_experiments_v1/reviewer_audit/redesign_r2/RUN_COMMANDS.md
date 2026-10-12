# R2 实际执行与复现命令

工作目录：`/Users/oukang/Desktop/投稿资料/fangxiang2`。全部使用既有 `Invariant-EvAM/.venv/bin/python`。以下命令已实际执行；中途纠错、未成功检查也留在日志和修正记录，不冒称全部一次通过。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 Invariant-EvAM/.venv/bin/python Invariant-EvAM/paper_experiments_v1/reviewer_audit/redesign_r2/run_simulations.py g1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 Invariant-EvAM/.venv/bin/python Invariant-EvAM/paper_experiments_v1/reviewer_audit/redesign_r2/run_g1_finite.py
OPENBLAS_NUM_THREADS=1 Invariant-EvAM/.venv/bin/python Invariant-EvAM/paper_experiments_v1/reviewer_audit/redesign_r2/test_exact.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 Invariant-EvAM/.venv/bin/python Invariant-EvAM/paper_experiments_v1/reviewer_audit/redesign_r2/run_simulations.py g2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 Invariant-EvAM/.venv/bin/python Invariant-EvAM/paper_experiments_v1/reviewer_audit/redesign_r2/run_simulations.py g3
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 Invariant-EvAM/.venv/bin/python Invariant-EvAM/paper_experiments_v1/reviewer_audit/redesign_r2/run_real.py g4 --probe
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 Invariant-EvAM/.venv/bin/python Invariant-EvAM/paper_experiments_v1/reviewer_audit/redesign_r2/run_real.py g4 --workers 4
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 Invariant-EvAM/.venv/bin/python Invariant-EvAM/paper_experiments_v1/reviewer_audit/redesign_r2/run_real.py g5 --workers 2
OPENBLAS_NUM_THREADS=1 Invariant-EvAM/.venv/bin/python Invariant-EvAM/paper_experiments_v1/reviewer_audit/redesign_r2/summarize_real.py
OPENBLAS_NUM_THREADS=1 Invariant-EvAM/.venv/bin/python Invariant-EvAM/paper_experiments_v1/reviewer_audit/redesign_r2/g4_joint_record_audit.py
OPENBLAS_NUM_THREADS=1 Invariant-EvAM/.venv/bin/python Invariant-EvAM/paper_experiments_v1/reviewer_audit/redesign_r2/plot_r2.py 1 2 3
OPENBLAS_NUM_THREADS=1 Invariant-EvAM/.venv/bin/python Invariant-EvAM/paper_experiments_v1/reviewer_audit/redesign_r2/plot_r2.py 3 4 5
OPENBLAS_NUM_THREADS=1 Invariant-EvAM/.venv/bin/python Invariant-EvAM/paper_experiments_v1/reviewer_audit/redesign_r2/plot_r2.py 4
Invariant-EvAM/.venv/bin/python Invariant-EvAM/paper_experiments_v1/reviewer_audit/redesign_r2/report_r2.py
OPENBLAS_NUM_THREADS=1 Invariant-EvAM/.venv/bin/python Invariant-EvAM/paper_experiments_v1/reviewer_audit/redesign_r2/verify_r2.py
OPENBLAS_NUM_THREADS=1 Invariant-EvAM/.venv/bin/python Invariant-EvAM/paper_experiments_v1/reviewer_audit/redesign_r2/derive_paired.py
```

PDF渲染：对g=1,2,3,4,5分别实际执行如下命令，其中g4在新增两图后重跑：

```bash
swift Invariant-EvAM/paper_experiments_v1/g4/src/render_pdf.swift Invariant-EvAM/paper_experiments_v1/g4/revision_r2/figures /tmp/invariant_r2_pdf_g4 > Invariant-EvAM/paper_experiments_v1/g4/revision_r2/reports/PDF_RENDER_CHECKS.json
```

G1有限样本和G4/G5逐任务JSON支持缓存续跑。若要从头复现，应先归档revision_r2/runs中的相应JSON，再按冻结协议运行，不能在旧原始结果目录执行清理。G4错误人数版400个bootstrap JSON已保存在revision_r2/archive/bootstrap_fixedN_bug；纠正版是当前runs数据。G3仅做等价PMF加速后全部重跑，原日志保留。旧G1–G5科学配置、代码和结果未覆盖。

脚本代码集中在共同reviewer_audit/redesign_r2以避免五套相互不一致的实现；原始产物、报告和图仍分组进入各g*/revision_r2。不得将本轮自然记录重复扰动改称新外部数据。

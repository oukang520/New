# R3 实际运行与复现命令

工作目录：`/Users/oukang/Desktop/投稿资料/fangxiang2`。以下脚本已实际执行；阶段和更改见CHANGE_LOG_CN.md。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 Invariant-EvAM/.venv/bin/python Invariant-EvAM/paper_experiments_v1/reviewer_audit/redesign_r3/g3_joint.py development
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 Invariant-EvAM/.venv/bin/python Invariant-EvAM/paper_experiments_v1/reviewer_audit/redesign_r3/g3_joint.py evaluation
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 Invariant-EvAM/.venv/bin/python Invariant-EvAM/paper_experiments_v1/reviewer_audit/redesign_r3/g4_regularized.py development --probe --workers 1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 Invariant-EvAM/.venv/bin/python Invariant-EvAM/paper_experiments_v1/reviewer_audit/redesign_r3/g4_regularized.py development --workers 4
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 Invariant-EvAM/.venv/bin/python Invariant-EvAM/paper_experiments_v1/reviewer_audit/redesign_r3/g4_regularized.py evaluation --workers 4
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 Invariant-EvAM/.venv/bin/python Invariant-EvAM/paper_experiments_v1/reviewer_audit/redesign_r3/g5_natural_transfer.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 Invariant-EvAM/.venv/bin/python Invariant-EvAM/paper_experiments_v1/reviewer_audit/redesign_r3/g5_interval.py development --probe --workers 1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 Invariant-EvAM/.venv/bin/python Invariant-EvAM/paper_experiments_v1/reviewer_audit/redesign_r3/g5_interval.py development --workers 4
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 Invariant-EvAM/.venv/bin/python Invariant-EvAM/paper_experiments_v1/reviewer_audit/redesign_r3/g5_interval.py evaluation --workers 4
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 Invariant-EvAM/.venv/bin/python Invariant-EvAM/paper_experiments_v1/reviewer_audit/redesign_r3/g3_reference_anchor.py development --workers 2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 Invariant-EvAM/.venv/bin/python Invariant-EvAM/paper_experiments_v1/reviewer_audit/redesign_r3/g3_reference_anchor.py evaluation --workers 4
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 Invariant-EvAM/.venv/bin/python Invariant-EvAM/paper_experiments_v1/reviewer_audit/redesign_r3/g4_calibrate.py development --workers 2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 Invariant-EvAM/.venv/bin/python Invariant-EvAM/paper_experiments_v1/reviewer_audit/redesign_r3/g4_calibrate.py evaluation --workers 2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 Invariant-EvAM/.venv/bin/python Invariant-EvAM/paper_experiments_v1/reviewer_audit/redesign_r3/g5_anchor_interval.py development --probe --workers 1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 Invariant-EvAM/.venv/bin/python Invariant-EvAM/paper_experiments_v1/reviewer_audit/redesign_r3/g5_anchor_interval.py development --workers 4
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 Invariant-EvAM/.venv/bin/python Invariant-EvAM/paper_experiments_v1/reviewer_audit/redesign_r3/g5_anchor_baseline.py --freeze-only
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 Invariant-EvAM/.venv/bin/python Invariant-EvAM/paper_experiments_v1/reviewer_audit/redesign_r3/g5_anchor_interval.py evaluation --workers 4
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 Invariant-EvAM/.venv/bin/python Invariant-EvAM/paper_experiments_v1/reviewer_audit/redesign_r3/g4_full_calibration.py development --workers 4
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 Invariant-EvAM/.venv/bin/python Invariant-EvAM/paper_experiments_v1/reviewer_audit/redesign_r3/g4_full_calibration.py evaluation --workers 4
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 Invariant-EvAM/.venv/bin/python Invariant-EvAM/paper_experiments_v1/reviewer_audit/redesign_r3/g5_anchor_baseline.py --workers 4
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 Invariant-EvAM/.venv/bin/python Invariant-EvAM/paper_experiments_v1/reviewer_audit/redesign_r3/g5_baseline_classic.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 Invariant-EvAM/.venv/bin/python Invariant-EvAM/paper_experiments_v1/reviewer_audit/redesign_r3/plot_r3.py 3 4 5
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 Invariant-EvAM/.venv/bin/python Invariant-EvAM/paper_experiments_v1/reviewer_audit/redesign_r3/plot_r3.py 3 4
```

逐任务JSON采用缓存。若需要从零复现，应先归档各revision_r3/runs相关JSON，不能删除原冻结结果。g5_baseline_classic.py必须在g5_anchor_baseline.py之后执行，以生成两种强基线区间。源数据保持只读。

最终派生与检查（已实际执行）：

```bash
Invariant-EvAM/.venv/bin/python Invariant-EvAM/paper_experiments_v1/reviewer_audit/redesign_r3/report_r3.py
OPENBLAS_NUM_THREADS=1 Invariant-EvAM/.venv/bin/python Invariant-EvAM/paper_experiments_v1/reviewer_audit/redesign_r3/plot_r3.py 4
OPENBLAS_NUM_THREADS=1 Invariant-EvAM/.venv/bin/python Invariant-EvAM/paper_experiments_v1/reviewer_audit/redesign_r3/verify_r3.py
swift Invariant-EvAM/paper_experiments_v1/g4/src/render_pdf.swift Invariant-EvAM/paper_experiments_v1/g3/revision_r3/figures /tmp/invariant_r3_pdf_g3 > Invariant-EvAM/paper_experiments_v1/g3/revision_r3/reports/PDF_RENDER_CHECKS.json
swift Invariant-EvAM/paper_experiments_v1/g4/src/render_pdf.swift Invariant-EvAM/paper_experiments_v1/g4/revision_r3/figures /tmp/invariant_r3_pdf_g4 > Invariant-EvAM/paper_experiments_v1/g4/revision_r3/reports/PDF_RENDER_CHECKS.json
swift Invariant-EvAM/paper_experiments_v1/g4/src/render_pdf.swift Invariant-EvAM/paper_experiments_v1/g5/revision_r3/figures /tmp/invariant_r3_pdf_g5 > Invariant-EvAM/paper_experiments_v1/g5/revision_r3/reports/PDF_RENDER_CHECKS.json
```

报告生成时的布尔配对差值类型错误已修复后重跑；不影响任何原始实验结果。G4全部校准设计的审计人数元数据也已核对修正，公开估计输入及方法结果哈希保持不变，详见G4_FULLCAL_METADATA_CORRECTION.json。

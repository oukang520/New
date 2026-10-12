# 本轮实际执行命令

工作目录：`/Users/oukang/Desktop/投稿资料/fangxiang2/Invariant-EvAM`。
环境沿用项目 `.venv`，未安装新依赖。以下科学计算已实际执行；runs缓存存在时读取对应冻结结果，不清空原始记录。科学计算子进程均设置 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python paper_experiments_v1/closure_r6/src/g5_explore.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python paper_experiments_v1/closure_r6/src/g4_attribution.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python paper_experiments_v1/closure_r6/src/g5_diagnostic_value.py calibration --workers 2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python paper_experiments_v1/closure_r6/src/g5_diagnostic_value.py evaluation --workers 2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python paper_experiments_v1/closure_r6/src/g4_transport_check.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python paper_experiments_v1/closure_r6/src/g4_transport_difference.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python paper_experiments_v1/closure_r6/src/summarize_g5.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python paper_experiments_v1/closure_r6/src/verify.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python paper_experiments_v1/closure_r6/src/plot.py g4 g5
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python paper_experiments_v1/closure_r6/src/publish.py
swift paper_experiments_v1/g4/src/render_pdf.swift /tmp/evam_r6_pdf /tmp/evam_r6_review
```

绘图、患者内外部不确定性脚本为图面修订/补充矩阵导出执行过多次；科学方案和G5冻结评价未因此调整。`publish.py`为交付更新脚本，不用于重算实验；历史快照在research_traceability，正式结果入口为各组final_results。

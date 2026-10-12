# R5 实际执行与复现命令

工作目录：`/Users/oukang/Desktop/投稿资料/fangxiang2/Invariant-EvAM`。Python `.venv/bin/python`（3.9.6）。现有环境含numpy/scipy/pandas/matplotlib；本轮补装requests用于外部数据请求。字体Arial，PDF文本嵌入、SVG可编辑。环境版本见 reports/ENVIRONMENT.txt。

下列科学计算命令本轮实际执行过；日志在 `closure_r5/reports/`。开发必须先于评价。原始 runs 缓存存在时命令会读取缓存；完整重算应复制整个项目至新复现目录，清空副本的 R5 runs 后执行，不删除原始冻结结果。R4拟合和原始患者数据是依赖，不能用TCGA重新训练替代。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python paper_experiments_v1/closure_r5/src/g5_closed_loop.py development --workers 2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python paper_experiments_v1/closure_r5/src/g5_closed_loop.py evaluation --workers 2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python paper_experiments_v1/closure_r5/src/g4_nested.py --workers 2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python paper_experiments_v1/closure_r5/src/external_validation.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python paper_experiments_v1/closure_r5/src/summarize.py g4
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python paper_experiments_v1/closure_r5/src/summarize.py g5
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python paper_experiments_v1/closure_r5/src/verify.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python paper_experiments_v1/closure_r5/src/audit_geometry.py
.venv/bin/python paper_experiments_v1/closure_r5/src/write_reports.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python paper_experiments_v1/closure_r5/src/plot.py g4 g5
.venv/bin/python paper_experiments_v1/figure_delivery/finalize_audit.py
```

本轮外部下载先执行 `download_external.py`，DataHub元数据成功，但突变LFS下载404；之后以官方API POST请求取得目标四基因记录。实际请求URL、JSON body、状态码和字节SHA保存在 `g4/revision_r5/data_external/API_RECEIPTS.json`。API请求最初通过Python heredoc执行，现抽出 `fetch_target_mutations.py` 保留同一请求逻辑；该脚本的缓存验证已实际执行。`--refresh`下载复现路径未经再次联网重跑，若上游字节变化会停止而不覆盖冻结数据。

```bash
.venv/bin/python paper_experiments_v1/closure_r5/src/download_external.py
.venv/bin/python paper_experiments_v1/closure_r5/src/fetch_target_mutations.py
```

实际PDF检查使用Mac系统PDFKit，`/tmp/evam_r5_selected`为五个新增PDF的临时软链接目录，渲染后人工目视对照本地参考截图。临时PNG不属于最终图交付。

```bash
swift paper_experiments_v1/g4/src/render_pdf.swift /tmp/evam_r5_selected /tmp/evam_r5_pdf_review > paper_experiments_v1/closure_r5/reports/PDF_RENDER_CHECK.json
```

科学输出没有按有利结果筛选。绘图显示的稀疏点采用固定索引，所有样本仍进入均值、区间、密度和检验。原52张最终图104文件逐项哈希不变；新5张均写入对应组唯一 final_figures。

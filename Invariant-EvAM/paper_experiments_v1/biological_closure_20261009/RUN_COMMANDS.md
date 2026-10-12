# 实际执行及复现命令

根代理命令的工作目录为 `/Users/oukang/Desktop/投稿资料/fangxiang2`：

```bash
Invariant-EvAM/.venv/bin/python -m pip install openpyxl scikit-learn lifelines
Invariant-EvAM/.venv/bin/python Invariant-EvAM/paper_experiments_v1/biological_closure_20261009/src/freeze_protocol.py
Invariant-EvAM/.venv/bin/python Invariant-EvAM/paper_experiments_v1/biological_closure_20261009/src/verify_inputs.py
Invariant-EvAM/.venv/bin/python Invariant-EvAM/paper_experiments_v1/biological_closure_20261009/src/summarize_acquisition.py
Invariant-EvAM/.venv/bin/python Invariant-EvAM/paper_experiments_v1/biological_closure_20261009/src/score_metastatic.py
```

freeze_protocol 是已执行的首次冻结命令，已有协议时故意拒绝覆盖；后续复现使用现有冻结文件。输入导出gzip含时间戳，不能无声覆盖哈希冻结输入。

采集代理实际命令（工作目录 Invariant-EvAM）：

```bash
.venv/bin/python paper_experiments_v1/biological_closure_20261009/src/acquisition.py --phase development --output-dir paper_experiments_v1/biological_closure_20261009/results/acquisition > paper_experiments_v1/biological_closure_20261009/acquisition_development.log 2>&1
.venv/bin/python paper_experiments_v1/biological_closure_20261009/src/acquisition.py --phase confirmation --model paper_experiments_v1/biological_closure_20261009/results/acquisition/FROZEN_MODELS.json --output-dir paper_experiments_v1/biological_closure_20261009/results/acquisition > paper_experiments_v1/biological_closure_20261009/acquisition_confirmation.log 2>&1
```

原生适配器实际命令（工作目录 Invariant-EvAM）：

```bash
.venv/bin/python paper_experiments_v1/biological_closure_20261009/src/audit_native.py
/Users/oukang/Desktop/投稿资料/IQE/iqe_project/.venv/bin/python paper_experiments_v1/biological_closure_20261009/src/audit_native.py --metastasis-only --decode-metastasis-trees --metastasis paper_experiments_v1/biological_closure_20261009/data/raw/tracerx_metastasis_2023/processed
.venv/bin/python paper_experiments_v1/biological_closure_20261009/src/verify_acquisition_independent.py
```

血浆代理实际命令：

```text
实际执行过的复现命令（从Invariant-EvAM/paper_experiments_v1/biological_closure_20261009执行）
../../.venv/bin/python src/audit_ctdna.py
../../.venv/bin/python src/audit_ctdna.py --audit-only
../../.venv/bin/python src/audit_ctdna.py --metadata-only
../../.venv/bin/python -m py_compile src/run_ctdna.py
../../.venv/bin/python src/run_ctdna.py --run > ctdna_replay_run.log 2>&1

注意：run_ctdna.py验证全部冻结输入哈希，使用现有processed输入和CTDNA_ASSAY_FREEZE；重新完整导出xlsx到gzip会因gzip时间戳产生不同字节哈希，请勿覆盖已冻结输入。首次trunk排除实现不合规运行在未检视评分前隔离，内部implementation_checks保留原因；有效results仅保留按冻结trunk+branch规则执行的版本。

追加实际执行命令
../../.venv/bin/python src/score_ctdna_future.py > ctdna_future_scorer_run.log 2>&1
```

下面补充已完成的报告、绘图与发布命令。


## 本轮报告、绘图与正式发布：实际执行

报告/发布工作目录为 `/Users/oukang/Desktop/投稿资料/fangxiang2`：

```bash
Invariant-EvAM/.venv/bin/python Invariant-EvAM/paper_experiments_v1/biological_closure_20261009/src/build_reports.py
Invariant-EvAM/.venv/bin/python Invariant-EvAM/paper_experiments_v1/biological_closure_20261009/src/publish.py
```

绘图工作目录为 `Invariant-EvAM`，实际执行：

```bash
.venv/bin/python paper_experiments_v1/biological_closure_20261009/src/plot_biological.py
```

导出后的PDF以PyMuPDF实际渲染，SVG以独立临时Chrome profile实际渲染；人工比较R29/R04/R21原图，检查标签、端帽、终点、字体和留白。审查预览和当前哈希位于review，未放入正式图目录。

这些命令使用已冻结输入；重新生成原始gzip、重新冻结协议或重新下载覆写输入不是本轮发布复现的必要步骤。实验来源下载、输入和软件版本由各SOURCE/receipt及configs/SOFTWARE_FREEZE.txt记录。报告保持未支持结论；不通过重复搜索种子改变结果。


## 独立发布核验与交付封存

独立核验实际工作目录为 `Invariant-EvAM/paper_experiments_v1/biological_closure_20261009`：

```bash
../../.venv/bin/python src/verify_release_bio.py > review/verify_release_bio_run.log 2>&1
```

退出码0，883项检查、0失败；验证器没有修改正式目录或冻结输入。

交付封存从 `/Users/oukang/Desktop/投稿资料/fangxiang2` 执行：

```bash
Invariant-EvAM/.venv/bin/python Invariant-EvAM/paper_experiments_v1/biological_closure_20261009/src/finalize_delivery.py
```

封存只汇集本轮实际结果、报告与命令的哈希，不再拟合、改选方法、患者或随机种子。DELIVERY_CHECK.json是本轮交付校验入口。

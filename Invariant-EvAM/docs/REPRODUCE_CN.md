# 安装、核验与实验复现

以下命令均从本仓库根目录执行。整理和烟雾检查使用Python3.9、NumPy2.0.2、SciPy1.13.1；完整环境记录在provenance/ENVIRONMENT_LOCK.txt。G2使用SciPy1.13的内部L-BFGS-B接口，因此该SciPy版本是实际兼容要求。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -B scripts/verify_release.py
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
```

`verify_release.py`只读核对五组清单、59张图、118个文件及来源映射，不重新产生科研数据或改图。数值测试检查CTMC概率/解析导数、投影秩和公共API；它们不等于重新跑完全部实验。

## 原始有效实验命令

R8原始运行记录见[实际命令](../paper_experiments_v1/closure_r8/reports/REPRODUCE_COMMANDS.md)。当前生物桥接实验命令见[实际记录](../paper_experiments_v1/biological_closure_20261009/RUN_COMMANDS.md)。这些记录描述已执行的研究运行；其中的本机绝对路径是历史来源字段，不能作为其他机器已安装数据的保证。

全量重跑应先把本仓库复制到独立计算目录，然后按数据依赖说明恢复输入。不要在Git发布目录运行旧publisher、freeze或清理脚本来覆盖正式结果。R8模拟可从冻结配置运行；脚本要求development/runs/results输出目录存在，先运行`scripts/prepare_run_dirs.py`创建空输出目录。它不删除任何文件。

## 本地C后端

G2、G3、G5依赖小型C动态库。本包保留当时使用的三个Mach-O x86_64冻结二进制，以追踪原运行；它们不适用于所有操作系统或CPU。G3/G5同时提供原C源，可由`python scripts/build_native.py --build-g3-g5`在本机编译，编译后的文件属于新计算工件，不是原二进制字节。

G2加载的文件名为ctmc_fast_v3.dylib，但原项目没有对应v3.c。现存v2/v4源码明确保留；v4不是经过证明的原v3源码，因此本包不把重编v4冒称原实验精确重现。`build_native.py --build-g2-v4`只生成单独v4后端，不改g2_engine.py也不覆盖冻结v3。完整跨平台G2重现需先解决该缺失；当前正式结果、Python代码、原v3字节和核心纯Python数值接口均完整保留。

## 数据与图

公开的ZIP、原始大表、RDS/FASTQ及个人绘图参考截图未放入Git；下载网址、原哈希及输入角色见数据依赖。现有图使用Arial、六色配色、PDF+SVG、600dpi导出设置。重绘需合法可用的Arial和外部参考图库；本次整理不转换字体或修改图件。

G2原始汇总脚本另写Parquet文件，运行该脚本需安装`python -m pip install -e ".[parquet]"`；本包的正式TSV作图数据和只读核验不需要PyArrow。官方oMHN对照的外部工具版本见`.[official-mhn]`，R数据读取见`.[research]`。这些可选依赖没有作为环境或vendor文件夹复制。

## 当前脚本的精确中间输入

依据59张正式图的静态溯源，补入了G1几何/兼容参数、G3修订汇总与重复级KL、G5预算评价与R3基线等当前有效数值输入；其用途与原哈希见`docs/FIGURE_TO_CODE.json`的`additional_current_panel_inputs`。保留原路径是为匹配冻结脚本的读取接口。它们是当前正式实验的计算输入，各组`final_figures`仍是唯一正式图目录；未复制历史图或整个旧结果目录。

有一份宽表`g4/results/G4_all_method_results.tsv`同时含已撤回终点，未纳入上传包。该表原路径、用途和哈希列于`provenance/EXTERNAL_DATA_REGISTRY.json`的`omitted_broad_runtime_tables`。G4当前有限校准消融的正式图和数值完整保留在`final_results`；运行原`repair_coverage.py`从宽表重新汇总，需在隔离计算目录通过`g4_pipeline.py`重生成或从原工程恢复该精确宽表。它不属于当前正式结果发布目录。

`data_private/reference_records`是原工程对公开队列的处理记录目录，名称沿用原接口，不表示本包含私人原始测序数据；数据继续适用原来源条款。

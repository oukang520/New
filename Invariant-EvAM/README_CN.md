# Invariant-EvAM：G1–G5 当前科研发布包

本文件夹用于上传Git。正式结果、图及其作图数据来自当前发布版；核心数值代码、有效实验入口和冻结配置保留原相对结构。打包没有重新拟合、筛患者、筛种子或改变科研结论。

完整科研故事、核心方法、五章实验逻辑闭环、创新点、研究意义与重点发现见[中文科研总说明](docs/RESEARCH_STORY_CN.md)，另有[可编辑Word版](docs/RESEARCH_STORY_CN.docx)。两版内容按当前保留的正式结果整理，包含证据来源链接。

| 章节 | 核心问题 | 当前单图数 | 正式报告 |
|---|---|---:|---|
| G1 | 信息是否足够、何时互补 | 8 | [报告](paper_experiments_v1/g1/final_results/REPORT_CN.md) |
| G2 | 信息能否成为可靠答案 | 9 | [报告](paper_experiments_v1/g2/final_results/REPORT_CN.md) |
| G3 | 未知观察与生物差异如何影响结论 | 10 | [报告](paper_experiments_v1/g3/final_results/REPORT_CN.md) |
| G4 | 真实记录中补测是否有效、为何有效 | 18 | [报告](paper_experiments_v1/g4/final_results/REPORT_CN.md) |
| G5 | 下一步收集什么数据 | 14 | [报告](paper_experiments_v1/g5/final_results/REPORT_CN.md) |

共59张独立单图，每张同时有PDF与SVG。各章的`final_figures`是唯一图目录，`final_results/plot_data`是对应作图数据，`final_results/experiment_data`保存当前有效实验数据。根目录`src`提供核心API；各章实验脚本、R2/R3与R8等有效分析代码位于`paper_experiments_v1`，目录名称记录其实际来源，并非重新加入旧版结果。

近期三张生物桥接图完整保留。50个固定均衡标记在10名配对患者中恢复11/20个后期组织支持目标，与完整面板逐患者相同；这仍是小样本计数回放。真实数据的诊断独有优势、临床收益和新致癌机制未被证明，相关强基线和范围限制保留在报告中。

G2九张正式图的实验计算、汇总与作图对应见[G2代码表](docs/G2_CODE_MAP_CN.md)；所有图的入口见[逐图清单](docs/FIGURE_TO_CODE.tsv)。

本次实际检查见[检查记录](docs/PACKAGING_CHECKS_CN.md)。环境安装、实验入口与外部数据恢复见[复现说明](docs/REPRODUCE_CN.md)、[代码入口](docs/CORE_CODE_INDEX_CN.md)、[数据依赖](docs/DATA_DEPENDENCIES_CN.md)。G3实际协议位于其configs及有效R2/R3配置；它原本没有METHOD.json，本包不补造方法文件。

原始研究文件未改动。文件来源、原SHA256、包内SHA256及仅为整理而做的链接变化见[来源映射](provenance/SOURCE_MAP.tsv)。被排除的外部/历史链接见[链接说明](docs/LINKS_CN.md)。第三方代码和来源数据不因进入本包而获得新的许可证，见[第三方说明](THIRD_PARTY_NOTICES.md)。

本项目以独立目录`Invariant-EvAM/`发布于GitHub仓库[oukang520/New](https://github.com/oukang520/New/tree/main/Invariant-EvAM)，与仓库原项目隔离。复现命令从本项目目录执行。原本地发布包的Git初始化与检查记录见检查说明。

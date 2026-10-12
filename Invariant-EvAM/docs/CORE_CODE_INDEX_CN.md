# 正式实验与核心代码索引

以各组当前`final_figures`与`final_results/MANIFEST.json`为准，共59张单图：G1=8、G2=9、G3=10、G4=18、G5=14。源项目保持原状，发布包排除了根目录早期continuous/phase2结果表和对应旧作图入口。共有721个来源映射文件。

[逐图代码表](FIGURE_TO_CODE.tsv)列出实验计算、汇总与作图入口，[详细清单](FIGURE_TO_CODE.json)保存查证范围及备注。G2九张均已确切定位，见[G2逐图说明](G2_CODE_MAP_CN.md)。当前已确切查证59张，没有未定位的候选项；这里的确切定位仅指静态代码和数据链条，未重跑研究实验。全部59张作图入口、正式图和作图数据均已定位并收录。

| 层级 | 包内代码 |
|---|---|
| 公共数值API | [invariant_evam](../src/invariant_evam)、[phase2](../src/phase2)、[minimal_model.py](../src/minimal_model.py) |
| 原有效G1–G5模块 | [G1](../paper_experiments_v1/g1/src)、[G2](../paper_experiments_v1/g2/src)、[G3](../paper_experiments_v1/g3/src)、[G4](../paper_experiments_v1/g4/src)、[G5](../paper_experiments_v1/g5/src) |
| 有效R2/R3实验 | [redesign_r2](../paper_experiments_v1/reviewer_audit/redesign_r2)、[redesign_r3](../paper_experiments_v1/reviewer_audit/redesign_r3) |
| G4真实扰动与归因 | [R4](../paper_experiments_v1/g4/revision_r4/src)、[R5](../paper_experiments_v1/closure_r5/src)、[R6](../paper_experiments_v1/closure_r6/src) |
| G1/G2/G5当前R8 | [closure_r8/src](../paper_experiments_v1/closure_r8/src) |
| 外部队列确认 | [cohort_expansion](../paper_experiments_v1/cohort_expansion_20261009/src) |
| 生物桥接与追踪 | [biological_closure](../paper_experiments_v1/biological_closure_20261009/src) |
| 原始TRACERx解析器 | [parse_tracerx_native.R](../data_adapters/parse_tracerx_native.R) |

目录里的R2/R3等名字记录有效源代码的实际来源。旧图、旧探索结果和退休发布脚本不随这些目录重新纳入。保留的数值与实验源代码逐文件保持原字节；来源及包内哈希见[SOURCE_MAP.tsv](../provenance/SOURCE_MAP.tsv)，静态依赖审计见[CORE_CODE_AUDIT.json](../provenance/CORE_CODE_AUDIT.json)。

安装、实际运行命令、数值检查以及G2原C源与外置数据的复现范围见[复现说明](REPRODUCE_CN.md)。参考截图、字体、原始大数据、环境和vendor均不纳入本包。

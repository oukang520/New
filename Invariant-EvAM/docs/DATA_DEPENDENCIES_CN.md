本次仅准备本地 Git 发布包的依赖清单，未推送、未声明公开、未复制外部原始数据或参考截图，也未改正式结果。包内可恢复的来源、SHA256与建议位置见[EXTERNAL_DATA_REGISTRY.json](../provenance/EXTERNAL_DATA_REGISTRY.json)。逐文件大小和个人路径审计保存在包外构建目录。

**正式发布内容**

五组当前 `final_figures` 与 `final_results` 共330文件、86.18 MiB；59 panel 的118个 PDF/SVG 均在此范围。最大单文件12.89 MiB，没有100 MiB以上文件。各组体积：G1 3.93、G2 2.62、G3 20.42、G4 8.58、G5 50.63 MiB。正式图、绘图数据和有效派生实验数据应按现有字节保留，含已冻结 `final_results/experiment_data/BIO`。这些数据沿用原来源条款，本项目不另授第三方数据的使用权。

不纳入包的内容：外部原始 GENIE/TCGA/cBioPortal 输入，TRACERx 原 ZIP/RDS/FST/xlsx/html/docx/PDF，BIO `data/` 下完整逐marker处理输入，参考截图、字体二进制、旧探索与已隔离的不合规实现运行。正式派生结果不能因目录名称含患者ID或marker就被删除；原输入与项目实验输出应按实际来源区分。

**恢复完整真实数据实验所需输入**

| 输入 | 官方入口或下载 | 原始大小与用途 |
|---|---|---|
| TRACERx421 primary | [官方记录7822002](https://zenodo.org/records/7822002)；[figurecode.zip](https://zenodo.org/api/records/7822002/files/figurecode.zip/content) | 178354403 bytes，170.09 MiB；生成421组织、区域与primary-only树审计输入 |
| TRACERx metastasis | [官方记录7649257](https://zenodo.org/records/7649257)；[metsFigures.zip](https://zenodo.org/api/records/7649257/files/metsFigures.zip/content) | 312573110 bytes，298.09 MiB；未来组织、corrected-tree坐标映射与seeding scorer |
| TRACERx ctDNA | [Nature原论文及公开补充表](https://www.nature.com/articles/s41586-023-05776-4)；[Supplementary Tables1–21 xlsx](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-023-05776-4/MediaObjects/41586_2023_5776_MOESM4_ESM.xlsx) | 48335320 bytes，46.10 MiB；Table17计数级回放、Table20正常来源、Table8/13临床与样本对应 |
| GENIE18.0-public | [AACR官方数据入口](https://www.aacr.org/professionals/research/aacr-project-genie/access-project-genie-data/) | 原突变表877861971 bytes，837.19 MiB；另需临床patient/sample、gene matrix、assay/genomic信息和meta文件，完整哈希见原 `g4/configs/DATA_FREEZE.json` |
| cBioPortal独立确认队列 | [官方Datahub](https://github.com/cBioPortal/datahub) | 采用下列9队列；按原cohort_expansion下载收据与哈希恢复，不能把当前API/master内容直接冒充冻结版 |

9个已采用确认队列：`luad_oncosg_2020`、`luad_cas_2020`、`luad_cptac_2020`、`luad_broad`、`lung_nci_2022`、`coad_silu_2022`、`coadread_dfci_2016`、`coadread_genentech`、`coadread_cass_2020`。本清单不把其余5个未采用候选伪装为正式确认结果。原 G4 TCGA外部评价使用 `luad_tcga_pan_can_atlas_2018` 与 `coadread_tcga_pan_can_atlas_2018`。精确临床/assay文件URL和SHA在 `g4/revision_r5/data_external/DOWNLOAD_RECEIPTS.json`；目标突变使用 `API_RECEIPTS.json` 的成功POST及body（原MAF链接404，未伪装成完整MAF）。这两个 `data_external` 原快照也不入包，只登记来源/hash，建议位置 `external_data/tcga_reference/`。GENIE训练/保留角色沿用冻结注册；只重画正式图无需下载这些原输入。

原始SHA256：

- `figurecode.zip`：`dc060458d0fb477f4f7c96acbc30b76c2785b926cf72eab0daa56a4c04253d5e`。
- `metsFigures.zip`：`bacb4f310fbffb1caaf6bb63525a7343c2450c4bb6bb2c77712aca0dbd03c058`。
- ctDNA补充xlsx：`ae36a332bfca17164c57ef45a36a8cf1107614c15862181b83c639c0f4b48154`。
- GENIE突变表：`f85c1ab8ce4b1ca45edc820052ace4ee5889e94a9cad7212dcbcaa9defdb657a`。

TRACERx421原下载manifest与[转移数据官方记录](https://zenodo.org/records/7649257)标注“academic non-commercial use only”。这里仅记录来源条款，不把开放下载解释为任意转载授权，也不把代码与第三方数据统一标为MIT/CC0。此说明不自动阻止按原条款准备研究用本地Git包；没有证据表明特定已冻结final_results派生文件被明确禁止科研再分发，故本次保留其原字节并注明沿用来源。

公开ctDNA输入是处理后UMI计数与Poisson P，不是FASTQ。受控EGA原reads未被下载；无需为本轮计数回放申请或打包原测序reads。公共病例字段为CRUK/GENIE等研究伪名ID和样本键，表头扫描未发现姓名、出生日期、地址、电话、SSN等直接身份字段；这不是对数据全面匿名性的额外承诺。

**保持哈希并实现可移植运行**

当前审查的25个文本文件含292处 `/Users/`，疑似credential扫描0命中。个人机器位置主要位于BIO冻结protocol、model/execution记录、各组figure registry和旧绘图provenance；它们不是认证凭证。明确的正式冻结副本包括 G4/G5 `final_results/experiment_data/BIO/configs/PROTOCOL.json` 以及G5 acquisition的 `MODEL_FREEZE.json`、`EXECUTION_MANIFEST.json`，不得为改路径而重写其字节并沿用旧SHA。

建议在发布包提供一份单独的external-data registry：按文件名+原SHA映射至 `external_data/tracerx421/`、`external_data/tracerx_metastasis/`、`external_data/tracerx_ctdna/`、`external_data/genie18/`、`external_data/cbioportal/`。运行wrapper以registry或环境变量解析位置，冻结protocol内原绝对位置只作历史provenance。科学配置、随机种子、患者split和输入/结果原SHA不变。

非冻结的当前目录索引若在目标包改成仓库相对路径，必须计算新的“打包元数据SHA”，另记原元数据SHA；不能声称改写后的JSON仍具有原字节哈希。源码也应区分原冻结分析源码与新增位置解析wrapper，避免修改科学算法后伪装为相同代码hash。

完整raw-to-native恢复还需要本项目旧自有解析器 `parse_tracerx_native.R`（SHA `4fa67cdbecf0bbb7116a90b6a40501a918f5253d4778512d9f930394fb1c67f5`）、R环境与fst支持。已按原字节收录到[data_adapters/parse_tracerx_native.R](../data_adapters/parse_tracerx_native.R)，来源与SHA记录在[SOURCE_MAP.tsv](../provenance/SOURCE_MAP.tsv)；R及fst依赖需另行安装。Arial用于重绘，安装位置应由字体环境解决，字体二进制不放Git。参考截图只保留文件名/原SHA及R编号，不转载图片，不虚构其网络URL；它们用于风格审查，不是科学输入。

本次未执行任何恢复下载或数据重生成。仅阅读现有文件与官方来源说明，写出这两份清单。

# G4 分母与 callability 审计

本轮 NEW_RUN；来源 GENIE 18.0-public、GRCh37、MSK-IMPACT468、Primary。原始来源只读，临床患者 manifest 211,526 人、样本 manifest 250,018 份。七个有来源 MD5 的输入全部与 manifest 一致；九个输入及审计元数据另存 SHA256。实际测量层级仅 nominal panel/gene-level linkage，不是逐样本 callable coverage。

| cancer | stage | n_samples | n_patients |
| --- | --- | --- | --- |
| LUAD | cancer_manifest | 27148 | 22777 |
| LUAD | primary | 13714 | 12310 |
| LUAD | uniform_panel | 2506 | 2242 |
| LUAD | one_lexicographic_sample_per_patient | 2242 | 2242 |
| COAD | cancer_manifest | 12693 | 12141 |
| COAD | primary | 7759 | 7633 |
| COAD | uniform_panel | 1448 | 1429 |
| COAD | one_lexicographic_sample_per_patient | 1429 | 1429 |
| IDC | cancer_manifest | 10964 | 10177 |
| IDC | primary | 6702 | 6490 |
| IDC | uniform_panel | 1951 | 1833 |
| IDC | one_lexicographic_sample_per_patient | 1833 | 1833 |
| PAAD | cancer_manifest | 9273 | 8915 |
| PAAD | primary | 4941 | 4878 |
| PAAD | uniform_panel | 1066 | 1054 |
| PAAD | one_lexicographic_sample_per_patient | 1054 | 1054 |

## 资格先于效果

| cancer | n_manifest_deduplicated | n_proxy_complete | n_no_mutation_rows | n_strict_callable_complete | state_support | eligible |
| --- | --- | --- | --- | --- | --- | --- |
| LUAD | 2242 | 2242 | 77 | 0 | 16 | CONDITIONAL_PROXY_RECORD |
| COAD | 1429 | 1429 | 10 | 0 | 16 | CONDITIONAL_PROXY_RECORD |
| IDC | 1833 | 1833 | 81 | 0 | 12 | INELIGIBLE |
| PAAD | 1054 | 1054 | 37 | 0 | 16 | INELIGIBLE |

阈值沿用 n≥700、四事件频率均在0.02–0.85、至少8个支持状态。IDC/PAAD仍不合格，不替换模块来凑成功来源。频率及原因完整保存在 ledger；不利用恢复效果作资格筛选。LUAD和COAD所有16态均有来源记录支持。

旧报告的2,165/1,419与本轮2,242/1,429之差，分别就是77/10份无任何mutation行的记录，不是新增版本患者。较早IQE的LUAD2,197样本属于不同筛选口径，未用来填充本轮。

样本以PATIENT_ID、SAMPLE_ID词典顺序选一份Primary IMPACT468样本；不依据目标基因或拟合效果。原始ID仅用于本地连接，导出患者key为SHA256。跨癌种有2名患者，角色相同，未重复计算任一模块的独立N。

## 事件与三通道

四事件顺序：LUAD TP53/KRAS/STK11/KEAP1；COAD APC/TP53/KRAS/PIK3CA。状态按MSB在前编码；记录 mutation / measurement_status / missing_reason。无测量参考应为−1并排除隐藏真值评分，不能填零。本轮合格模块没有这类排除，故partial/unknown计数为0。

非同义分类在DATA_FREEZE中枚举；同患者同事件重复变异合并为1。不纳入CNA/SV，不根据driver标签挑选。准确类型为小变异SNV/MNV/indel；冻结配置旧简写“SNV/indel only”在PROTOCOL_DEVIATIONS中更正但不改事件矩阵。参考基因组在审计突变行中均为GRCh37。

测量四类别全部显式保留：RECORDED_MUTATION、VERIFIED_CALLABLE_NEGATIVE、PANEL_ONLY_PROXY_NEGATIVE、UNMEASURED_OR_UNKNOWN。第二类全为0；不能用panel468名称替代sample callability证明。

| cancer | gene | PANEL_ONLY_PROXY_NEGATIVE | RECORDED_MUTATION | UNMEASURED_OR_UNKNOWN | VERIFIED_CALLABLE_NEGATIVE |
| --- | --- | --- | --- | --- | --- |
| COAD | APC | 408.0000 | 1021.0000 | 0.0000 | 0.0000 |
| COAD | KRAS | 814.0000 | 615.0000 | 0.0000 | 0.0000 |
| COAD | PIK3CA | 1069.0000 | 360.0000 | 0.0000 | 0.0000 |
| COAD | TP53 | 458.0000 | 971.0000 | 0.0000 | 0.0000 |
| LUAD | KEAP1 | 1985.0000 | 257.0000 | 0.0000 | 0.0000 |
| LUAD | KRAS | 1463.0000 | 779.0000 | 0.0000 | 0.0000 |
| LUAD | STK11 | 1919.0000 | 323.0000 | 0.0000 | 0.0000 |
| LUAD | TP53 | 1369.0000 | 873.0000 | 0.0000 | 0.0000 |

具体来源路径、SHA256、临床流量、16-state计数、旧分母重建及资格门槛分别位于configs/DATA_FREEZE.json和results对应TSV。没有下载新患者数据，没有外传患者级记录。来源已被早期项目探索，本轮split只建立本轮留出边界。

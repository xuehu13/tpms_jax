# 文件地图

文件状态核对2026-10-14，科研结果截至10月13日。正式工程R=/home/xuehu/projects/tpms_jax；W为当前Windows任务目录；A=E:/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus。无新计算队列，N32主线HEX27。原科学文件保持，旧下一步不自动恢复。

## 2026-10-14文件状态整理

- [已有总结的覆盖范围与实验索引](../validation/README.md)：10月7日入门总结、10月9日差异排查、10月7—12日完整报告、10月13日近期梯度总结，均已存在；本次不重复生成报告。
- `history/organization_status_20261014/before`：本次Windows及正式WSL入口修改前原字节；`archive_manifest_before.json`保留归档文件的原路径、大小和SHA256。
- `history/organization_status_20261014/artifact_qa`：4个已完成的Word/PPT排版临时目录，共232个文件、约70 MiB；没有删除文件，运行时junction仅保留链接、不遍历其目标。
- `tmp/`仅保留仍有引用的论文阅读资料及说明；`work/`无活动临时编辑稿。归档路径及恢复方法见同级归档说明，原构建收据里的旧路径按清单追溯。
- 正式WSL只同步文档和维护收据到R/docs/history/organization_status_20261014，不复制上述232个临时文件；科学源码、输入、结果、状态JSON、检查点、INP/ODB保持原位。

## 10月13日近期总结与归档

- 近期Word总结（本机成品：`W/output/TPMS近期研究与梯度验证阶段总结_20261013.docx`）：r53至r58的最近进展，重点为真实系数0.5%和10%梯度、瞬时力失败及成本。
- output/gradient_summary_20261013：唯一构建源、仅新增派生图、原图来源清单与排版检查记录。
- r57正式R/validation/equation_coefficients_20261012_r57，阅读output/r57_equation_coefficients_20261012；0.5%已完成。
- r58正式R/validation/equation_coefficients_10pct_20261013_r58，阅读output/r58_equation_coefficients_10pct_20261013；10%已完成。tools/extend.py是轻量延伸适配，analysis为一次性分析，不是新框架。
- history/organization_recent_20261013/before：Windows入口原字节；staging/r57_stage和r58_stage：已完成编辑/维护稿；staging_manifest.json及science_before.json、final_receipt.json为核对凭据。
- 正式对应归档为R/docs/history/organization_recent_20261013。科学源码、原输入、结果、失败、检查点和大数组未移动或删除。

## 10月12日完整报告与既有整理凭据

- output/TPMS固定背景可导力学计算研究_阶段完整技术报告_20261007至20261012.docx：10月7日至12日完整说明。
- output/research_report_20261007_20261012：图表来源、正文源、构建与质量检查凭据；引用原图，避免复制全部科学数据。
- history/report_consolidation_20261012/before：本轮整理前入口原字节；manifest.json记录修改。
- r56正式为R/validation/platen_xy_hex8_20261012_r56，阅读为output/r56_platen_xy_hex8_20261012；仅新增JAX前向，原壳/HEX27只引用。

## 文档分工

| 用途 | Windows入口 | 正式WSL |
|---|---|---|
| 总导航 | START_HERE.md | R/README.md |
| 长期问题 | RESEARCH_BACKGROUND.md | R/docs/RESEARCH_BACKGROUND.md |
| 当前事实／停点 | PROJECT_OVERVIEW.md | R/docs/RESEARCH_STATUS.md |
| 唯一安排 | RESEARCH_PLAN.md | R/docs/RESEARCH_PLAN.md |
| 方法与假定 | TPMS_RESEARCH_REVIEW.md | R/docs/TPMS_RESEARCH_REVIEW.md |
| 梯度完整综合分析 | [r53梯度报告](../validation/platen_xy_gradients_20261011_r53/analysis/parameter_summary/README.md) | R/validation/platen_xy_gradients_20261011_r53/analysis/parameter_summary/README.md |
| 全部实验索引 | [阅读成品索引](../validation/README.md) | R/validation/README.md |

## 已完成前向批次r54

正式`R/validation/platen_xy_diverse04_20261012_r54`；阅读`W/output/r54_platen_xy_diverse04_20261012`；原生`A/r54_diverse04_midsurface_end_20pct_20261012`。独立保存协议、输入、原曲线、峰规则／CSV、图和成本。复用r15的surface_geometry.npz及gauss_field.npz，不改原数组；轻量适配复用r51/r52程序。

## 暂停的梯度批次r53

正式B=`R/validation/platen_xy_gradients_20261011_r53`；阅读O=`W/output/r53_platen_xy_gradients_20261011`。状态看B/status.json，摘要／图／CSV有O副本，大型状态数组仅本机正式档案，不随Git普通克隆取得。

| B内目录 | 内容与范围 |
|---|---|
| input/ | 原协议、几何／距离缓存；原与补充形态执行／验收协议分开，未修改 |
| tools/ | 共享内力的轻量参数、形态、执行、分块反向及分析适配；唯一实际版本，未修改 |
| AD/、FD/ | G5厚度／宽度与实际t/w功反向 |
| G10/、G12/、G10_interface/、G12_interface/、P5/ | 独立延伸路径与切向，保留原前缀来源 |
| G5_shape/ | 原完整形态JVP和4条FD，原失败保持 |
| G5_shape_refined/ | 一次有界补充的4条FD，复用原JVP，不重复计费 |
| analysis/parameter_summary/ | 唯一详细梯度综合报告、原17行CSV、β／成本表、原3图、新3图、引用旧2图、绘图配方和来源清单 |
| analysis/G5_shape*_diagnostic/、G5_shape_geometry_diagnostic/ | 已有路径／保存态／几何定位；不是新的完整AD认证 |
| checks/、status.json | 原哈希、执行与停止收据；整体形态未过、完整形态VJP未启动 |

## 前向对照与原Abaqus文件

| 实验 | 正式WSL | Windows阅读副本／原生Abaqus |
|---|---|---|
| G共同端缘20% | R/validation/platen_xy_G_20261010_r51/experiments/midsurface_rim_20pct_20261010 | [G报告](../validation/platen_xy_G_20261010_r51/experiments/midsurface_rim_20pct_20261010/README.md)；A/r51_G_midsurface_rim_20pct_20261010/platen.inp及platen.odb |
| P/D共同端缘20% | R/validation/platen_xy_PD_20261010_r52 | [P/D报告](../validation/platen_xy_PD_20261010_r52/README.md)；原生目录路径按各构型报告引用 |
| 原r51短程／诊断 | R/validation/platen_xy_G_20261010_r51/diagnostics及abaqus、jax、analysis | [历史全过程](../validation/platen_xy_G_20261010_r51/RESEARCH_LOG.md)，原失败／INP／ODB保持 |
| 旧XYZ、速率、二值、网格等 | R/validation各原批次 | output各r*；不混作当前压板证据，完整索引见R/validation/README.md |

共享程序仍为R/hyperelastic_fem.py、surface_distance.py、pbc.py、fem.py和scripts/thin_target_explicit.py。用户原Abaqus为F:/auto_abaqus/work，论文为C:/Users/xuehu/Desktop/tpms优化，均未移动。

## 前序梯度整理与历史

Windows已完成的work/r53_stage归档到`history/organization_gradients_20261012/staging/r53_stage`，54个文件逐字节核对，文件清单见同级staging_manifest.json。没有删除原科学数据；正式B/tools不移动。当时的work和tmp已清理；本轮报告正文源与构建凭据集中保存，渲染页在tmp作为内部质量检查。

本次修改前导航、原长规划、整理脚本、93项科学小文件／共享源码保护哈希和最终收据在`history/organization_gradients_20261012`；正式收据及原导航在`R/docs/history/gradient_consolidation_20261012`。本轮之前的归档仍在history/execution_gradient_20261011_r53等原目录。

WEEKLY_RESEARCH_SUMMARY.md、MECHANISM_ANALYSIS.md和PAPER_ROUTE.md均为有明确日期的旧总结／机制／文献阅读快照；其“下一步”不作为当前任务。2026-10-07 Word／PPT、2026-10-09差异排查Word及用户要求的r51学习包按原字节保留，见[成品索引](../validation/README.md)。

## 新增批次r55

正式R/validation/platen_xy_diverse_batch_20261012_r55；阅读W/output/r55_platen_xy_diverse_batch_20261012；原生A/r55_<kind>_midsurface_end_20pct_20261012。sources按构型保存三角中面及实际Gauss场，各构型input／abaqus／jax／analysis隔离；共享力学实现仍为r51和仓库原程序。

人工审阅补充在r55原README，新增reviewed_force_comparison.png及reviewed_dynamic_windows.json/CSV，原自动图和原始指标不覆盖。审阅前文档在W/history/execution_diverse_batch_20261012_r55/review_before；哈希及审阅收据在R/validation/platen_xy_diverse_batch_20261012_r55/checks/result_review。

原Abaqus对照核查：W/output/r55_platen_xy_diverse_batch_20261012/original_model_audit，含25条历史曲线清单、8个ODB只读元数据／上板原始峰核对、参数／网格／XY方程审阅、两张新旧对照图、哈希及轻量复核配方。原F盘文件未移动或修改。

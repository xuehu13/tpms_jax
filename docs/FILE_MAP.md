# 文件地图

更新：2026-10-05。只维护一套正式FEM：WSL /home/xuehu/projects/tpms_jax。本Windows目录主要供阅读与历史核对；不要从work/tpms_jax旧副本执行。当前唯一规划为新大压缩四步，均未启动。

## 阅读入口

| 文件 | 职责 |
| --- | --- |
| [背景](RESEARCH_BACKGROUND.md) | 长期问题、研究口径、证据约定 |
| [主规划](RESEARCH_PLAN.md) | 唯一下一轮四步、验收与停止；不是历史清单 |
| [状态](RESEARCH_STATUS.md) | 当前完成/未完成/边界 |
| [综合报告](TPMS_RESEARCH_REVIEW.md) | 理论、符号、结果、大压缩差异、Explicit/接触/梯度判断 |
| [文献路线](PAPER_ROUTE.md) | 文献身份、用途及阅读边界 |
| [历史说明](history/README.md) | 冻结文档、规划和Git传输归档 |

正式仓库对应docs下同名文件，PROJECT_OVERVIEW对应RESEARCH_STATUS，START_HERE对应根README。综合PDF是同一综合报告的导出，不是第二份活动说明。

## 正式程序与当前实验

| 正式路径 | 用途 |
| --- | --- |
| fem.py、density_fem.py、pbc.py | 共享HEX8/实际Gauss场/周期约束 |
| hyperelastic_fem.py | 当前Neo-Hookean静力有限应变；无Explicit/接触/塑性 |
| surface_distance.py | 周期中面距离/恒厚表示；厚度AD不等于形态AD |
| geometry.py、volume.py、voxel_field.py | 隐式几何、用料、周期数组插值 |
| design_fem.py | 已核对参数梯度和专用刚度/用料规则 |
| binary_gyroid.py | 历史二值参考准备/限制；不自动继承失败待办 |
| scripts/README.md | 所有入口职责，复用既有求解/提取程序 |
| scripts/thin_target_finite.py、extract_thin_finite.py | 第3步静力路径与只读壳提取 |
| scripts/thin_thickness_gradient.py | 保存状态反力伴随、厚度扰动、两点损失 |
| pixi.toml、pixi.lock、tests/ | 锁定环境及维护测试；不是科研证据替代 |

当前薄壁实验根为validation/thin_target_20261004_r5，原字节冻结：

| 子目录/报告 | 内容 |
| --- | --- |
| 根input.json、step1.json及缓存；[第1步](../validation/thin_target_20261004_r5/README.md) | 中面/厚度、真实Gauss距离和占据、覆盖/周期/连接 |
| step2/及diagnostic_xyz/；[第2步](../validation/thin_target_20261004_r5/step2/README.md) | 原平端与XYZ小变形输入/场/壳提取；XYZ为后续口径 |
| step3_xyz/；[第3步](../validation/thin_target_20261004_r5/step3_xyz/README.md) | 1%/5%/10%输入、JAX保存状态、壳提取、response_summary图与停止记录；20%无作业 |
| step4_thickness_a01/与step4_thickness_a05/；[第4步](../validation/thin_target_20261004_r5/step4_thickness_a01/README.md) | 1%/5%中心复用、伴随、±厚度扰动、损失、独立壳扰动、3停止记录 |

大型缓存/场/源码快照和旧原始日志并非全随Git发布。公开仓库的清单/JSON/报告不能冒充包含全部本机数据，复现需本机冻结档案与锁定环境。source_at_run只是当时快照，不维护第二套FEM。M4最终10行CSV在正式results/m4_numerical_study.csv，Git忽略，原样保留。

## Abaqus与用户资料

命令为E:/ABAQUS/2026/Commands/abaqus.bat，计算包根为E:/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus。包根有INP/expected/网格，work通常保存ODB/dat/msg/sta/提取输出，scripts是当时执行副本；不搬动旧作业。

| 包根相对路径 | 内容 |
| --- | --- |
| thin_target_20261004_r5_step2及_step2_xyz | 原平端与XYZ小变形壳，INP/ODB/提取输出 |
| thin_target_20261004_r5_step3_xyz_01、_05、_05_refined、_10 | 第3步1%/5%/5%细分/10%独立壳作业 |
| thin_target_20261005_r5_step4_t0p498000及_t0p502000 | 第4步1%压缩下两个壳厚扰动，厚度0.498/0.502mm |
| uniform1001/abaqus_uniform_baseline、element_comparison_20261002 | 完整体/单元矩阵 |
| discrete_comparison_verified_20261002 | 同离散用户单元矩阵，原生USDFLD未全面认证 |
| binary_gyroid_20261002、mesh_quality_20261002 | 二值实体及网格质量 |
| geometry_interface_20261003_r2/、learning_bridge_20261003_r3/ | 第二/三轮新输入；G48失败原样保留 |
| mechanics_trust_20261004_r4/ | Primitive、薄Gyroid准备失败、均匀体有限应变 |

用户旧壳为F:/auto_abaqus/work/para_aly/Fine/T0p02/MS9/diverse_28；原厚0.328595mm、Explicit/塑性/摩擦压板，不与当前纯弹性XYZ直接比较。只读参考；材料在abaqus/blocks/material_section.inc，旧结果在results。文献原件在C:/Users/xuehu/Desktop/tpms优化，本次未改。

## 冻结历史与本次整理

| 位置 | 内容/状态 |
| --- | --- |
| 正式validation/near_term_20261003/ | 第一轮解析前向/占据/梯度 |
| validation/geometry_interface_20261003_r2/、learning_bridge_20261003_r3/ | 第二/三轮输入、专用梯度和停止；训练未执行 |
| validation/mechanics_trust_20261003_r4/ | 第四轮审计、构型筛查、完整体20%、旧Gyroid部分状态 |
| 正式validation/README.md | 所有冻结实验入口；旧失败不是当前待办 |
| history/planning_snapshots_20261005/ | 已完成薄壁四步原计划；正式docs/history/completed_plans/同用途 |
| history/planning_snapshots_20261004/ | 第四轮原计划及被替换的中等厚Gyroid计划 |
| history/git_transport/completed_20261005/ | 8项旧Git仓库/包集中归档，112文件/43778409字节，原字节保持 |
| history/reading_copies_before_review/、early_exports_20261002/ | 更早阅读副本/下载包，不能作为正式入口 |
| work/thin_target_step2_20261004/～step4_20261005/ | 各次一次性审计、调度、文档/发布收据，科学结果仍在正式目录 |
| work/large_compression_review_20261005/ | 本次前状态清单、6篇相关页摘录/预览、23PDF身份、迁移收据、PDF/发布QA |
| 正式docs/history/large_compression_review_20261005/ | 本次小型身份/核对/整理收据；不复制全部文献或大缓存 |
| [工作目录说明](history/README.md) | 阅读副本、已完成工具、历史归档的分工 |

旧Git收据中的旧路径不改，新work/large_compression_review_20261005/organization_receipt.json记录旧新绝对路径与哈希。历史数据、原始结果、停止日志不删除或重写；活动说明精简，正文集中到综合报告。未来新计算进新目录，不覆盖薄壁已完成记录。

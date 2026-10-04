# 文件地图

更新：2026-10-05。方法和结果看 [综合报告](TPMS_RESEARCH_REVIEW.md)，事实看 [状态](RESEARCH_STATUS.md)，执行只看 [唯一主规划](RESEARCH_PLAN.md)。本页只说明位置和职责。

## 正式计算入口与活动文档

正式目录 `/home/xuehu/projects/tpms_jax`；Windows访问 `\\wsl.localhost\Ubuntu-24.04\home\xuehu\projects\tpms_jax`。Python `.pixi/envs/default/bin/python`，环境由pixi.toml/pixi.lock锁定；安装JAX-FEM 0.0.12的solver.py与发行包RECORD哈希一致；它和只读源码checkout不同，不能因此称为本地补丁。运行接口按安装代码核对，不另维护求解器副本。`/home/xuehu/projects/jax-fem`只读参考，jax-fem-workspace为早期工作区。

| Windows阅读文件 | 正式WSL对应 | 唯一职责 |
| --- | --- | --- |
| START_HERE.md | README.md | 阅读入口 |
| RESEARCH_BACKGROUND.md | docs/RESEARCH_BACKGROUND.md | 长期研究问题与方法边界 |
| RESEARCH_PLAN.md | docs/RESEARCH_PLAN.md | 5%目标薄壁四步，第2步收口；用户选定XYZ；第3步收口，下一项已通过点的有效梯度 |
| PROJECT_OVERVIEW.md | docs/RESEARCH_STATUS.md | 当前事实及缺口 |
| TPMS_RESEARCH_REVIEW.md | docs/同名文件 | 完整理论、符号和进度解释 |
| output/pdf/TPMS_RESEARCH_REVIEW.pdf | docs/TPMS_RESEARCH_REVIEW.pdf | 同一综合报告的可分享导出 |
| FILE_MAP.md、PAPER_ROUTE.md | docs/同名文件 | 位置与文献来源 |

报告不再承担执行计划；详细历史过程只查冻结轮次报告。Windows与WSL文档是两个环境的同步阅读入口，维护代码仅在WSL，PDF由Markdown导出。

## 核心程序：一套维护实现

| 程序 | 职责/适用状态 |
| --- | --- |
| scripts/thin_target_finite.py、extract_thin_finite.py | 第3步分段XYZ Neo-Hookean压缩与只读壳ODB提取；1%/5%初筛通过、10%未接受、20%未提交 |
| scripts/thin_target_linear.py、extract_thin_shell.py | 第2步薄壁线弹性对照、唯一XYZ边界诊断和只读ODB提取；复用同一FEM |
| surface_distance.py、scripts/prepare_thin_target.py | 固定周期三角中面的距离、物理厚度投影及第1步真实Gauss准备；厚度AD接口与形态AD分开 |
| geometry.py、volume.py | Gyroid/Primitive、光滑占据及体积 |
| fem.py、density_fem.py、pbc.py | HEX8线性力学、真实Gauss场、周期约束/宏观松弛 |
| hyperelastic_fem.py | 匹配Neo-Hookean及初始Gauss能量权重；共享XY/XYZ及实际Gauss入口；目标薄壁1%/5%初筛，10%未接受，平衡梯度待验证 |
| design_fem.py | 历史参数、通用伴随接口及已检查专用K/Vf规则 |
| voxel_field.py | 周期数组插值，先G后投影 |
| binary_gyroid.py | 二值实体参考几何、网格及审查；薄壁/变量壁宽G48限制保留 |
| scripts/finite_strain_gyroid.py | 历史中等厚度Gyroid前向/捕获；不能作新薄壁输入 |
| scripts/finite_strain_uniform.py、extract_finite_strain.py | 均匀体基准、匹配INP/ODB提取与检查 |
| scripts/prepare_abaqus_binary.py、run_abaqus_binary.ps1、extract_abaqus_binary.py | 二值小变形参考生成/运行/提取，已有提取接口复用 |
| scripts/m1_*～m4_*及早期capture/prepare/extract | 历史验证入口；从报告查用途，不按编号自动全部重跑 |
| tests/；scripts/README.md | 维护检查与脚本职责入口；最新测试数查状态/本轮收据 |

此前维护合并有限应变重复构建、区分参数校验/可追踪更新；本轮最小增加XYZ/实际Gauss入口及已消耗切线值释放，未搬为大型包或增加物理模块。成本诊断仅在实际受阻时考虑，届时复用现有custom_solver入口，必要最小扩展放正式代码；一次性审计/发布工具不提升为通用框架。

## 冻结研究证据

| 正式目录/报告 | 内容 |
| --- | --- |
| validation/near_term_20261003/；[第一轮](../validation/near_term_20261003/README.md) | 解析前向、失败占据输入及诊断梯度 |
| validation/geometry_interface_20261003_r2/；[第二轮](ROUND2_REPORT.md) | 有符号隐式输入三锚点、导数与成本 |
| validation/learning_bridge_20261003_r3/；[第三轮](ROUND3_REPORT.md) | N64专用梯度、性能跨度不足和新参考失败 |
| validation/mechanics_trust_20261003_r4/step1/；[审计](ROUND4_STEP1_REPORT.md) | 原始输入/约束/拓扑/映射/响应，零新增FEM |
| 同一第四轮step2/；[构型筛查](ROUND4_STEP2_REPORT.md) | Primitive通过、薄壁参考失败、输入/执行/验收 |
| 同一第四轮step3/；[完整实体有限应变](ROUND4_STEP3_REPORT.md) | N2/N4/半步、两条Abaqus路径、解析/材料/节点/功 |
| 同一第四轮step4/；[Gyroid预检查](ROUND4_STEP4_REPORT.md) | N16/N32完整路径、N48部分进度/位移/停止记录、能源分组/计时 |

各目录的source_before/source_at_run等是本机复现快照，不是另外维护的FEM；未跟踪的快照目录/原始日志不新加入Git，关键清单/结果仍发布。克隆仓库不包含完整旧实验档案，复核旧运行时另取本机冻结档案及锁定环境。旧manifest的文档哈希指当时版本，不能改旧清单匹配新文档。原始输入、结果、日志、停止账本原位冻结。

第4步重点文件：`N16_d005.json`、`N32_d005.json`完整；`N48_d005.progress.json`、`.current.json`、`.log`仅部分完成；对应`.displacement.npz`/`.last_state.npz`为成功状态；`precision_decision.json`为预算停止；`summary.json`、`details.json`、`energy_groups.json`为零新增求解的汇总/后处理。N48没有完整成功JSON或独立TPMS非线性ODB。

早期验证全部索引在正式`validation/README.md`，包括均匀、单元矩阵、同离散、二值实体/质量、投影/网格/梯度。M4原10行CSV在`results/m4_numerical_study.csv`，Git忽略，保持原样。

已完成计划：WSL `docs/history/completed_plans/RESEARCH_PLAN_ROUND{1,2,3,4}_COMPLETED.md`；Windows第四轮原文在`history/planning_snapshots_20261004/RESEARCH_PLAN_ROUND4_COMPLETED.md`。它们不是第二份主规划。未执行的中等厚度成本/加密计划原文保存在`history/planning_snapshots_20261004/RESEARCH_PLAN_MIDDLE_GYROID_SUPERSEDED.md`；正式副本在`docs/history/thin_target_correction_20261004/`。新实验`validation/thin_target_20261004_r5/`已启动：input.json与原中面输入、实际Gauss距离/占据缓存、step1.json及截面图；它是独立5%目标，不覆盖旧结果。

## Abaqus原始包

根目录 `E:\ABAQUS\2026temp\Abaqus_Work\tpms_jax_abaqus`；命令 `E:\ABAQUS\2026\Commands\abaqus.bat`；用户旧壳算例 `F:\auto_abaqus\work`只作参考。新核查对象为`para_aly/Fine/T0p02/MS9/diverse_28/`，实际物理输入在`abaqus/blocks/material_section.inc`，原结果在`results/`；仅只读审查，不改旧作业。

| 根下路径 | 内容 |
| --- | --- |
| uniform1001/abaqus_uniform_baseline、element_comparison_20261002 | 完整体/单元矩阵基准 |
| discrete_comparison_20261002、discrete_comparison_verified_20261002 | 同离散用户单元矩阵，原生USDFLD未认证 |
| binary_gyroid_20261002、mesh_quality_20261002 | 基准实体及质量/加密 |
| geometry_interface_20261003_r2/{c048,c060}/{G32,G48} | 常数Gyroid新锚点 |
| learning_bridge_20261003_r3/ax035/G32；G48 | 成功粗参考；G48提交前失败，无结果 |
| mechanics_trust_20261004_r4/primitive/{G32,G48} | Primitive成功输入/ODB/质量/节点样本 |
| mechanics_trust_20261004_r4/thin_gyroid/ | G32仅输入；三个G48准备失败，没有分析 |
| mechanics_trust_20261004_r4/step3_uniform/{N4_d010,N4_d005} | 均匀体20%基准及半步，原始ODB/日志/验收 |

包根保存INP、mesh.npz、expected；work保存ODB/dat/msg/sta及提取/验收；scripts是当时执行副本。原包不搬动；本轮新增以下两套弹性壳作业，不能与旧Explicit参考混用。

## Windows归档与一次性工具

| 目录 | 角色 |
| --- | --- |
| history/reading_copies_before_review/ | 更早的报告/图/日志阅读副本及索引 |
| history/early_exports_20261002/ | 原outputs/中的早期下载包/结果；不能作正式计算入口 |
| history/git_transport/ | 原work/中的6个Git bundle，字节不改 |
| history/planning_snapshots_20261003/ | 原tmp/中的三批过期规划/范围快照，保留原字节 |
| history/planning_snapshots_20261004/ | 本次替换前的第四轮原计划 |
| history/research_review_20261003/ | 原综合Markdown/PDF，冻结阅读版本 |
| work/tpms_jax、work/archive_20261002/ | 历史Git副本/工具，原4个未提交文件保留 |
| work/archive_20261003*、planning_round4_20261003/ | 历史草案/工具/制定第四轮时的输入快照 |
| work/mechanics_trust_* | 第四轮一次性工具/图/发布收据，原位保持 |
| work/documentation_review_20261003/、documentation_review_20261004/ | 各次整理前文档、文献核查、导出/QA与收据 |
| tmp/research_questions_20261003/、tmp/pdfs/ | 文献提取/渲染缓存，身份索引路径保持；不是执行程序 |
| output/pdf/ | 当前可分享PDF |

本次收据 `work/documentation_review_20261004/organization_receipt.json` 逐文件记录旧/新绝对路径和SHA256，144个文件10组归档均核对一致；`verification.json`核对59个源码/环境记录、842个冻结证据及文档链接/同步。WSL副本位于`docs/history/documentation_review_20261004/`。上述是此前整理的收据；本轮维护提交另记，不修改旧收据，也没有环境升级或科研FEM/Abaqus作业。


## 本次薄壁目标纠正的记录

`work/thin_tpms_feasibility_20261004/`仅为一次审查：reference_files.json、geometry_and_reference.json、gyroid_thickness.json、literature_reading.json；before_local/before_formal保存修订前文本/PDF；publication_receipt.json、verification.json核对同步与冻结证据。没有新FEM源码或实验结果，临时RESEARCH_PLAN.md只作发布稿，唯一活动规划仍在根目录/正式docs。对应正式历史记录在`docs/history/thin_target_correction_20261004/`。

修订前综合报告及PDF位于`history/research_review_20261004_before_thin/`，只供追溯；当前output/pdf导出与新的综合Markdown一致。没有重复建立另一份薄壁背景报告。

## 2026-10-04维护与Git发布

`work/repository_cleanup_20261004/`保存整理前源码/文档、Git清单、安装包RECORD核对、维护测试日志、冻结哈希验证和发布收据；`transport.git`/bundle仅传输工具，不是新维护仓库。正式`docs/history/repository_cleanup_20261004/`保存小型维护摘要与验证收据。没有再创建一份研究报告或活动规划；此处记录的是维护时的状态；最新5%薄壁实验目录现已另建。

第1步正式目录`validation/thin_target_20261004_r5/`；Windows说明为 [第1步报告](../validation/thin_target_20261004_r5/README.md)，图与小收据在work/thin_target_20261004_r5/。大Gauss缓存仅在正式WSL，避免重复复制。

第2步正式目录`validation/thin_target_20261004_r5/step2/`，其中`diagnostic_xyz/`为唯一边界诊断；说明见 [第2步报告](../validation/thin_target_20261004_r5/step2/README.md)。两份INP/JSON/模式图可随Git发布；NPZ、原始日志及source_*复现快照留本机。Abaqus原始目录为包根下`thin_target_20261004_r5_step2`和`thin_target_20261004_r5_step2_xyz`。本轮一次性后处理/输出修复/发布收据在`work/thin_target_step2_20261004/`，不是第二套维护程序。

最新第3步正式证据位于`validation/thin_target_20261004_r5/step3_xyz/`；三个a0p*目录是观察点，attempt*是冻结停止记录，不是自动重跑待办。Abaqus四个独立包为E盘`thin_target_20261004_r5_step3_xyz_01`、`_05`、`_05_refined`、`_10`。Windows阅读/一次性审阅和发布记录在`work/thin_target_step3_20261004/`。完整结果见 [第3步报告](../validation/thin_target_20261004_r5/step3_xyz/README.md)，决定只读step3_decision.json。没有20%/设计梯度/训练结果。

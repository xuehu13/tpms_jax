# 文件地图：当前入口、冻结证据与历史

更新2026-10-09：r46只读复盘/整理已完成，未新增仿真。N64峰仍未捕获；本轮具体范围见状态。R=`/home/xuehu/projects/tpms_jax`为唯一生产仓库，W为Windows阅读目录。原科学文件、INP/ODB和论文不搬迁。

## 当前入口各管一件事

| Windows入口 | 内容 | 正式仓库位置 |
| --- | --- | --- |
| [START_HERE](../README.md) | 导航 | R/README.md |
| [背景](RESEARCH_BACKGROUND.md) | 长期研究问题、固定物理范围 | R/docs/RESEARCH_BACKGROUND.md |
| [唯一规划](RESEARCH_PLAN.md) | 下一轮四步及启动边界 | R/docs/RESEARCH_PLAN.md |
| [状态](RESEARCH_STATUS.md) | 已完成/失败/尚未执行 | R/docs/RESEARCH_STATUS.md |
| [机制审查](MECHANISM_ANALYSIS.md) | 初始占据与动态峰位机制 | R/docs/MECHANISM_ANALYSIS.md |
| [综合说明](TPMS_RESEARCH_REVIEW.md) | 方法、符号、结果指标、可行性 | R/docs/TPMS_RESEARCH_REVIEW.md |
| [文献](PAPER_ROUTE.md) | 实际阅读范围与出处 | R/docs/PAPER_ROUTE.md |
| [周总结](WEEKLY_RESEARCH_SUMMARY.md) | 2026-10-07事实快照 | R/docs/WEEKLY_RESEARCH_SUMMARY.md |
| [工作约定](../AGENTS.md) | 后续执行边界 | R/AGENTS.md |

## 最新科学证据

| 正式相对R目录 | 内容 | Windows阅读 |
| --- | --- | --- |
| validation/binary_mesh_reassessment_20261009_r46/ | 既有N32/N64共同窗口/惯性只读复盘、一图、报告、保护与整理收据；零新作业 | [r46](../validation/binary_mesh_reassessment_20261009_r46/REVIEW.md) |
| validation/binary_n64_20261009_r45/ | N64 T1ms至13.59%预算停、峰未捕获；唯一peak_N64路径/完整检查点/匹配壳/两图/口径 | [r45](../validation/binary_n64_20261009_r45/REVIEW.md) |
| validation/binary_full20_20261008_r44/ | 完整二值20%及保载、预算精确接续、两图/摘要/冻结收据；0数值失败，精度未达10% | [r44](../validation/binary_full20_20261008_r44/REVIEW.md) |
| validation/initial_tangent_20261008_r30/ | 原N32初始切线/Standard壳、原场、协议、结果 | [r30–r31](../validation/initial_tangent_20261008_r30/REVIEW.md) |
| validation/membrane_diagnostic_20261008_r31/ | 平直膜向补片与同Gauss解析 | 同上 |
| validation/initial_bias_mechanism_20261008_r32/ | 原壳S/E、膜弯/迹分析与区域 | [r32](../validation/initial_bias_mechanism_20261008_r32/REVIEW.md) |
| validation/local_quadrature_20261008_r33/ | 原1941选区/原与8/12积分及综合报告 | [r33–r34](../validation/local_quadrature_20261008_r33/REVIEW.md) |
| validation/local_reequilibrium_20261008_r34/ | 一次仅积分初始干预、原状态/结果 | 同上 |
| validation/thickness_kinematics_20261008_r35/ | 原两状态、27点分区/法向线、协议/两图 | [r35](../validation/thickness_kinematics_20261008_r35/REVIEW.md) |
| validation/compatible_mode_20261008_r36/ | 一个兼容模式/8和12级/微小固定节点释放 | [r36–r38](../validation/interface_width_mixed_20261008_r38/REVIEW.md) |
| validation/interface_width_20261008_r37/ | 两个新宽度原27点静力/同选区固定场复积分 | 同上 |
| validation/interface_width_mixed_20261008_r38/ | 一个窄界面同部分密积分确认/综合报告/两图 | 同上 |
| validation/binary_occupancy_20261008_r40/ | 二值27点、冻结选区8/12及一次部分密积分平衡；综合报告/图 | [r40–r41](../validation/binary_occupancy_20261008_r40/REVIEW.md) |
| validation/initial_diverse28_20261008_r41/ | 当前N32零态/匹配Standard壳/全部周期检查/摘要/重建配方 | 同上 |
| validation/binary_diverse28_20261008_r43/ | 自身3896选区、二值原点及一次配对/局部积分与能量账目/原中断/两图；非生产 | [r43](../validation/binary_diverse28_20261008_r43/REVIEW.md) |
| validation/initial_bias_reassessment_20261008_r42/ | 四个既有初始状态的能量账目/协议/原数据保护清单/一图；零新求解 | [机制审查](MECHANISM_ANALYSIS.md) |
| validation/curved_patch_20261008_r39/ | 圆柱设计/解析/两Abaqus参照/背景原失败/只读审计/图 | [r39](../validation/curved_patch_20261008_r39/REVIEW.md) |

全部阶段见R/validation/README.md。代表C²与厚度范围为r12/r13，迁移与原失败为r15/r17/r18，同速率r20，未收敛模式r21/r22，未编译背景稿r23/r24，填充r25、支撑r26、板弯r27、N64 r28、初始只读r29。旧共同保存态桥接在r6/background_bridge，不是独立路径。

生产源码：R/hyperelastic_fem.py、surface_distance.py、pbc.py、fem.py、scripts/thin_target_explicit.py；环境pixi.lock。R/scripts/README.md只作程序用途索引，不再复制近期计划。validation内诊断是冻结实验，不是第二套生产框架。

## Abaqus、用户原件与成品

命令 `E:/ABAQUS/2026/Commands/abaqus.bat`；原生作业根 `E:/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus/`：

- `binary_n64_20261009_r45_shell_T0p001000/`：当前同速率壳原INP/ODB；`binary_n64_20261009_r45_shell_T0p000250/`为早期快速参照，均保留原质量检查。
- `initial_diverse28_20261008_r41_diverse28/`：本轮代表构型初始Standard壳INP/ODB，全节点结果在R对应abaqus目录。
- `curved_patch_20261008_r39/`：本轮CAX8与S3R初始参照、原INP/ODB。
- `initial_tangent_20261008_r30_diverse04/`：初始Standard壳。
- `n64_resolution_20261008_r28_shell_T0p002/`：r28同速率壳。
- `shell_rate_20261007_r20_diverse04_explicit_T0p004/`、`geometry_transfer_20261006_r15_diverse04_explicit_T0p040/`：diverse_04速率参照。
- `plate_bending_20261008_r27/`、`shell_fill_20261007_r25_nearzero/`、`background_bridge_20261005/`：冻结专项。

用户中面 `F:/auto_abaqus/work/para_aly/Fine/T0p02/MS9/diverse_04/abaqus/ingredients/shell_mesh.inc`，diverse_28同级，仅借几何。论文 `C:/Users/xuehu/Desktop/tpms优化/`，原Word/PPT在 `C:/Users/xuehu/Desktop/tpms总结1006/`，本次不改成品或原件。

## 本次整理与归档

当前入口修改前原文：W/history/before_r46_mesh_reassessment_20261009/；正式对应入口/validation索引：R/docs/history/before_r46_mesh_reassessment_20261009/。此前各轮before_*与completed_tools_*原归档保留，不逐轮在当前地图追加重复段落。

r46只读配方、统一比较、机制结论及新规划在R/validation/binary_mesh_reassessment_20261009_r46/，阅读在W/output/r46_binary_mesh_reassessment_20261009/。Windows r45的已完成辅助脚本与两日志归档至W/history/completed_tools_20261009/r45_support/，archive_receipt.json记录哈希不变；没有删除原文件。

r45配方、峰后窗口修订、匹配壳、预算停止和保存状态位于R/validation/binary_n64_20261009_r45/，阅读在W/output/r45_binary_n64_20261009/；peak_N64是唯一有效非零路径，full_N64仅旧零态准备。未执行N32、保载或AD，原大文件不搬迁。

r44配方、预算精确接续与只读后处理在R/validation/binary_full20_20261008_r44/；结果与保护收据同目录，完整路径/场/原日志留本机。两次后处理格式中断原版本与原因在postprocess_attempt01/02，不是求解重试。r43支持脚本和原中断仍留W/history/completed_tools_20261008/r43_support/及正式r43目录，不再作为当前待办。work只保留用途说明。

W/output的成品入口README.md只指向完整Word和最新重点版PPT。前两版PPT按原字节归档至W/history/presentation_versions_20261008/，archive_receipt.json记录原/新路径与SHA256。它们仍可恢复；本轮没有改成品内容。

Git保留精选配方、协议、结果摘要、必要图与当前文档。重复before_*快照、日志、大数组及重复壳输入/全节点结果留本机；已跟踪历史不改字节。Git克隆不是完整实验档案，原计算复核需结合本机validation数据和原生Abaqus目录。

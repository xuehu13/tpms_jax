# 文件地图：当前入口、冻结证据与历史

更新2026-10-08。R=`/home/xuehu/projects/tpms_jax`，唯一正式程序；W=当前Windows阅读目录。原科学文件、INP/ODB、论文不搬迁。各轮完成不等于全部验证通过。

## 当前入口各管一件事

| Windows入口 | 内容 | 正式仓库位置 |
| --- | --- | --- |
| [START_HERE](../README.md) | 导航 | R/README.md |
| [背景](RESEARCH_BACKGROUND.md) | 长期研究问题、固定物理范围 | R/docs/RESEARCH_BACKGROUND.md |
| [唯一规划](RESEARCH_PLAN.md) | 仅当前三步顺序 | R/docs/RESEARCH_PLAN.md |
| [状态](RESEARCH_STATUS.md) | 已完成/失败/尚未执行 | R/docs/RESEARCH_STATUS.md |
| [机制审查](MECHANISM_ANALYSIS.md) | 约5%原因、理论与判断 | R/docs/MECHANISM_ANALYSIS.md |
| [综合说明](TPMS_RESEARCH_REVIEW.md) | 方法、符号、结果指标、可行性 | R/docs/TPMS_RESEARCH_REVIEW.md |
| [文献](PAPER_ROUTE.md) | 实际阅读范围与出处 | R/docs/PAPER_ROUTE.md |
| [周总结](WEEKLY_RESEARCH_SUMMARY.md) | 2026-10-07事实快照 | R/docs/WEEKLY_RESEARCH_SUMMARY.md |
| [工作约定](../AGENTS.md) | 后续执行边界 | R/AGENTS.md |

## 最新科学证据

| 正式相对R目录 | 内容 | Windows阅读 |
| --- | --- | --- |
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
| validation/curved_patch_20261008_r39/ | 圆柱设计/解析/两Abaqus参照/背景原失败/只读审计/图 | [r39](../validation/curved_patch_20261008_r39/REVIEW.md) |

全部阶段见R/validation/README.md。代表C²与厚度范围为r12/r13，迁移与原失败为r15/r17/r18，同速率r20，未收敛模式r21/r22，未编译背景稿r23/r24，填充r25、支撑r26、板弯r27、N64 r28、初始只读r29。旧共同保存态桥接在r6/background_bridge，不是独立路径。

生产源码：R/hyperelastic_fem.py、surface_distance.py、pbc.py、fem.py、scripts/thin_target_explicit.py；环境pixi.lock。R/scripts/README.md只作程序用途索引，不再复制近期计划。validation内诊断是冻结实验，不是第二套生产框架。

本轮当前文档修改前副本在W/history/before_r36_r38_20261008及R/docs/history/before_r36_r38_20261008；新支持工具一次归档至W/history/completed_tools_20261008/r36_r38_support。原科学文件、壳输入和大数组未迁移。

r39修改前当前入口在W/history/before_r39_20261008及R/docs/history/before_r39_20261008；本轮支持工具归档在W/history/completed_tools_20261008/r39_support，正式诊断仍在原validation目录。

r40–r41修改前入口在W/history/before_r40_r41_20261008及R/docs/history/before_r40_r41_20261008；支持工具在W/history/completed_tools_20261008/r40_r41_support。当前只保留一份综合报告/图；重复壳输入、全节点结果和提取器留本机，Git用重建配方/摘要，不删除原件。

## Abaqus、用户原件与成品

命令 `E:/ABAQUS/2026/Commands/abaqus.bat`；原生作业根 `E:/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus/`：

- `initial_diverse28_20261008_r41_diverse28/`：本轮代表构型初始Standard壳INP/ODB，全节点结果在R对应abaqus目录。
- `curved_patch_20261008_r39/`：本轮CAX8与S3R初始参照、原INP/ODB。
- `initial_tangent_20261008_r30_diverse04/`：初始Standard壳。
- `n64_resolution_20261008_r28_shell_T0p002/`：r28同速率壳。
- `shell_rate_20261007_r20_diverse04_explicit_T0p004/`、`geometry_transfer_20261006_r15_diverse04_explicit_T0p040/`：diverse_04速率参照。
- `plate_bending_20261008_r27/`、`shell_fill_20261007_r25_nearzero/`、`background_bridge_20261005/`：冻结专项。

用户中面 `F:/auto_abaqus/work/para_aly/Fine/T0p02/MS9/diverse_04/abaqus/ingredients/shell_mesh.inc`，diverse_28同级，仅借几何。论文 `C:/Users/xuehu/Desktop/tpms优化/`，原Word/PPT在 `C:/Users/xuehu/Desktop/tpms总结1006/`，本次不改成品或原件。

## 本次整理与归档

修改前九个当前入口原文在W/history/before_research_reassessment_20261008/；正式对应入口及程序/validation/历史索引在R/docs/history/before_research_reassessment_20261008/。旧逐轮快照保留原位，不作为活动文档。文档积累历史收回归档，当前地图不再逐轮重复“本次归档”段落。

本次收据与本地文献元数据在R/docs/history/research_reassessment_20261008/，Windows支持工具/摘录/阅读页图在W/history/completed_tools_20261008/research_reassessment/。W/work只留说明；每个output轮次仅一个阅读报告与必要图。正式大数组、原冻结报告/源码不移动。

Git保留维护源码、精选协议/结果摘要与必要图；重复文档快照和原日志/大数组留本机。已跟踪历史不改字节，新的before_*副本不重复发布。Git克隆不是完整实验档案，复核原运行须结合本机数据和Abaqus原目录。

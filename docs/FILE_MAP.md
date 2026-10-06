# 文件地图：当前入口与冻结证据

更新2026-10-06。正式程序仅WSL `/home/xuehu/projects/tpms_jax`，Windows是阅读/历史。详细旧地图已保存在`history/before_research_synthesis_20261006/FILE_MAP.md`，本页只列阅读及定位所需路径。

## 当前阅读入口

| 文件 | 唯一用途；正式仓库对应 |
| --- | --- |
| RESEARCH_BACKGROUND.md | 长期问题与边界；docs同名 |
| RESEARCH_PLAN.md | 唯一四步及当前第4步；docs同名 |
| RESEARCH_STATUS.md | 完成/未完成事实；docs/RESEARCH_STATUS.md |
| TPMS_RESEARCH_REVIEW.md | 当前完整方法、数字解释、进度及可行性；docs同名 |
| PAPER_ROUTE.md | 文献及实际阅读范围；docs同名 |
| START_HERE.md、AGENTS.md | 阅读入口与工作约定；README.md、根AGENTS.md |
| THICKNESS_RANGE_PROGRESS.md | r13匹配0.45/0.55mm完整20%及范围；docs同名 |
| VOID_CONTINUATION_PROGRESS.md | 最新r12执行细节，冻结原字节；docs同名 |
| 其余*_PROGRESS.md、*_REPORT.md | 冻结各轮执行证据，历史下一步不是待办；docs同名 |

## 正式程序与科学结果

以下路径均相对WSL正式根。维护现有一套FEM，历史入口仍可复现；此次不做无依据的代码搬迁或重构。

| 路径 | 用途 |
| --- | --- |
| hyperelastic_fem.py | 唯一共享NH/可选C²虚域材料、内力、HEX8/HEX27；默认NH保持 |
| scripts/thin_target_explicit.py | 当前XYZ中央差分/HRZ/回退减步入口，研究候选需明确选objective_void |
| surface_distance.py、pbc.py | 周期中面距离/真实Gauss占据、周期自由度 |
| fem.py、density_fem.py及其他原模块 | 原线性/几何/用料/专用设计计算，按既有范围维护 |
| scripts/README.md、tests/、pixi.toml/lock | 现有入口分类、维护检查、锁定环境 |
| validation/thickness_range_20261006_r13/ | 匹配厚度输入、两条完整20%、两份壳参照、27/125点与模式、范围决定及哈希 |
| validation/void_continuation_20261006_r12/ | 当前C²候选：输入、45项检查、原6点、快慢20%、125点、速率、范围决定及证据哈希 |
| validation/large_compression_20261005_r6/ | 冻结原HEX8/HEX27、壳参照、背景桥接、20%原梯度失败与定位 |
| validation/thin_target_20261004_r5/ | 当前恒厚中面/Gauss基础及早期薄壁范围 |
| validation/simulation_error_20261006_r7/ | 保存场定位及失败三维体网格准备 |
| validation/jax_improvement_20261006_r8/ | 64点未采纳候选及27/125点检查 |
| validation/virtual_kernel_20261006_r9/ | 经典虚域转动缺陷，未采纳 |
| validation/objective_virtual_20261006_r10/ | 客观核原保存场验证，当时未接时间推进 |
| validation/objective_path_20261006_r11/ | 上一客观路径改善及6点混合域失败，保持 |
| validation/README.md、results/m4_numerical_study.csv | 全部实验索引；早期M4最终10工况CSV（Git忽略） |

r12的`T0p004/`、`T0p008/`含完整输入、结果、最终field.npz及运行源码；`*_analysis/`含对壳、模式和27/125点保存场检查；`decision.json`、`rate_check.json`、`evidence_manifest.json`记录验收及哈希。大场/日志/源码快照留本机，GitHub不含全部本机数据。

## Abaqus与用户资料

- 命令：`E:/ABAQUS/2026/Commands/abaqus.bat`。
- 包根：`E:/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus`；现有20%壳包为`large_compression_20261005_r6_standard`、`_explicit_T0p020`、`_explicit_T0p040`，含INP/ODB/日志。
- 用户原中面：`F:/auto_abaqus/work/para_aly/Fine/T0p02/MS9/diverse_28/abaqus`。
- 用户论文：`C:/Users/xuehu/Desktop/tpms优化`；原件未移动/改写。
- 原模型和资料仅参考，不照搬原厚度、塑性或压板边界。r13新增0.45/0.55mm各一个Explicit匹配壳，原模型/资料不改。

## Windows历史与整理回执

所有已完成`work/<旧目录名>/`移动到`history/completed_tools_20261006/<旧目录名>/`，目录内部原字节保留，包括日志、摘录、transport.git和发布收据。旧脚本中的旧绝对路径仅作历史上下文，不能直接重跑；恢复定位先看`Windows history/completed_tools_20261006/README.md`（Windows归档索引）。完整逐文件映射/哈希在`work/research_synthesis_20261006/organization_receipt.json`，后续可将本轮收据整体归档。

本次改写前活动说明在`history/before_research_synthesis_20261006/`。正式仓库备份/整理收据在`docs/history/research_synthesis_20261006/`。科学结果、用户论文、ODB和WSL运行源码没有移动或删除。

更早的`history/reading_checkout_20261006/tpms_jax`是旧阅读检出，`history/git_transport/`是原已完成传输，`history/rendered_reports_20261005/`为旧PDF。它们均非活动程序/最新说明，原详细映射可查备份地图。本轮不新增重复PDF。

## 当前新增证据

r13新原生作业：`E:/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus/thickness_range_20261006_r13_t0p45_explicit_T0p040`及同名`t0p55`目录。ODB留该处；正式实验各厚度`abaqus/explicit_T0p040/`有提取和retention.json。各厚度`T0p004/`含场/完整结果、`analysis/`含27/125点/模式与comparison.json；一份规范INP和厚度单行差异可复现两侧输入。Windows `work/thickness_range_20261006`仅本轮一次性准备/报告/发布收据，不是FEM入口。

# 文件地图：暂停后的阅读入口与证据

更新2026-10-07。正式程序只在WSL `/home/xuehu/projects/tpms_jax`（以下R）；Windows本目录（以下W）为阅读/历史。当前暂停，没有由本轮启动的研究作业。

## 根目录10个文件各有一个用途

| W文件 | 用途；R对应 |
| --- | --- |
| [WEEKLY_RESEARCH_SUMMARY.md](WEEKLY_RESEARCH_SUMMARY.md) | 本周脉络、结果含义、可行性；docs同名 |
| [RESEARCH_BACKGROUND.md](RESEARCH_BACKGROUND.md) | 长期目标与固定边界；docs同名 |
| [RESEARCH_PLAN.md](RESEARCH_PLAN.md) | 唯一恢复后大致四步，当前全部暂停；docs同名 |
| [RESEARCH_STATUS.md](RESEARCH_STATUS.md) | 完成/未完成事实；docs/RESEARCH_STATUS.md |
| [TPMS_RESEARCH_REVIEW.md](TPMS_RESEARCH_REVIEW.md) | 方法、符号、精度/梯度范围；docs同名 |
| [MECHANISM_ANALYSIS.md](MECHANISM_ANALYSIS.md) | 当前原因排序及解释边界；docs同名 |
| [PAPER_ROUTE.md](PAPER_ROUTE.md) | 已读来源与阅读边界，非任务；docs同名 |
| [START_HERE.md](../README.md) | 导航；R/README.md |
| [AGENTS.md](../AGENTS.md) | 研究工作约定与暂停要求；R/AGENTS.md |
| [FILE_MAP.md](FILE_MAP.md) | 本地图；docs同名 |

## 历史报告和准备稿

- W原根目录14份阶段报告移到`history/progress_reports_20261007/`，目录README按主题索引；阅读链接已调整。移动前原字节在`history/before_pause_20261007/`，正式原报告仍在R的docs及各validation轮次，未移动科学原件。
- W原`work/background_probe_20261007/`移到`history/paused_background_probe_20261007/`：仅hex27_reference.cpp和prepare.py，未编译/未运行；R `validation/background_probe_20261007_r24/`保留准备清单并新增暂停说明，不覆写原记录。
- W其余7个work工具目录移到`history/completed_tools_20261007/remaining_tools/`；tmp的2个旧阅读目录移到`history/completed_tools_20261007/early_reading/`。这些是历史辅助/摘录，不是第二套生产程序。旧路径一一映射和哈希见`history/organization_20261007_paused/`。
- W/output保留历史图位置，旧history不再次搬迁；不为整齐移动大型正式数组、ODB、用户论文。R正式整理收据在`docs/history/pause_organization_20261007/`，入口备份在`docs/history/before_pause_20261007/`。

## 正式程序和关键科学证据（相对R）

| 路径 | 作用 |
| --- | --- |
| hyperelastic_fem.py；scripts/thin_target_explicit.py | 唯一共享能量/内力、显式入口；本轮不改 |
| surface_distance.py；pbc.py | 周期距离/占据、周期归并；本轮不改 |
| validation/README.md | 各轮科学证据索引，历史下一步不执行 |
| validation/abaqus_discrete/ | 三项独立线性HEX8矩阵对照 |
| validation/large_compression_20261005_r6/background_bridge/ | C3D8/SDVINI/UHYPER指定保存态；反力/储能1.13%/1.08%，非独立路径 |
| validation/void_continuation_20261006_r12/ | 当前C²候选、局部核/快慢20%/密采样 |
| validation/thickness_range_20261006_r13/ | diverse_28三采样厚度前向 |
| validation/geometry_transfer_20261006_r15/；mechanism_20261007_r17/ | diverse_04原迁移失败与首错/短重放 |
| validation/step_control_20261007_r18/ | 从零到17.5363%受控路径，预算停 |
| validation/shell_rate_20261007_r20/ | 同速率快壳及峰错位 |
| validation/critical_mode_20261007_r21/；fine_mode_20261007_r22/ | 未定位临界/未收敛模式，保留原失败 |
| validation/background_control_review_20261007_r23/ | 严格同背景提议审查，现为历史未执行路线 |
| validation/background_probe_20261007_r24/ | 准备与暂停事实，零编译/求解 |

535个r15～r22及15个r23文件共550个冻结文件、6个共享程序/环境文件按r24开始清单核验保持。较早科学证据和结果也不编辑；Git本就有未提交修改，不称洁净。

## Abaqus与用户资料

- 命令`E:/ABAQUS/2026/Commands/abaqus.bat`；作业根`E:/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus/`。原始INP/ODB不搬迁。
- 旧填充：`background_bridge_20261005/point_check/`及`saved20/`；后者bridge_saved20.odb为指定态检查。
- 当前快壳：`shell_rate_20261007_r20_diverse04_explicit_T0p004/thin_shell.odb`；原慢壳在`geometry_transfer_20261006_r15_diverse04_explicit_T0p040/`。
- 用户中面`F:/auto_abaqus/work/para_aly/Fine/T0p02/MS9/diverse_04/abaqus/ingredients/shell_mesh.inc`，代表diverse_28在同级；只借中面。
- 用户论文`C:/Users/xuehu/Desktop/tpms优化/`，原件不改。来源及实际阅读范围见PAPER_ROUTE，旧摘录属history。

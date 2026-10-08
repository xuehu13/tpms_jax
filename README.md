# TPMS研究入口

更新：2026-10-08，初始机制诊断r39完成。当前方法已在一个薄壁构型的三个厚度上完成20%压缩对照，但另一构型仍有明显峰位差和计算中断；通用20%计算及完整设计梯度尚未成立。

当前问题是：**diverse_04的无惯性初始刚度比同中面Abaqus壳高5.70%，为什么？** 已排除“只因加载惯性”；一个兼容模式仅微小松弛，界面变陡后同部分密积分仍偏硬5.53%。r39规则圆柱实体/壳仅差0.62%，背景保存解反而偏软；其一项基函数检查未过，保留有限判读。TPMS主因仍待区分。新图与依据见[r39报告](validation/curved_patch_20261008_r39/REVIEW.md)，界面证据见[r36–r38](validation/interface_width_mixed_20261008_r38/REVIEW.md)。

继续科研依次阅读[长期背景](docs/RESEARCH_BACKGROUND.md)、[唯一近期规划](docs/RESEARCH_PLAN.md)、[当前状态](docs/RESEARCH_STATUS.md)、[文件地图](docs/FILE_MAP.md)。本次完整原因分析见[机制审查](docs/MECHANISM_ANALYSIS.md)，方法与可行性见[综合说明](docs/TPMS_RESEARCH_REVIEW.md)，文献实际阅读范围见[PAPER_ROUTE](docs/PAPER_ROUTE.md)。

正式程序只在WSL `/home/xuehu/projects/tpms_jax`；Windows是阅读与历史目录。旧轮次报告保留原始事实，下一步只由唯一规划决定。本次提交范围、冻结核查及整理收据见文件地图。

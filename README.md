# TPMS研究入口

更新：2026-10-08，研究复盘与文献审查。本次没有新增力学求解。当前方法已在一个薄壁构型的三个厚度上完成20%压缩对照，但另一构型仍有明显峰位差和计算中断；通用20%计算及完整设计梯度尚未成立。

当前问题是：**diverse_04的无惯性初始刚度比同中面Abaqus壳高5.70%，为什么？** 已排除“只因加载惯性”，局部补积分没有改善，位移近似限制与实体/壳模型差别尚需区分。

继续科研依次阅读[长期背景](docs/RESEARCH_BACKGROUND.md)、[唯一近期规划](docs/RESEARCH_PLAN.md)、[当前状态](docs/RESEARCH_STATUS.md)、[文件地图](docs/FILE_MAP.md)。本次完整原因分析见[机制审查](docs/MECHANISM_ANALYSIS.md)，方法与可行性见[综合说明](docs/TPMS_RESEARCH_REVIEW.md)，文献实际阅读范围见[PAPER_ROUTE](docs/PAPER_ROUTE.md)。

正式程序只在WSL `/home/xuehu/projects/tpms_jax`；Windows是阅读与历史目录。旧轮次报告保留原始事实，下一步只由唯一规划决定。本次提交范围、冻结核查及整理收据见文件地图。

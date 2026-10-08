# TPMS研究入口

更新2026-10-08，实验完成至r41，r42仅作已有状态能量分析。目标仍是可信薄壁TPMS背景压缩与有效梯度，服务以后学习/逆设计；训练后置。

**近期确认：占据表示会影响初始刚度，位移重新分布占重要部分，但全部误差来源未确定。** diverse_04原平滑对壳高5.70%；同部分密积分二值使K降2.12%，仍对壳高4.24%，原27点1.47%具有积分敏感性。diverse_28当前平滑零态对壳高3.00%，其二值尚未计算。生产方法保持。

已有diverse_28三个厚度到20%的有限范围结果；diverse_04峰位与数值中断、完整20%梯度仍未解决。不能用初始差缩小外推全部大压缩。

继续前依次读[长期背景](docs/RESEARCH_BACKGROUND.md)、[唯一近期规划](docs/RESEARCH_PLAN.md)、[当前状态](docs/RESEARCH_STATUS.md)、[文件地图](docs/FILE_MAP.md)。近期推断集中在[机制审查](docs/MECHANISM_ANALYSIS.md)，方法/符号见[综合说明](docs/TPMS_RESEARCH_REVIEW.md)，文献范围见[PAPER_ROUTE](docs/PAPER_ROUTE.md)。

正式程序仅WSL `/home/xuehu/projects/tpms_jax`；Windows为阅读与历史。Word/PPT位置见[文件地图](docs/FILE_MAP.md)，它们是周总结快照。逐轮冻结报告保存原事实，不产生额外待办。

# TPMS研究阅读入口

先读[背景](docs/RESEARCH_BACKGROUND.md)、[唯一主规划](docs/RESEARCH_PLAN.md)、[状态](docs/RESEARCH_STATUS.md)、[文件地图](docs/FILE_MAP.md)。完整方法/误差/可行性判断看[综合评估](docs/TPMS_RESEARCH_REVIEW.md)，来源见[文献索引](docs/PAPER_ROUTE.md)。

2026-10-06：近期仿真替代和误差改进优先，训练后置。定位已完成，复杂三维参照后置。第2步积分因素与经典虚域候选核验证均完成，候选未采纳；虚域候选解决了冻结场的材料求值，但有大转动假能量，活动材料核已恢复。下一项仅先核对转动兼容的虚域延拓，执行只看[主规划](docs/RESEARCH_PLAN.md)。证据见[虚域核记录](docs/VIRTUAL_KERNEL_PROGRESS.md)和[JAX积分记录](docs/JAX_INTEGRATION_PROGRESS.md)。没有新20%路径、Abaqus或完整AD；壳参照及旧科学结果冻结。

代表薄壁HEX27已到20%，对慢壳反力/曲线/功差9.45%/5.87%/6.87%，速率复核通过；125点保存场检查新增软孔隙翻转限制，尚不能宣称单元内部处处有效；10%/15%完整厚度导数通过，20%保载及损失导数未过。仿真前向与梯度分别评价，不宣布全面替代。

原过程见[前向冻结报告](docs/FORWARD20_PROGRESS_REPORT.md)、[厚度路径冻结报告](docs/GRADIENT20_PROGRESS_REPORT.md)，它们不是活动计划。正式程序仅在WSL /home/xuehu/projects/tpms_jax，Windows是阅读/历史。

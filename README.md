# TPMS研究阅读入口

先读[背景](docs/RESEARCH_BACKGROUND.md)、[唯一主规划](docs/RESEARCH_PLAN.md)、[状态](docs/RESEARCH_STATUS.md)、[文件地图](docs/FILE_MAP.md)。完整方法/误差/可行性判断看[综合评估](docs/TPMS_RESEARCH_REVIEW.md)，来源见[文献索引](docs/PAPER_ROUTE.md)。

2026-10-06：近期仿真替代和误差改进优先，训练后置。本轮第1步诊断完成，第2步尖锐边界就绪、体网格未通过，第3/4步未执行；见[本轮报告](docs/SIMULATION_ERROR_PROGRESS.md)。共享求解代码和原科学文件未改，不从历史记录推导待办。

代表薄壁HEX27已到20%，对慢壳反力/曲线/功差9.45%/5.87%/6.87%，速率复核通过；10%/15%完整厚度导数通过，20%保载及损失导数未过。仿真前向与梯度分别评价，不宣布全面替代。

原过程见[前向冻结报告](docs/FORWARD20_PROGRESS_REPORT.md)、[厚度路径冻结报告](docs/GRADIENT20_PROGRESS_REPORT.md)，它们不是活动计划。正式程序仅在WSL /home/xuehu/projects/tpms_jax，Windows是阅读/历史。

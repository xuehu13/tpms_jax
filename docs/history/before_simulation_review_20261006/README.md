# TPMS研究阅读入口

目标：可信薄壁TPMS背景压缩和有效梯度，服务后续构型/参数学习与逆设计；训练后置。

依次读[背景](docs/RESEARCH_BACKGROUND.md)、[唯一主规划](docs/RESEARCH_PLAN.md)、[状态](docs/RESEARCH_STATUS.md)、[文件地图](docs/FILE_MAP.md)。方法、符号和科学判断看[综合说明](docs/TPMS_RESEARCH_REVIEW.md)，最新实际计算看[厚度路径与梯度执行记录](docs/GRADIENT20_PROGRESS_REPORT.md)。

当前HEX27薄壁显式已到20%，对慢Abaqus壳参考反力/曲线/功差9.45%/5.87%/6.87%，速率复核通过。本轮正负完整路径及JVP已执行，10%/15%导数核对通过，20%保载与损失导数未通过。20%保载JVP为+76.093668N/mm，独立差分为-10.797551N/mm，符号不一致；不能用于20%逆设计。

当前四项是原大压缩第4步，不重开边界/网格或自动启动训练。旧端点伴随候选及Newton停止保留；[前向冻结记录](docs/FORWARD20_PROGRESS_REPORT.md)和[旧PDF](docs/TPMS_RESEARCH_REVIEW.pdf)不充当最新活动计划。

正式维护和计算只在WSL /home/xuehu/projects/tpms_jax，一套FEM内力。当前结论限代表薄壁、弹性XYZ及局部厚度变化，尚不认证形态、高维反向或接触/塑性替代。大场和日志本机保留，GitHub包含精选证据。

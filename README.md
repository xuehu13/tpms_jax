# TPMS研究阅读入口

目标：可信薄壁TPMS背景压缩和有效梯度，服务后续构型/参数学习与逆设计；训练后置。

依次读[背景](docs/RESEARCH_BACKGROUND.md)、[唯一主规划](docs/RESEARCH_PLAN.md)、[状态](docs/RESEARCH_STATUS.md)、[文件地图](docs/FILE_MAP.md)。方法、符号、结果解释和科研可行性看[最新综合说明](docs/TPMS_RESEARCH_REVIEW.md)。

当前HEX27背景显式已到20%，对Abaqus慢壳参考反力/曲线/功差9.45%/5.87%/6.87%，必要加载速率复核通过。20%中心平衡和伴随求解完成，厚度扰动验证失败，可靠20%总梯度仍未认证。

下一项只推进原大压缩第4步的四项执行任务，详见主规划。本次调查/整理零新增科学作业。旧阶段报告和修改前文件在history；[冻结执行记录](docs/FORWARD20_PROGRESS_REPORT.md)保留实际过程，[旧PDF](docs/TPMS_RESEARCH_REVIEW.pdf)为此前快照，不含最新判断或活动规划。

正式程序只在WSL /home/xuehu/projects/tpms_jax维护，本目录work中的源码是阅读/执行历史。保持一套FEM，当前无接触/塑性，不因前向成功宣称全面替代。

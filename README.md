# TPMS研究阅读入口

先读[背景](docs/RESEARCH_BACKGROUND.md)、[唯一主规划](docs/RESEARCH_PLAN.md)、[状态](docs/RESEARCH_STATUS.md)、[文件地图](docs/FILE_MAP.md)。方法/符号及长期可行性看[综合评估](docs/TPMS_RESEARCH_REVIEW.md)。正式程序仅在WSL /home/xuehu/projects/tpms_jax；Windows为阅读/历史。

2026-10-06最新：客观虚域候选已接入现有显式入口，35项针对性检查通过，一次代表薄壁20%与保载完整完成，对慢壳反力/曲线/功差6.81%/4.93%/5.97%，11.17min、没有减步。但125点发现6个低占据NH混合尾部翻转点，全域材料域未过，候选未采纳默认。详见[本轮路径报告](docs/OBJECTIVE_PATH_PROGRESS.md)。默认仍NH，objective_void仅为未采纳实验选项。

下一项只处理混合虚域能量求值域，先纯核/冻结场，之后才考虑新路径及必要速率检查；按唯一主规划推进。训练、复杂三维参照和完整20%AD后置。原约10%对壳目标保留为有范围的离散证据，约5%仍是改善方向。

长期目标是可信薄壁背景压缩及有效梯度，为学习/生成/逆设计服务；当前仿真可行性有依据，尚不认证任意构型、接触/塑性或20%有效梯度。原10%/15%厚度AD通过、20%保载/损失导数失败保持。新前向结果不能自动使旧梯度有效。

冻结过程见[原前向](docs/FORWARD20_PROGRESS_REPORT.md)、[原梯度](docs/GRADIENT20_PROGRESS_REPORT.md)、[积分](docs/JAX_INTEGRATION_PROGRESS.md)、[经典虚域](docs/VIRTUAL_KERNEL_PROGRESS.md)、[客观核](docs/OBJECTIVE_VIRTUAL_PROGRESS.md)，其中旧下一步不生成任务。只维护一套FEM；新实验新目录、原字节保留，无100秒限制。

# TPMS研究阅读入口

先读[背景](RESEARCH_BACKGROUND.md)、[唯一主规划](RESEARCH_PLAN.md)、[状态](PROJECT_OVERVIEW.md)、[文件地图](FILE_MAP.md)。长期方法、符号和可行性看[综合评估](TPMS_RESEARCH_REVIEW.md)。正式程序仅WSL /home/xuehu/projects/tpms_jax，Windows阅读/历史。

2026-10-06最新：第2步材料域修复、代表快慢路径与速率门槛通过，可固定该版本进入有限工况验证。45项针对性检查及原6点局部导数核对通过；慢路径对壳反力/曲线/功差6.85%/5.05%/5.98%，快/慢10.74/21.14min。125点全域候选能量/应力有限，实际虚域折叠保留。默认仍NH，研究候选需显式选择objective_void；20%有效路径梯度未认证。见[本轮报告](VOID_CONTINUATION_PROGRESS.md)，执行仅看[唯一主规划](RESEARCH_PLAN.md)。

下一项：第3步固定该方法，少量真实厚度工况需匹配壳参考；不重新建复杂三维模型。长期目标是可信薄壁背景压缩及有效梯度，为学习/生成/逆设计服务；训练后置。当前结论限diverse_28、t/L=5%、XYZ、无接触弹性0-20%。新局部导数通过不认证20%路径梯度；原10%/15%厚度AD通过、20%失败保持。

冻结报告：[原前向](FORWARD20_PROGRESS_REPORT.md)、[原梯度](GRADIENT20_PROGRESS_REPORT.md)、[积分](JAX_INTEGRATION_PROGRESS.md)、[经典虚域](VIRTUAL_KERNEL_PROGRESS.md)、[客观核](OBJECTIVE_VIRTUAL_PROGRESS.md)、[上轮路径失败](OBJECTIVE_PATH_PROGRESS.md)。旧下一步不生成任务；只维护一套FEM，新实验新目录，无100秒限制。

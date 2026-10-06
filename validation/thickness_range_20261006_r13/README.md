# 固定方法的匹配厚度范围

本轮执行唯一主规划第3步：同diverse_28中面、XYZ周期、真实NH，0.45/0.55mm双方匹配；0.50mm复用r12。输入/门槛先固定，无数值参数拟合和新AD/训练。

解释/结论见[执行报告](../../docs/THICKNESS_RANGE_PROGRESS.md)，任务只看[主规划](../../docs/RESEARCH_PLAN.md)。input/preparation先冻结；各厚度T0p004为前向及源码/场，abaqus/explicit_T0p040为提取/留存，analysis为完整响应、模式和27/125点。decision、verification、evidence_manifest为范围及哈希。

tools仅一次性调度/分析过程，不是第二套FEM；分析复用r12保存场函数并显式对齐物理厚度和匹配壳路径。厚度范围不是梯度认证。大场、日志、重复INP及提取器副本留本机，规范0.45mm INP和preparation的厚度单行差异可复现0.55mm；ODB留E盘独立作业目录。冻结原r6-r12不改。

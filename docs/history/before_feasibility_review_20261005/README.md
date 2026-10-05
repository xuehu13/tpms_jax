# TPMS研究阅读入口

更新：2026-10-05。目标是可信薄壁TPMS背景压缩与有效梯度，后续服务学习/逆设计；当前先解决大压缩，训练后置。

依次阅读 [背景](docs/RESEARCH_BACKGROUND.md)、[唯一主规划](docs/RESEARCH_PLAN.md)、[当前状态](docs/RESEARCH_STATUS.md)、[文件地图](docs/FILE_MAP.md)。理论、进度和大压缩/Explicit判断集中在 [综合报告](docs/TPMS_RESEARCH_REVIEW.md)，[PDF版](docs/TPMS_RESEARCH_REVIEW.pdf)为前次方法调查快照；其中旧四步已替换，执行只看主规划。

t/L=0.05、XYZ周期的旧四步已收口。新四步已经开始：Abaqus三条至20%诊断完成，JAX静力15%遇线性求解阻塞，物理显式候选已在目标N64上实际尝试，首次有效到16.2139%后保护停止，减步重试已完成20%及保载；首个HEX8候选20%反力差22.41%；同厚度/材料/XYZ的二次HEX27背景候选改善至9.45%，整段曲线/功差5.87%/6.87%。加载时间加倍复核通过，第3步响应工作目标已达到；第4步已实际尝试：20%中心平衡和伴随线性求解完成，厚度扰动重平衡未通过；20%总梯度未认证，完整路径扰动未执行。最新事实和位置见 [本轮推进报告](docs/FORWARD20_PROGRESS_REPORT.md)，详细审计后置，基本物理检查随计算。

正式程序在WSL /home/xuehu/projects/tpms_jax；本目录是阅读/历史。维护一套FEM，静力和新显式入口共用材料内力、Gauss积分和XYZ周期。当前没有接触/塑性；显式尝试不自动认证大压缩准确性或梯度。

旧报告、原数据、运行快照冻结；8项旧Git传输材料已集中归档，映射见 [历史说明](docs/history/README.md)。文献原件及Abaqus旧作业未改。

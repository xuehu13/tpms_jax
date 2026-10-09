# TPMS研究入口

更新2026-10-09：最新为[r46近期复盘](validation/binary_mesh_reassessment_20261009_r46/REVIEW.md)，本次零新FEM/Abaqus/AD。N32 T4ms与N64 T1ms不能作纯网格排名；N64只至13.59%、没有捕获峰。小压缩接近不等于20%路径通过。

长期目标仍为可信薄壁TPMS背景压缩与有效梯度，服务以后学习/逆设计；仿真优先。代表diverse_28已有有限范围20%结果，迁移diverse_04峰附近仍有明显差异。生产连续占据与唯一WSL程序保持，硬二值只作诊断。

继续前依次读[长期背景](docs/RESEARCH_BACKGROUND.md)、[唯一近期规划](docs/RESEARCH_PLAN.md)、[当前状态](docs/RESEARCH_STATUS.md)、[文件地图](docs/FILE_MAP.md)。[机制审查](docs/MECHANISM_ANALYSIS.md)解释判断，[综合说明](docs/TPMS_RESEARCH_REVIEW.md)解释方法/符号，[文献索引](docs/PAPER_ROUTE.md)标明实际读取范围。下一轮四步尚未执行，不自动接续N64或恢复N32。

正式程序仅`/home/xuehu/projects/tpms_jax`；Windows为阅读与历史。旧报告、失败与预算保持；Word/PPT为周总结快照。Git保存精选配方/摘要/图，原大场、日志、INP/ODB留本机，位置只见文件地图。

# TPMS研究入口

更新2026-10-11。当前已完成一轮共同中面端部压板、XY周期、自由横向伸缩的跨构型对照。**G、P的0–20%宏观压缩响应达到本轮约10%的工作目标，足以支持下一阶段在较小压缩范围验证参数梯度。** D初段总力接近，但局部变形分歧较大，壳在约18.4%失败，不能宣布D完整验证通过。

先读[当前状态](docs/RESEARCH_STATUS.md)，再读[近期四步建议](docs/RESEARCH_PLAN.md)。新建议以12%为压缩上限，先厚度、再界面参数、最后一个有明确几何意义的构型系数；用户已授权执行，r53先验证G5%厚度总导数；实际结果/停点见状态页。

- [长期研究背景](docs/RESEARCH_BACKGROUND.md)：可信TPMS计算及有效设计梯度，训练继续后置。
- [方法与可导性说明](docs/TPMS_RESEARCH_REVIEW.md)：几何、材料、积分、显式、接触及梯度链。
- [G完整结果及接触图解](validation/platen_xy_G_20261010_r51/experiments/midsurface_rim_20pct_20261010/README.md)；[P/D结果与D失败记录](validation/platen_xy_PD_20261010_r52/README.md)。
- [文件地图](docs/FILE_MAP.md)：正式程序、原始数据、INP/ODB和历史快照位置。

正式程序仅WSL `/home/xuehu/projects/tpms_jax`；Windows用于阅读与历史。原XYZ结果、原Word/PPT及学习包保持日期快照。当前梯度任务在r53独立执行，新接触模型的完整设计梯度尚待本轮核对。

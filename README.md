# TPMS研究阅读入口

更新：2026-10-05。目标是可信薄壁TPMS背景压缩与有效梯度，后续服务学习/逆设计；当前先解决大压缩，训练后置。

依次阅读 [背景](docs/RESEARCH_BACKGROUND.md)、[唯一主规划](docs/RESEARCH_PLAN.md)、[当前状态](docs/RESEARCH_STATUS.md)、[文件地图](docs/FILE_MAP.md)。理论、进度和大压缩/Explicit判断集中在 [综合报告](docs/TPMS_RESEARCH_REVIEW.md)，[PDF版](docs/TPMS_RESEARCH_REVIEW.pdf)为同一内容导出。

t/L=0.05、XYZ周期的旧四步已收口：小变形及1%/5%前向初筛、厚度梯度通过；10%差14.19%与参考质量未过；20%未提交。新四步全部未启动：10%审查 → 匹配Explicit参考 → 按机制补最小能力 → 大压缩前向/一项梯度。

正式程序在WSL /home/xuehu/projects/tpms_jax；本目录是阅读/历史。当前程序静力弹性，无接触/塑性/物理Explicit；官方示例有相关基础，尚不是项目能力。本次仅整理和调查，没有新作业。

旧报告、原数据、运行快照冻结；8项旧Git传输材料已集中归档，映射见 [历史说明](docs/history/README.md)。文献原件及Abaqus旧作业未改。

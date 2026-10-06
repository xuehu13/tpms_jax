# TPMS研究阅读入口

依次读[研究背景](docs/RESEARCH_BACKGROUND.md)、[唯一主规划](docs/RESEARCH_PLAN.md)、[当前状态](docs/RESEARCH_STATUS.md)、[文件地图](docs/FILE_MAP.md)。想完整了解方法、数字及可行性，读[综合说明](docs/TPMS_RESEARCH_REVIEW.md)。

上一轮四步已完成并冻结：同diverse_28中面三个采样厚度至20%，最大反力/曲线/功差8.28%/7.48%/7.31%。当前diverse_04四步的第1步限定输入就绪已完成；原4条法向末点失败/首出口诊断分别留存。第2步已执行但迁移目标未通过，JAX中止、壳质量未过；第3/4步限定诊断与范围收口已完成（r16），问题处理留下一轮决定，不自动修复或追加作业。20%新核完整路径梯度仍未认证，训练后置；新计划不会自动启动AD、其他构型或材料扫描。

正式程序仅WSL `/home/xuehu/projects/tpms_jax`。Windows是阅读和历史，work仅放当前实验/整理收据；已完成工具归档，不作为运行入口。最新科学细节见[厚度执行报告](docs/THICKNESS_RANGE_PROGRESS.md)，核修复见[冻结执行报告](docs/VOID_CONTINUATION_PROGRESS.md)，论文阅读范围见[来源索引](docs/PAPER_ROUTE.md)。

本轮主要发现与范围见[本轮诊断与收口](validation/geometry_transfer_review_20261006_r16/REVIEW.md)：双方响应转折差异早于JAX数值拒绝；未完成的20%验证不算通过。

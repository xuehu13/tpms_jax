# TPMS研究阅读入口

先读 [综合报告：薄壁目标、方法、当前证据与可行性](docs/TPMS_RESEARCH_REVIEW.md)，可分享版本为 [PDF](docs/TPMS_RESEARCH_REVIEW.pdf)。已按2026-10-04用户纠正补充真实厚度、薄壁风险、不同压缩机制和梯度判断。

继续工作按 [研究背景](docs/RESEARCH_BACKGROUND.md) → [唯一近期主规划](docs/RESEARCH_PLAN.md) → [当前状态](docs/RESEARCH_STATUS.md) → [文件地图](docs/FILE_MAP.md)。最新目标厚度改为单胞边长5%（L=10mm时t=0.5mm）。四步为目标薄壁表示、小变形壳对照、分段压缩范围、有效梯度；第1步表示通过；第2步原平端未通过、XYZ周期诊断通过；用户已统一XYZ，原平端差异后置，第3步按判据收口：1%/5%三向周期有限应变初筛通过，反力差5.34%/7.89%；10%可求解但反力差14.19%及参考能量质量未过，20%未提交。下一项为第4步，在1%已通过点优先核对厚度平衡总梯度；不启动训练或重开边界/网格扫描。先回答目标相关可行性，完整精度收敛/训练后置；约15分钟主要计算尝试可接受。此前中等厚度Gyroid的成本/N48/N64/半步计划未执行、已替换。

小变形整体响应已有证据；完整均匀实体20%通过；Gyroid精细网格和独立非线性精度尚未认证。[文献记录](docs/PAPER_ROUTE.md)只作来源索引，不另立任务。

冻结报告：[第一轮](validation/near_term_20261003/README.md)、[第二轮](docs/ROUND2_REPORT.md)、[第三轮](docs/ROUND3_REPORT.md)、第四轮 [参考审计](docs/ROUND4_STEP1_REPORT.md)／[构型筛查](docs/ROUND4_STEP2_REPORT.md)／[均匀体有限应变](docs/ROUND4_STEP3_REPORT.md)／[Gyroid预检查](docs/ROUND4_STEP4_REPORT.md)。旧报告中的“下一步”不是当前待办。

正式程序在WSL `/home/xuehu/projects/tpms_jax`；Windows主要为阅读与历史。归档按 [历史索引](docs/history/README.md)查询，当前工作按主规划推进。

程序职责见正式`scripts/README.md`；本次维护记录见 [状态](docs/RESEARCH_STATUS.md)。Git保存维护源码与关键证据摘要，本机冻结快照/原始输出另按文件地图取用，不能误认克隆仓库就是完整实验档案。

[第1步](validation/thin_target_20261004_r5/README.md)为表示检查，[第2步](validation/thin_target_20261004_r5/step2/README.md)保留小变形历史；最新 [第3步](validation/thin_target_20261004_r5/step3_xyz/README.md)给出1%/5%通过、10%未接受及停止记录。

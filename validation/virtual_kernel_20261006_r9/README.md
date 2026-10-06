# 一个虚域候选的理论与核验证（冻结）

结论与符号见[完整记录](../../docs/VIRTUAL_KERNEL_PROGRESS.md)，活动步骤仅看[唯一规划](../../docs/RESEARCH_PLAN.md)。

一个经典Wang型候选的实体极限、参考刚度、局部导数及原回归共16项通过；原20%冻结位移的125点材料映射有定义，但整体转动有明显人工能量，候选未采纳。原活动材料核恢复。

输入/环境/哈希在input.json、probe_manifest.json、evidence_manifest.json；结果在rule_4.json、rule_8.json、result.json、checks.json、decision.json。candidate.patch、check_at_run.py、experiment.py为冻结复现记录，不作当前入口；复现路径与限制见完整记录。本地source_*和probe.log保留，完整求解器快照不重复上传。

本轮零新完整TPMS路径、Abaqus、完整路径AD或训练；不同采样/转角不是独立物理案例。不得将材料映射J当实际体积比，或将同状态内力量当重新平衡后的精度。

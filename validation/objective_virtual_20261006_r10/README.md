# 客观虚域核与冻结场验证（非新压缩路径）

完整解释见[报告](../../docs/OBJECTIVE_VIRTUAL_PROGRESS.md)，只按[唯一主规划](../../docs/RESEARCH_PLAN.md)推进。28项核/回归和27/125点保存场探针通过；候选未接入默认求解器，实际深虚域翻转保留，尚无新精度/路径梯度结论。输入、环境、源码、结果/日志的哈希见probe_manifest和evidence_manifest；本机大场仍在原位置。

唯一候选维护在根hyperelastic_fem.py，测试为tests/test_objective_void.py。experiment.py是冻结场诊断，不是第二套求解器；复现需对应候选源码及本机旧大场，并指定全新--output。旧27点原能量/内力恢复一致；125点原NH总量保留null，没有正J子域冒充总量。

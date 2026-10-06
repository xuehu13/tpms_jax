# C²混合虚域：核、冻结场与快慢20%路径

解释见[执行报告](../../docs/VOID_CONTINUATION_PROGRESS.md)，执行只看[主规划](../../docs/RESEARCH_PLAN.md)。45项针对性检查、原6点局部AD/FD、两条20%与保载、必要速率和新125点材料域完成，通过当前有限范围门槛；虚域实际负J保留，默认NH，候选需显式选择objective_void。不是接触/塑性、实体真值或20%路径梯度认证。

材料仅hyperelastic_fem.py、显式仅scripts/thin_target_explicit.py；experiment.py/six_points.py为分析，source_*仅冻结字节。公式/参数先固定在input.json；初次FD尺度和额外计数推断问题见kernel_admission/checks_initial.log/source_attempt，候选常数未调。来源、命令、环境、结果和旧科学哈希见各manifest/receipt和evidence_manifest。大场/日志本机保留，复核需对应版本和新目录，不覆盖本目录。

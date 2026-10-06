# 一次客观虚域20%与保载：改善，材料域尚未过

完整解释见[本轮报告](../../docs/OBJECTIVE_PATH_PROGRESS.md)，执行只看[主规划](../../docs/RESEARCH_PLAN.md)。35项针对性检查通过、一次完整路径、零Abaqus/全AD。反力/曲线/功差6.81%/4.93%/5.97%、11.17min、零减步；125点6个NH混合尾部负J使完整能量无定义，候选未采纳默认。

唯一材料核hyperelastic_fem.py、唯一显式入口scripts/thin_target_explicit.py；objective_void是未采纳实验选项，默认nh。输入/环境/源码/结果哈希见run_manifest、analysis_manifest、evidence_manifest。experiment.py仅做保存场分析，locate_domain.py只定位失效点，不维护求解器。复核需本机原大场、对应源码和新实验目录，不能覆盖本目录；旧r6至r10保持。

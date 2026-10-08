# 程序用途索引

正式维护程序仅在本仓库。本页不复制状态或后续任务；当前证据看[状态](../docs/RESEARCH_STATUS.md)，方法看[综合说明](../docs/TPMS_RESEARCH_REVIEW.md)，启动顺序只看[唯一规划](../docs/RESEARCH_PLAN.md)。默认NH保持，objective_void/C²须明确选用。

| 入口 | 用途与边界 |
| --- | --- |
| thin_target_explicit.py / extract_thin_explicit.py | 唯一当前显式入口/只读ODB提取；共享材料、中央差分、XYZ、Q2/HRZ；部分20%前向有证据，完整设计AD未认证 |
| thin_target_linear.py / extract_thin_shell.py | 冻结早期线性对照与模式；不能自动外推大压缩 |
| thin_target_finite.py / extract_thin_finite.py | 冻结静力分段与提取；原15%线性求解阻塞保持 |
| prepare_thin_target.py | 周期中面距离/恒厚占据，无位移求解；完成目录不得覆盖 |
| thin_thickness_gradient.py | 原低压缩保存态/厚度反力伴随，非当前20%梯度入口 |
| finite_strain_uniform.py / extract_finite_strain.py | 均匀实体有限应变基准，不认证TPMS |
| finite_strain_gyroid.py | 冻结中等厚度预检查，非当前恒厚目标 |
| m1/m2/m3/m4_* | 早期几何/周期线性与数值分析，参数c不等于物理厚度 |
| prepare/run/extract_abaqus_binary、capture/check_* | 对应冻结验证；输入、结果和范围查[实验索引](../validation/README.md) |

当前生产配置N32 HEX27/27点。`--quadrature-order`是已有Basix规则参数，数值4对应本项目原27点配置，并不表示64点。变积分规则须在新Problem实际Gauss点重新求占据，不能复用旧规则数组。`--gauss-field`与`--surface-geometry`互斥；后者从同中面在自身Gauss点求场。

`--geometry-on-cpu`、`--force-batch-cells`只管理参考预处理/内力资源，不改物理。`--case-input`只读取匹配物理参数，不能把输入就绪当响应验证。`--checkpoint-compressions`保存原接受块；`--diagnose-first-failure`在首拒绝块单步定位；`--replay-state`只重放原短时间窗，不能拼成完整路径。

`--stability-states`只评估保存状态保守步长；`--state-step-control`为已有opt-in前向诊断，保持原质量/材料/内力，含有限回退及旧质量门槛，不等于当前推荐的生产路径。状态频率上界不是精确谱；这些控制分支的设计导数未认证。不要从历史命令自动启动N64、长路径或重复模式迭代。

完整运行命令留在相应冻结protocol/报告；当前独立初始/局部诊断在validation原目录，不建立另一个通用求解框架。`DensityHyperelasticity._set_params_jax`只供已验证字段追踪，固定状态材料偏导不等于平衡/完整路径设计梯度。

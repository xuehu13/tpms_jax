# 程序入口与范围

维护FEM只有根目录的一套实现；不从validation/source_*或Windows历史副本执行。当前研究任务只看 [唯一主规划](../docs/RESEARCH_PLAN.md)。5%目标第1步通过；第2步原XY平端未过、XYZ周期诊断通过；用户已统一XYZ；第3步1%/5%通过、10%未接受，第4步1%/5%厚度总梯度及两点损失通过，四步限定收口；下面历史入口不是批量待办。

| 入口 | 用途/边界 |
| --- | --- |
| thin_target_explicit.py、extract_thin_explicit.py | 新NH/XYZ物理显式与只读ODB提取；wave/probe/target，目标减步到20%；当前前向入口，20%设计梯度/接触未认证 |
| thin_thickness_gradient.py | 第4步保存状态反力伴随、厚度扰动与两点诊断损失；复用原求解器，形态/任意目标/训练未认证 |
| thin_target_finite.py、extract_thin_finite.py | 静力分段XYZ压缩/只读ODB；原结果冻结，新输出/上一状态参数支持独立续算；本轮15%线性求解阻塞 |
| thin_target_linear.py、extract_thin_shell.py | 本轮两组弹性对照/模式，唯一XYZ边界诊断开关与只读ODB提取；复用既有FEM，不能自动转为大变形 |
| prepare_thin_target.py | 第1步周期三角距离/恒厚占据及真实N64 Gauss检查，拒绝覆盖完成目录；没有位移求解 |
| m1_volume.py、m2_*_cube.py、m3_gyroid_first.py／m3_lateral_relaxation.py、m4_numerical_study.py | 早期几何/线性基准，参数名c不是恒定物理厚度 |
| capture_binary_projection_reference.py | 在Problem自身Gauss点评估解析或有符号数组输入；旧占据数组仅保留诊断 |
| prepare_abaqus_binary.py、run_abaqus_binary.ps1、extract_abaqus_binary.py | 二值贴体小变形独立对照；薄壁裁剪限制保留 |
| finite_strain_uniform.py、extract_finite_strain.py | 完整均匀实体匹配Neo-Hookean基准及专用ODB提取；不认证TPMS |
| finite_strain_gyroid.py | 已冻结的中等厚度Gyroid预检查定义；不作为目标薄壁输入 |
| check_design_gradients.py及其他capture/prepare/extract | 对应历史验证；用途、输入及执行结果查validation索引 |

`DensityHyperelasticity.set_params`验证具体输入；`_set_params_jax`仅为已验证输入的可追踪字段更新。其固定状态残差导数检查不等于平衡解伴随、薄壁有效梯度或训练完成。本轮专项反力伴随已复用同一求解器完成厚度检查，任意损失/形态接口以后实际需要再验证，不搭通用框架。

## 2026-10-05状态更新

JAX显式减步在同一N64薄壁上完成20%及保载；平均反力幅值2.241863N，对慢Explicit差22.41%，整段曲线RMS/参考特征峰值差13.47%，输入功差15.51%。实际主体耗时18.57分钟，1次无效块被拒绝，终态动能/弹性储能0.00005%，能量与输入功差0.008%。 维护一套内力实现，新增显式入口未修改核心FEM/材料/周期/环境。20%设计参数求导接口未完成。新实验见validation/large_compression_20261005_r6；旧薄壁结果冻结。

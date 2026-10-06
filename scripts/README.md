# 程序入口与范围

唯一维护FEM在正式根目录；下列是功能/历史索引，不是自动待办。当前只按[唯一主规划](../docs/RESEARCH_PLAN.md)执行：客观虚域核与冻结场检查通过，下一项一次20%受控路径。候选未接入默认入口，训练和完整20%AD后置。

| 入口 | 用途/边界 |
| --- | --- |
| thin_target_explicit.py、extract_thin_explicit.py | 新NH/XYZ物理显式与只读ODB提取；wave/probe/target，目标减步到20%；前向及可追踪Q2材料/HRZ接口；20%局部路径证据见新报告，接触未认证 |
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

## 当前任务与证据

唯一活动步骤见[主规划](../docs/RESEARCH_PLAN.md)。最新[客观虚域核记录](../docs/OBJECTIVE_VIRTUAL_PROGRESS.md)已通过；原积分/经典虚域候选未采纳。原20%对壳约10%一致性及125点虚域翻转限制保留。

`thin_target_explicit.py`仍复用同一材料/FEM内力和中央差分/HRZ。新增`--quadrature-order`暴露已有Basix支持，`--surface-geometry`在新Problem自身Gauss点计算同一中面距离占据；它与`--gauss-field`互斥。改变规则不能复用旧规则的占据数组。`--geometry-on-cpu`和`--force-batch-cells`用于实际参考预处理显存/编译开销，默认行为保留，未增加物理稳定化。

本轮64点代表命令（WSL仓库内）：

```sh
.pixi/envs/default/bin/python scripts/thin_target_explicit.py target \
  --element-degree 2 --cells 32 --quadrature-order 6 \
  --surface-geometry validation/thin_target_20261004_r5/gauss_field.npz \
  --geometry-on-cpu --force-batch-cells 128 --adaptive --load-time 0.004 \
  --output validation/jax_improvement_20261006_r8/T0p004_q6_cpu_geometry
```

输出拒绝覆盖已有目录，复现请使用新目录。高阶参考映射GPU预处理曾显存不足，原失败日志保留；CPU预处理沿用原Problem，不维护第二套FEM。材料、厚度、界面、η和加载未改。经典虚域候选仅在共享核中实验验证后冻结为补丁；原NH函数和求解器保持；本轮客观候选为hyperelastic_fem.py的未接入纯核，没有新显式材料分支。

旧前向/梯度/诊断分别见[前向报告](../docs/FORWARD20_PROGRESS_REPORT.md)、[梯度报告](../docs/GRADIENT20_PROGRESS_REPORT.md)、[冻结定位](../docs/SIMULATION_ERROR_PROGRESS.md)；旧建议不生成新任务。

# 程序入口与范围

唯一维护FEM在正式根目录。本页为现有功能/历史入口索引，不生成待办。当前C²候选的代表20%前向/材料域/速率通过，旧四步已收口，新一轮diverse_04单一几何匹配四步待执行，下一项仅[主规划](../docs/RESEARCH_PLAN.md)第1步建立双方输入；方法、数字和可行性看[综合说明](../docs/TPMS_RESEARCH_REVIEW.md)。默认NH保持，研究候选明确选择objective_void，完整20%AD/训练后置。

| 入口 | 用途/边界 |
| --- | --- |
| thin_target_explicit.py、extract_thin_explicit.py | 当前NH/可选C²虚域、XYZ显式与只读ODB；wave/probe/target，Q2/HRZ；代表前向已到20%，完整20%设计梯度仍未认证 |
| thin_thickness_gradient.py | 冻结早期保存状态反力伴随、厚度扰动与两点损失；低压缩原范围，不是当前第4步执行入口或20%有效梯度 |
| thin_target_finite.py、extract_thin_finite.py | 静力分段XYZ压缩/只读ODB；原结果冻结，新输出/上一状态参数支持独立续算；原15%线性求解阻塞 |
| thin_target_linear.py、extract_thin_shell.py | 冻结原两组弹性对照/模式，唯一XYZ边界诊断开关与只读ODB提取；复用既有FEM，不能自动转为大变形 |
| prepare_thin_target.py | 原周期三角距离/恒厚占据及真实N64 Gauss检查，拒绝覆盖完成目录；没有位移求解 |
| m1_volume.py、m2_*_cube.py、m3_gyroid_first.py／m3_lateral_relaxation.py、m4_numerical_study.py | 早期几何/线性基准，参数名c不是恒定物理厚度 |
| capture_binary_projection_reference.py | 在Problem自身Gauss点评估解析或有符号数组输入；旧占据数组仅保留诊断 |
| prepare_abaqus_binary.py、run_abaqus_binary.ps1、extract_abaqus_binary.py | 二值贴体小变形独立对照；薄壁裁剪限制保留 |
| finite_strain_uniform.py、extract_finite_strain.py | 完整均匀实体匹配Neo-Hookean基准及专用ODB提取；不认证TPMS |
| finite_strain_gyroid.py | 已冻结的中等厚度Gyroid预检查定义；不作为目标薄壁输入 |
| check_design_gradients.py及其他capture/prepare/extract | 对应历史验证；用途、输入及执行结果查validation索引 |

`DensityHyperelasticity.set_params`验证具体输入；`_set_params_jax`仅为已验证输入的可追踪字段更新。其固定状态残差导数检查不等于平衡解伴随、薄壁有效梯度或训练完成。早期专项反力伴随只在原报告限定范围完成，当前20%有效导数未认证；以后接口按实际需求验证，不搭通用框架。

## 当前任务与证据

唯一活动步骤见[主规划](../docs/RESEARCH_PLAN.md)。最新理论、公式和结果见[C²虚域报告](../docs/VOID_CONTINUATION_PROGRESS.md)，上轮6点失败见[冻结路径报告](../docs/OBJECTIVE_PATH_PROGRESS.md)。objective_void只用于显式，静力旧提取拒绝该选项。required_positive_J_*表示未延拓原NH域，不沿用旧NH_active_*的混合域含义。

`thin_target_explicit.py`仍复用同一材料/FEM内力和中央差分/HRZ。新增`--quadrature-order`暴露已有Basix支持，`--surface-geometry`在新Problem自身Gauss点计算同一中面距离占据；它与`--gauss-field`互斥。改变规则不能复用旧规则的占据数组。`--geometry-on-cpu`和`--force-batch-cells`用于实际参考预处理显存/编译开销，默认行为保留，未增加物理稳定化。

冻结0.50mm代表复现命令（WSL仓库内，输出必须使用全新目录）：

```sh
.pixi/envs/default/bin/python scripts/thin_target_explicit.py target \
  --material-model objective_void --element-degree 2 --cells 32 --quadrature-order 4 \
  --gauss-field validation/large_compression_20261005_r6/quadratic_candidate/gauss_field.npz \
  --force-batch-cells 2048 --adaptive --load-time 0.004 --output YOUR_NEW_DIRECTORY
```

代表新保存场27/125点材料域已检查，具体门槛看报告；有限采样不证明处处有效。慢路径只将load-time改为0.008，其他物理输入/算法相同。大场留本机；省略material-model即默认原NH。旧64点命令和资源处理见冻结[JAX积分报告](../docs/JAX_INTEGRATION_PROGRESS.md)，不作为当前活动任务。

旧前向/梯度/诊断分别见[前向报告](../docs/FORWARD20_PROGRESS_REPORT.md)、[梯度报告](../docs/GRADIENT20_PROGRESS_REPORT.md)、[冻结定位](../docs/SIMULATION_ERROR_PROGRESS.md)；旧建议不生成新任务。

匹配厚度r13沿用同一入口，在上列代表命令加`--thickness-mm 0.45`或`0.55`并使用全新输出。占据和HRZ质量随真实厚度计算；不能改厚度后仍对原0.50mm壳。新核/时间算法未改，r13原证据不覆盖；读取详细口径见[厚度报告](../docs/THICKNESS_RANGE_PROGRESS.md)。

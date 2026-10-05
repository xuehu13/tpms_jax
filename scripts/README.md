# 程序入口与范围

唯一维护FEM在正式根目录，不从validation/source_*或Windows归档执行。当前[唯一主规划](../docs/RESEARCH_PLAN.md)将仿真误差与三维参照优先，训练后置；第1步已完成、第2步体网格未通过、第3/4步未执行。下面各入口是功能/历史范围索引，不是批量待办。

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

HEX27前向20%约10%工作目标通过；原完整JVP的10%/15%导数通过，20%保载/损失导数未过，不能用于20%逆设计。本轮新增保存场/厚度诊断和有限次参照网格尝试，共享求解代码未改。当前改进顺序见[唯一规划](../docs/RESEARCH_PLAN.md)，不是持续重跑梯度或启动训练。

前向证据见[冻结报告](../docs/FORWARD20_PROGRESS_REPORT.md)，导数失败见[冻结记录](../docs/GRADIENT20_PROGRESS_REPORT.md)。能量插值尚未实现，原默认NH、HRZ与科学原件保留。

本轮实际执行见[误差定位与三维参照准备](../docs/SIMULATION_ERROR_PROGRESS.md)。新证据在validation/simulation_error_20261006_r7，不覆盖原r6。

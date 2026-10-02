# Abaqus 2026 / JAX 单元算子比较

2026-10-02 实际运行两个无约束单元的矩阵生成作业，均完成，错误和警告均为 0。
材料 E=10、ν=0.3、ρ=1，节点采用标准 HEX8 顺序。比较的是完整 24×24
刚度矩阵，没有施加边界条件、周期约束或加载；六个刚体零模态保留。

| 几何 | ‖K_Abaqus−K_JAX‖F / ‖K_JAX‖F | JAX 与独立全积分 | Abaqus 与独立 B̄ |
|---|---:|---:|---:|
| 单位规则立方体 | 0.2829196402 | 3.87e-16 | 4.97e-15 |
| 轻微畸变 HEX8 | 0.2833124232 | 3.51e-16 | 4.84e-15 |

JAX 矩阵来自实际安装的 `jax-fem==0.0.12` 与项目的
`DensityLinearElasticityPeriodic.newton_update`，不是用 NumPy 矩阵冒充。
独立积分验证其对应 K=Σ Bqᵀ D Bq JxWq。B̄ 将 B 的三个法向行各增加
(单元体积加权平均的体积行−当前体积行)/3，剪切行保持原值。
该预测与实际 Abaqus C3D8 矩阵吻合；两个软件在九个仿射梯度模式上的
矩阵作用仍一致至约 1.6e-16。两套矩阵均具有六个零模态和十八个正模态。

规则单元测试位移 ux=(x−0.5)(y−0.5)、uy=uz=0 时，JAX 能量为
0.7211538462，Abaqus 为 0.3739316239，相差 48.15%。这是指定非仿射模式
的单元能量差，不是 Gyroid 压缩试验误差；矩阵的 28.29% 也不能解释成
宏观反力误差。均匀实体的仿射基准通过，不能证明两个离散算子相同。

该结论目前覆盖这两种几何的均匀材料。非均匀材料需另做积分点及算子
检查，不从此表外推。C3D8R/C3D8I/C3D8H、UMAT 或 USDFLD 本身均不能
作为已证明的等价替代。

官方理论说明了 C3D8 对体积应变的修正，官方矩阵生成说明了无约束矩阵
导出格式：[单元理论](https://docs.software.vt.edu/abaqusv2025/English/SIMACAETHERefMap/simathe-c-solidisoquadhex.htm)、
[矩阵生成](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEANLRefMap/simaanl-c-mtxgenerationperturbation.htm)。
公开网页为 2025 文档，2026 行为由本目录实际作业验证。

复现步骤：

```bash
# WSL 项目环境；仅生成 INP
.pixi/envs/default/bin/python scripts/abaqus_element_comparison.py prepare --output results/element_check
```

```powershell
# Windows，在包含这两份 INP 的新目录运行；实际提交矩阵作业
abaqus job=c3d8_regular input=c3d8_regular.inp interactive cpus=1
abaqus job=c3d8_distorted input=c3d8_distorted.inp interactive cpus=1
```

检查 `.sta/.dat/.msg`，将 `*_STIF1.mtx` 复制至准备目录后：

```bash
.pixi/envs/default/bin/python scripts/abaqus_element_comparison.py compare --directory results/element_check --output results/element_check/comparison.json
```

`comparison.json` 的 `checks_pass=true` 表示“差异已被量化且由积分定义
解释”，不是等价验收；两工况的 `same_discrete_operator=false`。
原始 `.mtx`、矩阵数组、完成状态和文件 SHA256 均已保存。

# 同一 M4 线性离散问题的 Abaqus 验证

> 历史阶段记录：以下结果、测试数量及“下一步”保留当时口径。当前状态和后续顺序以 [当前研究目标与生成式路线](../../docs/RESEARCH_STATUS.md) 为准；本次未改动该阶段原始数值证据。

采用独立 NumPy 代码计算每个 HEX8 的 K_e=Σ Bqᵀ D(Eq,ν) Bq JxWq，
Eq=E_min+ρq(E_s−E_min)，ρ 为 Gyroid 双 sigmoid 投影。导出前核对完整
稀疏算子与实际安装的 M4 JAX 切线，并在真实 Gauss 点核对密度公式。
矩阵不是从 JAX 复制出来的。Abaqus 2026 通过 `*USER ELEMENT, LINEAR`
和 `*MATRIX, TYPE=STIFFNESS` 导入各单元矩阵，独立组装和求解。

统一参数：L=1、c=0.541062、β=20、E_s=10、E_min/E_s=1e-3、ν=0.3、
εz=−0.01。XY 周期，上下平面保持平整；固定工况 εx=εy=0，自由工况
QX/QY 未施加位移或力，横向广义力自然为零。周期从节点均直接约束到
独立主节点，角点不重复消元；顶面 uz=QZ，底面 uz=0，角点去除水平平移。

验收阈值在运行前固定：反力、重建能量相对差异 ≤1e-6，全部物理节点
位移绝对差异 ≤1e-6×0.01=1e-8；自由工况检查零横向广义力，并检查宏观
位移与反力功。不是只核对总反力。

N=4 固定、N=16 固定、N=16 自由共三项实际作业已验收通过，最终数据见
各 `.acceptance.json`、`.reference_summary.json` 和 `run_manifest.json`。
N=16 独立矩阵相对差异为 2.47e-15，密度最大差异 1.33e-15。

参考求解显式使用 JAX-FEM 的 `spsolve_solver` 和 Newton 1e-12 容限。
首次使用默认迭代参考时，N=16 自由工况的最大位移差异为 1.508e-8，
因此未通过；保持 INP 的 SHA256 完全一致并使用同一方程的直接解后，
位移差异降为 4.649e-10。参考缩减残差从 2.13e-12 降为 1.21e-17。
没有放宽验收标准或重跑 Abaqus 来改变结果。首次失败的报告、参考摘要和
精度诊断均保存。`fem.solve`、`solve_lateral_relaxation`、`solve_case` 新增
可选 `solver_options`；未指定时维持原默认行为。

此路径验证当前线性、固定材料场的离散方程、独立积分/组装、周期约束、
求解和宏观响应。它不检验 Abaqus 原生 C3D8 本构积分，不是 USDFLD 验证，
也不是去掉孔隙的真实实体模型。用户线性单元没有原生 S/E/IVOL 输出，
本报告的能量来自独立 K_e 与 Abaqus 节点 U；另与控制点 0.5 RF·U 对照，
不宣称为 Abaqus 原生 ALLSE。需要原生应力时另开 C3D8+USDFLD 的响应
对照，并保留单元公式差异的解释。

实际 `.dat` 每个单元均出现已知的“用户单元不支持单元输出”提示：N=4
共 64 条，N=16 共 4096 条。错误为 0，没有其他类型警告。
批处理设置 `ABA_OUTPUT_DIAGNOSTICS=unlimited`，逐项核查，不能将默认
“后续警告被抑制”视为已核查。错误、未知警告、不完整 STA 或任一数值
验收失败都会停止批次；不依赖 Abaqus bat 的退出码。

复现：

```bash
# WSL；生成模型和高精度 JAX 参考，不提交 Abaqus
.pixi/envs/default/bin/python scripts/prepare_abaqus_discrete.py --output results/abaqus_discrete --n 16 --lateral fixed
.pixi/envs/default/bin/python scripts/prepare_abaqus_discrete.py --output results/abaqus_discrete --n 16 --lateral relaxed_free
```

把生成目录复制到新的 Windows 作业目录，附上 `scripts/` 下
`extract_abaqus_discrete.py` 和 `extract_uniform_baseline.py` 两份文件，然后：

```powershell
& scripts\run_abaqus_discrete.ps1 -PackageDirectory 'E:\YOUR_NEW_JOB_FOLDER' -Cases @('discrete_gyroid_N16_fixed','discrete_gyroid_N16_relaxed_free')
```

脚本检查 INP 哈希，执行 datacheck、求解和 ODB 提取；已有 ODB 的目录
拒绝覆盖。不同安装位置通过 `-AbaqusCommand` 指定命令。原始大型 INP、
ODB、完整参考和位移文件留在本地工作目录，SHA256 见清单；Git 只保存
N=4 可复现示例和 N=16 轻量证据。一单元一类型受 U1–U9999 限制，当前
生成器不能用于 N=24/32；后续需通用 UEL 或可扩展矩阵输入，不能截断网格。

格式采用每列上三角、每行至多四个逗号分隔数值；最初的连续 F20 字段
输入被 2026 读取器拒绝，修复后通过实际检查。功能依据
[线性用户单元](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEELMRefMap/simaelm-c-userelem.htm)，
实际关键字行为以本次 2026 作业为准。

# Abaqus 2026 均匀实体基准：三种工况均已实际验证

> 历史阶段记录：以下结果、测试数量及“下一步”保留当时口径。当前状态和后续顺序以 [研究主线与四阶段计划](../../docs/RESEARCH_STATUS.md) 为准；本次未改动该阶段原始数值证据。

JAX 对照基准为提交 `d75a7daf07dec927562e9c222833a1d6a61f8152`。
本目录是第一级均匀实体验证，**没有 Gyroid 材料场、USDFLD、UEL、
接触、摩擦、塑性或大变形**。所有 INP 均为 Abaqus/Standard 静力分析，
NLGEOM=NO。2026-10-01 已完成三种工况的实际 Abaqus 2026 求解与结果验收：
fixed 由用户运行，relaxed_prescribed 与 relaxed_free 由 agent 使用命令完成
datacheck、求解和提取。三者均成功完成、无错误或警告，验收状态均为 ok。
详见 [`verified/summary.csv`](verified/summary.csv)、`verified/run_manifest.json`
及各工况的 `.acceptance.json` 和 `.sta`。

## 文件与理论值

单位立方体，E=10、ν=0.3、εz=-0.01。长度、应力和力采用一致的任意单位；
应力取压缩为负，Fz 为加载控制节点 QZ 的 RF3，V=A=1。
N=4，每个模型 64 个 C3D8、125 个物理节点和三个宏观控制节点。

| 输入文件 | 横向条件 | εx=εy | σzz / Fz | 总应变能 |
| --- | --- | ---: | ---: | ---: |
| `uniform_xy_fixed.inp` | 宏观横向应变固定为零 | 0 | -0.1346153846 | 0.0006730769231 |
| `uniform_xy_relaxed_prescribed.inp` | 给定解析横向应变，隔离验证输入和提取流程 | +0.003 | -0.1 | 0.0005 |
| `uniform_xy_relaxed_free.inp` | QX、QY 的宏观自由度不施加力或位移，自动松弛 | 理论 +0.003 | -0.1 | 0.0005 |

fixed 的 σxx=σyy=-0.05769230769；两种 relaxed 的理论平均横向应力为零。
自由松弛模型通过宏观控制自由度的自然平衡得到零平均横向应力；不是将所有
侧面节点分别设置为无牵引。首轮先运行 fixed 与 relaxed_prescribed，通过后
再运行 relaxed_free。

## 约束与离线检查

XY 周期约束施加在总位移上：u_slave-u_master=H·(X_slave-X_master)。
角点与棱边通过周期等价类直接连接到唯一代表节点。被消去的 DOF 不重复出现
在其他方程或边界条件中。上下表面的 z 自由度不重复施加周期方程：底面 uz=0，
顶面 uz=QZ.U3。底角 ux=uy=0 去除横向平移；顶底横向局部位移保持可波动。

QX.U1=εx·L、QY.U2=εy·L、QZ.U3=εz·L，L=1。QZ 是各顶面轴向方程的
共同控制节点，所以读取其 RF3 得到总加载反力；不要把普通周期从节点 RF
直接相加当作完整约束力。宏观控制节点不位于任何实体单元内。

准备工具检查了独立消元、BC 冲突、仿射跳量、HEX8 节点方向及体积。
`offline_tests.txt` 中的九个独立 NumPy/SciPy 有限元检查覆盖 N=1/2/4 和三种
横向条件，验证全部节点位移、轴向反力及能量。这些离线结果**不代表 Abaqus
已接受 INP，也不代表已经完成 Abaqus 软件对照**。

## 后续运行与提取

推荐将本目录复制到 Windows 的专用作业文件夹，避免直接在 Git 验证目录产生
ODB 等大型结果。保持作业名与 INP 文件名一致。下一阶段先查看并确认模型，
再进行 datacheck 和求解；准备工具本身不会执行任何作业。后续采用命令运行。

若以后获准运行，在作业文件夹手动运行示例：

```text
abaqus job=uniform_xy_fixed input=uniform_xy_fixed.inp interactive
```

求解后用 Abaqus Python 运行仓库中的只读提取工具（将路径替换为本机路径）：

```text
abaqus python scripts/extract_uniform_baseline.py PATH/TO/uniform_xy_fixed.odb --expected PATH/TO/expected.json
```

工具输出 `.acceptance.json`，检查最终加载时间、积分点数和体积、JxW/IVOL
加权平均应力、QZ 反力、ALLSE、宏观位移以及全部物理节点仿射位移。
理论零应力使用轴向应力尺度的绝对阈值；非零量相对验收目标为 1e-6。
对后续作业仍需检查 `.sta/.msg/.dat` 中的求解完成状态和警告。提取工具已
完成三种实际 ODB 的验证。曾发现 Abaqus 返回的 NumPy 标量导致 JSON
序列化失败，已统一转换为 Python float/bool，并以实际 ODB 重新验收通过。

## 边界与材料场验证的范围

此均匀仿射基准不能证明 Abaqus C3D8 与 JAX HEX8 在非均匀材料下具有相同
刚度。后续积分点材料场阶段必须先核查选择性体积积分差异。二值 Gyroid
贴体实体是另一个独立层次，不在本目录中。若以后改变几何尺度，应同时
修改宏观跳量、反力/应力换算以及验收工具；当前工具仅用于单位立方体。

官方参考（本轮公开可访问的是 2025 文档，实际执行时核对本机 2026 帮助）：

- [方程约束及约束力](https://docs.software.vt.edu/abaqusv2025/English/SIMACAECSTRefMap/simacst-c-equation.htm)
- [一阶实体选择性积分](https://docs.software.vt.edu/abaqusv2025/English/SIMACAETHERefMap/simathe-c-solidisoquadhex.htm)
- [积分点 IVOL](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEOUTRefMap/simaout-c-std-elementintegrationpointvariables.htm)
- [ALLSE](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEOUTRefMap/simaout-c-std-wholeandpartialmodelvariables.htm)

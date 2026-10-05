# 本轮20%压缩证据

JAX显式减步在同一N64薄壁上完成20%及保载；平均反力幅值2.241863N，对慢Explicit差22.41%，整段曲线RMS/参考特征峰值差13.47%，输入功差15.51%。实际主体耗时18.57分钟，1次无效块被拒绝，终态动能/弹性储能0.00005%，能量与输入功差0.008%。

以上为原HEX8基准。最新HEX27同物理候选的慢路径反力/曲线/功差9.45%/5.87%/6.87%，进入约10%响应目标；加载时间加倍复核通过。旧第4步中心平衡/伴随及减薄Newton失败保留；最新完整路径结果见gradient20_path_20261005。严格参考质量/真实接触替代认证尚未成立。完整解释见[本轮报告](../../docs/FORWARD20_PROGRESS_REPORT.md)。

| 位置 | 内容 |
| --- | --- |
| static/a0p150000 | 静力线性求解失败，有限/正体积记录保留 |
| static/a0p200000 | 仅准备INP/输入；没有JAX静力20%作业 |
| abaqus/ | 三条至20%无接触弹性壳、提取/留存哈希；大ODB在E盘 |
| explicit/wave_check、N64_probe | 集中质量/周期积分检验与成本 |
| explicit/N64_T0p004 | 固定步首次有效到16.2139%后停止；原字节保留 |
| explicit/N64_T0p004_adaptive | 有效响应、减步/拒绝记录、终态场 |
| comparison_summary.json、field_summary.json、status.json | 数字定义、质量边界、本轮计数和状态 |
| background_bridge/ | 相同参考Gauss占据/指定20%位移的原生C3D8材料场桥接，不是独立曲线 |
| quadratic_candidate/ | 二次表示准备、必要资源停止、两条0至20%路径、比较/模式/速率证据 |
| quadratic_wave/、quadratic_wave_runtime_geometry/ | Q2质量/积分基础检验，不作TPMS精度证据 |
| gradient20/ | 第4步成本/切线/伴随实验及实际停止，未认证路径梯度 |

大数组、源码快照、原始日志留在本机，Git不包含完整实验档案。时间积分与动态反力复用原NH残差；无接触/塑性/质量缩放/材料调参。


## 最新原第4步执行

正负完整路径及JVP已执行，10%/15%导数核对通过，20%保载与损失导数未通过。20%保载JVP为+76.093668N/mm，独立差分为-10.797551N/mm，符号不一致；不能用于20%逆设计。新增全部在[gradient20_path_20261005](gradient20_path_20261005/README.md)，旧输入、曲线、JSON、场和停止日志冻结。解释见[执行记录](../../docs/GRADIENT20_PROGRESS_REPORT.md)，实际下一项只看[主规划](../../docs/RESEARCH_PLAN.md)。

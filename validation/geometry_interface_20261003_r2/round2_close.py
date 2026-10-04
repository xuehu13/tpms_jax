from pathlib import Path
import sys,json,hashlib,shutil,re,subprocess
R=Path('/home/xuehu/projects/tpms_jax');W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an');O=R/'validation/geometry_interface_20261003_r2'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
step1=json.loads((O/'step1/summary.json').read_text());step2=json.loads((O/'step2/summary.json').read_text());small=json.loads((O/'step3/N8/summary.json').read_text());cost=json.loads((O/'step3/N32/summary.json').read_text());gate=json.loads((O/'step3/N64_resource_gate.json').read_text())
assert gate['decision']=='skip_N64' and small['directional_passed']
assert '130 passed' in (O/'regression.console.txt').read_text()
old=R/'validation/abaqus_binary';old32=json.loads((old/'binary_gyroid_G32_R0_C3D10_fixed.acceptance.json').read_text());old48=json.loads((old/'binary_gyroid_G48_R0_C3D10_fixed.acceptance.json').read_text())
oldN48=json.loads((R/'validation/near_term_20261003/step1/N48_fixed.json').read_text());oldN64=json.loads((R/'validation/projection_effects_20261002/N64_beta40_emin4_fixed_c.json').read_text())
base={'label':'c0','c':.541062,'K_N48':abs(oldN48['Fz_top'])/.01,'K_N64':abs(oldN64['Fz_top'])/.01,'K_G32':abs(old32['measured']['macro_RF'][2])/.01,'K_G48':abs(old48['measured']['macro_RF'][2])/.01,'K_M64_N64':step1['cases'][0]['K'],'vf_analytic_N64':oldN64['vf_int']}
base.update(background_relative_change=abs(base['K_N48']/base['K_N64']-1),reference_relative_change=abs(base['K_G32']/base['K_G48']-1),analytic_binary_relative=abs(base['K_N64']/base['K_G48']-1),mapping_relative=step1['cases'][0]['mapping_relative'],mapped_binary_relative=step1['cases'][0]['binary_relative'],analytic_anchor_passed=True,array_anchor_passed=True)
anchors=[step2['anchors'][0],base,step2['anchors'][1]]
variations=[]
for a,b in [(anchors[0],base),(base,anchors[2])]:
 indicator=max(abs(x['K_N48']-x['K_N64']) for x in [a,b])
 indicator=max(indicator,max(abs(x['K_G32']-x['K_G48']) for x in [a,b]))
 dc=b['c']-a['c'];da=b['K_N64']-a['K_N64'];dr=b['K_G48']-a['K_G48'];dm=b['K_M64_N64']-a['K_M64_N64']
 variations.append({'from':a['label'],'to':b['label'],'delta_c':dc,'delta_K_analytic':da,'delta_K_binary':dr,'delta_K_mapped':dm,'slope_analytic':da/dc,'slope_binary':dr/dc,'slope_mapped':dm/dc,'max_absolute_mesh_change_indicator':indicator,'screening_threshold':3*indicator,'same_sign':da*dr>0 and dm*dr>0,'resolved_direction':min(abs(da),abs(dr),abs(dm))>3*indicator,'claim':'Finite physical trend only, not exact continuum shape derivative or confidence interval'})
(O/'step4').mkdir(exist_ok=False)
(O/'step4/physical_changes.json').write_text(json.dumps(variations,indent=2)+'\n')
compilation={}
for N in [8,32]:
 lines=(O/f'gradient_N{N}.console.txt').read_text().splitlines();times=[float(m.group(1)) for line in lines if (m:=re.search(r'Finished XLA compilation .* in ([0-9.e+-]+) sec',line))]
 compilation[str(N)]={'XLA_compilation_events':len(times),'logged_XLA_compilation_seconds_sum':sum(times),'notes':'Compilation-event sum, not independent wall-clock timing; no second high-resolution forward added for speed testing'}
(O/'step3/compilation.json').write_text(json.dumps(compilation,indent=2)+'\n')
ledger=json.loads((O/'execution.json').read_text());large=sum(x['kind']=='large_forward' for x in ledger)+sum(x.get('forward_count',0) for x in ledger if x['kind']=='cost_forward_adjoint');sf=sum(x.get('forward_count',0) for x in ledger if x['kind']=='small_forward_adjoint');adj=sum(x.get('adjoint_count',0) for x in ledger);aba=sum(x['kind']=='abaqus' for x in ledger);seconds=sum(x['seconds'] for x in ledger if x['kind'] in ['large_forward','small_forward_adjoint','cost_forward_adjoint'])
assert (large,sf,adj,aba)==(9,9,2,4) and seconds<5400
maxrel=max(x['relative_error'] or 0 for x in json.loads((O/'step3/N8/differences.json').read_text()))
summary={'status':'round_closed_with_explicit_limits','stages_completed':[1,2,3,4],'selected_array_M':64,'M32_array_passed':False,'anchors':anchors,'physical_variations':variations,'small_derivative_max_relative':maxrel,'N32_cost':cost,'N64_gradient_executed':False,'N64_resource_gate':gate,'counts':{'large_forward':large,'small_forward':sf,'primary_K_adjoint':adj,'abaqus_analysis':aba,'abaqus_datacheck':4,'jax_case_wall_seconds':seconds,'jax_budget_seconds':5400,'regression_tests':130},'training_executed':False,'nonlinear_executed':False,'next_minimum_problem':'Target-scale stiffness-gradient access, then small geometry-prior + performance-guidance comparison; current round stops before implementing either','next_problem_prerequisite':'N64 gradient cost is unresolved; stationary-energy route may reduce cost but is only checked here at N8/N32, requires target-scale verification before use'}
(O/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
report=f'''# 第二轮四步报告：可信几何接口与梯度使用边界

日期：2026-10-03。四步已按预定判据收口，包含不通过和资源停止结论。优先解决可行性，创新性暂后置。长期目标仍是用 JAX-FEM 的可信 TPMS 压缩响应和有效梯度服务构型／参数学习、生成与性能驱动逆设计；本轮没有训练、已知答案反推或大变形计算。

## 1. 计算定义与本轮回答的问题

同一均质基材与孔隙的 sheet Gyroid，G=sinX cosY+sinY cosZ+sinZ cosX，真实实体 |G|≤c。L=1、Es=10、ν=.3、uz=−.01；XY 周期、平整轴向加载、宏观横向固定。HEX8 八点积分、β40、Emin/Es=1e-4、p=1；K=|Fz|/.01=2U/uz²（单位胞）。φ 和软孔隙只是几何／数值近似。

- M：几何输入数组每轴分辨率，xyz 单元中心采样。
- N：背景 HEX8 每轴单元数，材料在本 Problem 的实际 Gauss 点计算。
- G：独立二值实体几何近似分辨率，贴体 C3D10；孔隙没有材料。

新链为 g_i=G(x_i) → 周期三线性插值 g_q → 一次 sigmoid 差投影 φ_q → E_q → 平衡 → K。g 是有符号隐式值，不是 SDF，不限 [0,1]；不做幅值重归一化、阈值拟合或体积标定。

## 2. 第一步：输入保真（已完成）

| N64 输入 | K | 相对解析新增差 | 相对二值 G48 差 | 判据结论 |
|---|---:|---:|---:|---|
'''
for x in step1['cases']:report+=f"| M{x['M']} 隐式场 | {x['K']:.9f} | {100*x['mapping_relative']:.3f}% | {100*x['binary_relative']:.3f}% | {'通过' if x['passed'] else '未达标'} |\n"
report+='''
选择 M64。M32 未通过实体 2% 目标，停止推广该分辨率，没有追加数组候选。M64 过渡区比例 9.642%，解析 9.628%；上轮占据场插值为 16.579%。加权占据差从上轮 0.01825 降为 0.0007053。不是通过调 c 获得匹配。两个新解的残差、反力、功、周期与加载面检查通过。

## 3. 第二步：三个几何锚点（已完成）

| c | 解析 N64 K | M64→N64 K | 二值 G48 K | 解析／实体差 | 数组／实体差 | 结论 |
|---|---:|---:|---:|---:|---:|---|
'''
for a in anchors:report+=f"| {a['c']:.6f} | {a['K_N64']:.6f} | {a['K_M64_N64']:.6f} | {a['K_G48']:.6f} | {100*a['analytic_binary_relative']:.3f}% | {100*a['mapped_binary_relative']:.3f}% | 通过 |\n"
report+='\n两个新锚点 N48→N64 变化分别 0.332%/0.371%，G32→G48 变化 0.576%/0.614%；映射新增差 0.276%/0.295%，均通过预设目标。支持三个锚点的全局线弹性刚度，不认证整个区间、多构型或任意生成场。参考体积分数另用重复 Sobol 估计，保留在 step2/。\n\n'
report+='| 新 Abaqus 工况 | C3D10 数 | 畸变单元数 | 最小正 detJ | 总用时（含校核／提取） |\n|---|---:|---:|---:|---:|\n'
for a in step2['abaqus']:report+=f"| {a['label']} G{a['G']} | {a['elements']} | {a['diagnostics']['distorted_elements']} | {a['minimum_detJ']:.3e} | {a['seconds']:.1f}s |\n"
report+='''
四项分析与四项 datacheck 均成功，原 INP/mesh 哈希与 ODB 完成记录核对通过，周期、反力、能量、体积等检查通过。两项 G48 均有外存求解警告；畸变单元和极小正 Jacobian 未消失，因此仍是暂用的全局参考，不是严格真实误差界或局部应力真值。

## 4. 第三步：所选链导数与成本（已完成，N64 按门槛未执行）

N8/M64 检查真实选定数组链（g 与 c 共 262145 个输入），一个 c 方向、一个周期空间方向，两步长 .001/.0005，9 次前向与一次 K 伴随。所有解通过一致性；最大相对导数差 '''+f'{maxrel:.3e}'+'''，小于 1e-3。c 方向误差在步长减半后约缩为四分之一；空间方向误差约 1e-11。输入、方向、梯度与扰动结果保留。

| FEM 分辨率／输入 | 首次前向（含编译／追踪） | 首次伴随（含编译／追踪） | 主机峰值 | 采样设备峰值 | 证据 |
|---|---:|---:|---:|---:|---|
'''
for x in [small,cost]:report+=f"| N{x['N']}/M64 | {x['forward_seconds_including_first_trace_compile']:.2f}s | {x['adjoint_seconds_including_first_trace_compile']:.2f}s | {x['memory']['adjoint_peak_MiB']/1024:.2f}GiB | {x['memory']['sampled_device_peak_MiB']}MiB | 导数／一致性通过 |\n"
report+=f'''
N8 复用编译的扰动前向约 0.32～0.39s。高分辨率用 PETSc CG/GAMG 前向和伴随，没有沿用小网格直接解法。XLA 编译事件与求解阶段日志保留；表中首次用时不是纯线性求解或公平软件速度基准。设备峰值为每秒 nvidia-smi 采样，包含其他应用占用，可能漏掉瞬时峰值。

N32 的伴随与线弹性能量驻值梯度相对 L2 差 {cost['envelope_relative_L2_error']:.3e}。K 的平衡位移导数项理论上消失，这支持该刚度梯度；不能直接代表任意损失、非线性历史或神经网络梯度的验证。

N64 资源预评估：已有前向峰值 {gate['known_max_N64_forward_peak_MiB']:.1f}MiB；按 N32 伴随增量×8 估计合计 {gate['estimated_N64_forward_plus_adjoint_peak_MiB']:.1f}MiB，当前可用 {gate['available_host_MiB']:.1f}MiB，仅余 {gate['estimated_host_headroom_MiB']:.1f}MiB，低于预定 1024MiB。**未执行 N64 成本前向／伴随**，没有调整 WSL 内存或转向 N128。估计不是 OOM 证明；显存 N64 余量也未认证。

N32 此次 K={cost['K']:.6f}，相对 c0 二值 G48 差 {100*abs(cost['K']/base['K_G48']-1):.3f}%，超过当前 2% 目标。N32 是成本过渡，不能以其较低费用替代 N64 精度证据。N8 导数正确也不能用来认证实体刚度。

## 5. 第四步：物理变化和后续取舍（已完成，无额外求解）

| c 变化 | 解析 ΔK | 二值 ΔK | 映射 ΔK | 三倍最大绝对网格变化指示量 | 趋势筛查 |
|---|---:|---:|---:|---:|---|
'''
for a in variations:report+=f"| {a['from']}→{a['to']} | {a['delta_K_analytic']:.6f} | {a['delta_K_binary']:.6f} | {a['delta_K_mapped']:.6f} | {a['screening_threshold']:.6f} | {'同向且超过筛查量' if a['resolved_direction'] and a['same_sign'] else '不足'} |\n"
report+='''
物理变化明显超过网格变化指示量，独立实体支持增厚等值带后刚度提高的方向与量级。这里只支持有限变化趋势；N32 局部 AD、三个锚点的割线及真实连续形状导数不是同一个量。

能力取舍：已有可信的 N64 小变形前向、M64 隐式数组接口和正确的 N8/N32 离散刚度梯度。目标分辨率的通用伴随成本仍未验证。没有训练集、学习模型或构型逆设计结果；三个锚点不是训练数据规模的证明。

**选择的下一最小问题（仅定义，未执行）：** 使可信 N64 刚度评价与可承受梯度接入一个小型“几何先验＋性能引导”比较，观察目标性能匹配是否优于同先验随机抽样，并用独立二值实体复核输出设计。输入应为周期隐式几何／低维生成变量，输出为明确二值 TPMS 和刚度；不要求恢复唯一原始 c，也不把三个常数 c 锚点拟合成模型当作构型学习。

第一道前置证据是 N64 梯度成本。对当前线弹性 K，能量驻值形式可能避开额外伴随矩阵；本轮只在 N8/N32 核对了恒等式，N64 使用前仍须验证资源、离散导数与结果口径。若它可用，再制定最多四步的最小学习试验；若不可用，先报告计算接口限制，不自动改用未认证 N32、扩大标签数据或放弃 JAX 路线。学习阶段的几何变化还需独立前向抽查，本轮锚点不能代替它。

本轮没有执行上述后续问题，不锁定网络、训练样本数或新 TPMS 族，也没有新增 β/η/c 拟合扫描。后续问题的正式执行预算待新一轮计划锁定；当前自动停止于第二轮。

## 6. 文件、复现与预算

- 正式程序：`/home/xuehu/projects/tpms_jax`。
- 原始证据：`validation/geometry_interface_20261003_r2/`；step1 输入与保真，step2 锚点与实体提取，step3 导数／成本／资源停止，step4 物理变化。
- Abaqus 原始 INP/mesh/ODB/日志：`E:/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus/geometry_interface_20261003_r2/{c048,c060}/{G32,G48}/`。本目录的 work/ 保留实际分析文件。formal step2/ 留提取、完成、诊断及原始文件哈希。
- `execution.json` 保存实际命令、用时和计数，`source_manifest.json` 保存代码、库与证据哈希；`source_before/` 与 `source_after/` 保存本轮变动代码。
- 本轮只提取投影公用函数、增加有符号数组薄适配、可选伴随求解配置，并让原二值网格/体积/配对工具显式传 c；核心 FEM/周期/材料及安装库不变。没有新 FEM、训练框架或平行规划。
'''
report+=f'\n实际新增：高／中前向 {large}/10，小前向 {sf}/9，K 伴随 {adj}/3，Abaqus 分析 {aba}/4（另 4 次 datacheck）。JAX 实验墙钟合计 {seconds:.1f}s（{seconds/60:.2f}min）/90min。130/130 回归通过，84.49s；起始选择检查 24 项，29.23s，另记，不冒充科研算例。全部案例在单项时间预算内。\n'
report+='''
两项执行修正保留：首次 M64 汇总读错 Fz 字段，修正为 Fz_top 后复用成功结果，没有重算；首次 Windows 包装器选择了不兼容 PowerShell，提交前退出，修正宿主后提交，未增加 Abaqus 求解。第一轮、旧 CSV 和历史传输副本没有覆盖。未 Git 提交、推送或合并。
'''
(R/'docs/ROUND2_REPORT.md').write_text(report,encoding='utf-8');(W/'ROUND2_REPORT.md').write_text(report,encoding='utf-8')
(O/'README.md').write_text('# 第二轮证据入口\n\n四步按判据收口；结论见 [第二轮报告](../../docs/ROUND2_REPORT.md)。原始数据不再追加或覆盖。\n\n执行计划 executed_plan.md；预算 budget.json；实际执行 execution.json；结果 summary.json；来源 source_manifest.json。各步骤目录包含输入、日志、结果与停止证据。\n')
if (O/'progress.md').exists():(O/'progress.md').rename(O/'progress_step1.md')
status=f'''# 研究状态：第二轮四步已收口

更新：2026-10-03。长期目标见 [研究背景](RESEARCH_BACKGROUND.md)，执行约定见 [主规划](RESEARCH_PLAN.md)，文件见 [文件地图](FILE_MAP.md)。先完成目标、确认可行性，创新性暂后置。本轮结果见 [第二轮报告](ROUND2_REPORT.md)；没有正在运行或自动排队的科研作业。

## 当前能力与限制

| 环节 | 实测结果 | 边界 |
|---|---|---|
| 输入保真 | 先插值 G 再投影，M64→N64 新增刚度差 0.286%、实体差 1.524% | M32 实体差 2.186%，未达标；不追加候选 |
| 三几何锚点 | c=.48/.541062/.60；解析 N64／实体差 1.333%/1.234%/1.138%，M64／实体差 1.613%/1.524%/1.436% | 全局固定横向线弹性；不认证整个区间、多构型或生成场 |
| 链导数 | N8/M64 两方向两步长，最大相对差 {maxrel:.3e} | 小网格代数正确，不是实体精度 |
| 成本 | N32/M64 首次前向 11.53s、伴随 7.00s，主机峰值 2.67GiB | N32 相对实体刚度差 2.674%，不替代 N64 精度；N64 伴随按资源门槛未执行 |
| 物理变化 | 两段 ΔK 同向且超过三倍绝对网格变化筛查量 | 有限变化趋势，不是精确连续形状导数 |

两个新二值几何各 G32/G48，共四项 Abaqus 分析通过周期、反力、能量与完成检查；仍有畸变单元、小正 Jacobian 和 G48 外存警告，仅暂用全局参考。宏观 free 为横向平均应力零的周期松弛；本轮没有新 free 工况。

本轮实际 9 个高／中前向、9 小前向、2 K 伴随、4 Abaqus 分析＋4 datacheck；JAX 实验 {seconds/60:.2f}min，130/130 回归通过。实验目录 validation/geometry_interface_20261003_r2；基线 HEAD 5e1d03d，改动未提交或推送，原始代码／输入／日志／哈希保留。

## 已有基础（历史，不追加待办）

第一轮见 [第一轮报告](NEAR_TERM_REPORT.md)：解析 c0 N64 fixed/free 相对实体差约 1.234%/1.250%；旧占据插值 M64 未达标、M32 未执行；N8/M8 占据导数只作诊断。旧 β20、η1e-3 结果和 M4 CSV 不变，第一轮目录冻结。

继承 B1 均匀解析基准、B3 独立 NumPy 用户单元同离散检查、B4 二值全局响应与网格敏感性。原生 C3D8＋USDFLD、旧壳模型替代、多构型、训练和 20% 大压缩均未认证，也不作为本轮停止项的自动续跑任务。

## 后续选择（尚未执行）

研究主线不变：把可信力学接口接入最小几何先验与性能引导比较；不再扫描 β/η 拟合结果。唯一前置缺口是 N64 梯度成本：当前资源预估仅余 208MiB，小于 1GiB 门槛；线弹性 K 的能量驻值路线可能降低成本，尚需目标尺度验证。之后才能锁定一个最多四步的学习试验及预算。本轮没有生成下一轮完整执行清单、训练集、网络或逆设计结果。
'''
(W/'PROJECT_OVERVIEW.md').write_text(status,encoding='utf-8');(R/'docs/RESEARCH_STATUS.md').write_text(status.replace('(PROJECT_OVERVIEW.md)','(RESEARCH_STATUS.md)'),encoding='utf-8')
p=W/'RESEARCH_PLAN.md';t=p.read_text().replace('状态：第二轮执行中，从第一步按判据推进。','状态：第二轮四步已按判据收口，包含未达标与资源停止。').replace('本文件是唯一近期执行入口。','本文件保留本轮唯一执行定义，不自动重跑；结论见 [第二轮报告](ROUND2_REPORT.md)。')
t=t.replace('四步结束即形成结论并重新选择下一轮。','四步结论已经形成；下一最小问题定义见第二轮报告，尚未制定／执行下一轮完整清单。')
p.write_text(t,encoding='utf-8');(R/'docs/RESEARCH_PLAN.md').write_text(t.replace('(PROJECT_OVERVIEW.md)','(RESEARCH_STATUS.md)'),encoding='utf-8')
p=W/'FILE_MAP.md';t=p.read_text().replace('当前第二轮按主规划执行中','当前第二轮按判据已收口').replace('四参数可导适配及复用解的检查入口；未接网络','可导适配、可选前向／伴随求解配置及复用解检查；未接网络').replace('周期连续占据数组映射和继承原求解器的薄适配；M64 保真未达标、N8/M8 导数通过','周期占据／有符号隐式数组薄适配；M64 隐式链通过三锚点，M32 隐式输入未达标').replace('数组接口当前是诊断能力，不是通过精度认证的学习接口；未增加网络或 learning 框架。','M64 隐式接口已有三个几何锚点的全局精度证据；任意生成场、网络与目标尺度伴随尚未认证。未增加网络或 learning 框架。')
t+='\n## 第二轮证据\n\n[报告](ROUND2_REPORT.md)；正式原始目录 validation/geometry_interface_20261003_r2/，step1 输入、step2 锚点、step3 导数／资源停止、step4 物理变化。Abaqus 原始包在 E:/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus/geometry_interface_20261003_r2/，按 c048/c060 和 G32/G48 分开。复现命令以 execution.json 为准；source_before/source_after 是冻结的来源，不是另一个维护分支。\n'
p.write_text(t,encoding='utf-8');(R/'docs/FILE_MAP.md').write_text(t.replace('(PROJECT_OVERVIEW.md)','(RESEARCH_STATUS.md)'),encoding='utf-8')
for p in [W/'AGENTS.md',R/'AGENTS.md']:
 t=p.read_text().replace('；尚未执行。按主规划的证据条件和预算推进','；四步已按判据收口，第二轮报告见 '+('[第二轮报告](ROUND2_REPORT.md)' if p.parent==W else 'docs/ROUND2_REPORT.md')+'。M64 隐式输入通过三个锚点、M32 未达标，N64 伴随按资源门槛未执行，均不作为自动续跑待办。按主规划的证据条件和预算推进');p.write_text(t,encoding='utf-8')
p=W/'START_HERE.md';t=p.read_text().replace('第二轮规划已制定、尚未执行','第二轮四步已按判据收口，见 [第二轮报告](ROUND2_REPORT.md)；下一轮尚未制定／执行');p.write_text(t,encoding='utf-8')
p=R/'README.md';t=p.read_text().replace('第二轮规划已制定、尚未执行','第二轮四步已按判据收口，见 [第二轮报告](docs/ROUND2_REPORT.md)；下一轮尚未制定／执行').replace('第二轮仅规划','第二轮已收口');p.write_text(t)
for name in ['round2_step1.py','round2_step2.py','round2_gradient.py','round2_collect.py','round2_resource_gate.py','round2_close.py','run_round2_abaqus.ps1']:
 shutil.copy2(W/'tmp'/name,O/name)
old_manifest=json.loads((O/'preservation_before.json').read_text())
for n,h in old_manifest['experiment_sha256'].items():assert sha(R/'validation/near_term_20261003'/n)==h,n
lib=R/'.pixi/envs/default/lib/python3.13/site-packages/jax_fem'
for n,h in old_manifest['installed_jax_fem_sha256'].items():assert sha(lib/n)==h,n
for n in ['fem.py','density_fem.py','pbc.py','pixi.toml','pixi.lock']:assert sha(R/n)==old_manifest['source_sha256'][n],n
assert sha(R/'results/m4_numerical_study.csv')=='dc6c3c18881d75737cb99678e60376bc30f54b93f54504ea84a9ff8141bd0fb4'
names=['voxel_field.py','geometry.py','design_fem.py','fem.py','density_fem.py','pbc.py','scripts/capture_binary_projection_reference.py','scripts/m4_numerical_study.py','binary_gyroid.py','scripts/prepare_abaqus_binary.py','scripts/prepare_binary_lateral_pair.py','scripts/estimate_binary_volume.py','scripts/run_abaqus_binary.ps1','scripts/extract_abaqus_binary.py','scripts/extract_uniform_baseline.py','tests/test_voxel_field.py','tests/test_abaqus_binary.py','pixi.toml','pixi.lock']
for n in names:
 p=O/'source_after'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(R/n,p)
(O/'source_changes.diff').write_text(subprocess.check_output(['git','diff','--',*names],cwd=R,text=True))
preservation={'round1_evidence_unchanged':True,'installed_library_unchanged':True,'FEM_material_pbc_environment_unchanged':True,'M4_CSV_unchanged':True,'old_transport_clone_untouched':True,'new_git_commit_push_merge':False}
(O/'preservation_after.json').write_text(json.dumps(preservation,indent=2)+'\n')
manifest={'base_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),'source_sha256':{n:sha(R/n) for n in names},'installed_jax_fem_sha256':old_manifest['installed_jax_fem_sha256'],'experiment_sha256':{str(f.relative_to(O)):sha(f) for f in O.rglob('*') if f.is_file() and f.name!='source_manifest.json'},'old_manifest_sha256':sha(R/'validation/near_term_20261003/source_manifest.json'),'working_tree_changes_uncommitted':True}
(O/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'counts':summary['counts'],'preservation':preservation,'variations':variations,'report':str(R/'docs/ROUND2_REPORT.md')},indent=2),flush=True)

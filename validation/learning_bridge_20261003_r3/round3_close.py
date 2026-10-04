from pathlib import Path
import json,hashlib,shutil,subprocess,re
R=Path('/home/xuehu/projects/tpms_jax');O=R/'validation/learning_bridge_20261003_r3';S=O/'step2'
W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
initial=json.loads((O/'initial_manifest.json').read_text())
step1=json.loads((O/'step1/summary.json').read_text());fwd=json.loads((S/'forward_summary.json').read_text())
assert step1['passed'] and fwd['background_passed']
uniform=[]
for p in (R/'validation/near_term_20261003').rglob('*.json'):
 try:m=json.loads(p.read_text())
 except (ValueError,UnicodeError):continue
 if isinstance(m,dict) and m.get('N')==48 and m.get('beta')==40 and m.get('emin_ratio')==1e-4 and m.get('lateral')=='fixed' and 'Fz_top' in m:
  uniform.append((p,abs(m['Fz_top'])/.01))
assert uniform and max(abs(v-uniform[0][1]) for _,v in uniform)<1e-10
uniform_K48=uniform[0][1];uniform_K64=fwd['uniform_baseline_K']
max_mesh=max(fwd['background_max_absolute_mesh_change'],abs(uniform_K48-uniform_K64))
threshold=3*max_mesh;span=fwd['physical_span_background_only'];assert span<threshold
accept=json.loads((S/'ax035_G32_acceptance.json').read_text());assert accept['status']=='ok'
prepared=json.loads((S/'ax035_G32_prepared.json').read_text())
K32=abs(accept['measured']['macro_RF'][2])/.01
ax64=next(c['K'] for c in fwd['cases'] if c['label']=='ax035' and c['N']==64)
partial_diff=abs(ax64-K32)/K32
log=(S/'ax035_G48_prepare.console.txt').read_text();assert 'Nonmanifold tetrahedral mesh' in log
failed=Path('/mnt/e/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus/learning_bridge_20261003_r3/ax035/G48')
assert not list(failed.glob('*.inp')) and not list(failed.rglob('*.odb'))
(failed/'FAILED_BEFORE_SUBMISSION.json').write_text(json.dumps({'stage':'geometry_preparation','reason':'Nonmanifold tetrahedral mesh in post-remesh audit','Abaqus_submitted':False,'retry_planned':False,'log':str(S/'ax035_G48_prepare.console.txt')},indent=2)+'\n')
abaledger=json.loads((O/'abaqus_execution.json').read_text());ledger=json.loads((O/'execution.json').read_text())
count=sum(r.get('forward_count',0) for r in ledger);aba_count=sum(r.get('Abaqus_analysis_count',0) for r in abaledger)
assert count==14 and aba_count==1
facts={'phase':'round3_closed_at_step2','step1':'passed','step2':'not_passed','step3':'not_started_condition_not_met','step4':'not_started_condition_not_met','research_forward_count':count,'research_general_adjoint_count':0,'research_small_forward_count':0,'Abaqus_analysis_count':aba_count,'Abaqus_datacheck_count':sum(r.get('datacheck_count',0) for r in abaledger),'JAX_research_seconds':sum(r.get('seconds',0) for r in ledger),'Abaqus_preparation_run_and_retry_seconds':sum(r.get('seconds',0) for r in abaledger),'regression_partition_counts':[130,3],'training_runs':0,'new_dependencies':0,'performance_span':span,'background_three_times_mesh_lower_bound':threshold,'uniform_N48_K':uniform_K48,'uniform_N48_evidence':str(uniform[0][0]),'reference':'ax035 G32 consistency passed; ax035 G48 post-remesh nonmanifold failed before submission; az035 references not executed','ax035_G32_K':K32,'ax035_N64_vs_G32_relative_diagnostic_only':partial_diff,'mesh_failure_is_not_FEM_failure':True,'stopped_items_not_automatic_todo':True}
(S/'summary.json').write_text(json.dumps(facts,indent=2)+'\n');(O/'summary.json').write_text(json.dumps(facts,indent=2)+'\n')
archive=R/'docs/RESEARCH_PLAN_ROUND3_COMPLETED.md';assert not archive.exists()
shutil.copy2(O/'source_before/docs/RESEARCH_PLAN.md',archive)
rows=[]
for a in ['ax035','az035']:
 pair=[c for c in fwd['cases'] if c['label']==a];change=abs(pair[0]['K']-pair[1]['K'])/pair[1]['K']
 rows.append(f"| {a} | {pair[0]['K']:.9f} | {pair[1]['K']:.9f} | {100*change:.3f}% | {pair[1]['Vf']:.9f} |")
report=f'''# 第三轮报告：N64 专用梯度通过，最小学习示例在第二步停止

日期：2026-10-03。研究工况已收口，无正在运行或排队的作业。第三、四步因前置条件不满足未启动；不是训练或逆设计已完成，也不作为自动遗留任务。

## 研究判断

目标尺度的小变形刚度与几何体积梯度可在当前硬件上获得，不需要再求一个通用伴随系统。这个关键可行性环节已增加实测证据。

本轮选择的相近用料、有限周期壁宽空间产生的刚度变化未越过预定数值分辨筛查门槛。新几何的 G48 参考又未通过拓扑审查，因此不能认证这一新参数域或据它宣称生成引导有效。本轮停止的是这个最小学习示例；长期目标仍是 TPMS 构型／参数学习及性能驱动逆设计。固定用料是本轮选用的对照约束，不是用户长期目标的必需条件；四参数也不是长期空间上限。

## 四步实际状态

| 步骤 | 结果 | 证据与边界 |
|---|---|---|
| 1 N64 专用 K/Vf 导数 | 通过 | N32 对照已存完整伴随；N64 两方向两步长差分；无新研究通用伴随 |
| 2 相近用料壁宽锚点 | 未达标，收口 | 4个背景工况通过一致性／网格检查；性能跨度不足；仅 ax035 G32 实体完成，G48 网格失败 |
| 3 参数先验／网络链 | 未启动 | 第2步前置条件不满足；没有训练、生成模型或神经网络到 FEM 的链证据 |
| 4 性能引导／实体复核 | 未启动 | 无在线设计或随机生成对照，无已知答案反推替代 |

## 1. N64 专用梯度

锁定 L=1、Es=10、ν=.3、uz=−.01、XY 周期、平整加载面、宏观横向固定、HEX8 八点积分、β40、η1e-4、p1。第一步输入为 M64 有符号 Gyroid 数组先插值再投影。

专用导数依据当前位移控制、固定约束与固定背景下的能量驻值：K=2U/(V uz²)，dK/dφq=2(Es−Emin)ψ1q JxWq/(V uz²)。分块计算几何 VJP，复用原平衡求解器。此规则只用于当前 K/Vf，不推广到任意位移、应力、历史或接触损失。

- N32 全向量刚度梯度与第二轮已存完整伴随相对 L2 差 **3.35e-14**；响应差约1.58e-16。未重新计算该伴随。
- N64 K={step1['N64_K']:.12f}，Vf={step1['N64_Vf']:.12f}，dK/dc={step1['N64_dK_dc']:.12f}。
- c方向与归一化空间场方向，h=.001/.0005，K/Vf 共8项导数比较全部通过；最大非零相对差 **{step1['max_nonzero_relative_FD_error']:.3e}**。方向沿用第二轮、扰动前锁定。
- N64 基准含构建、编译及检查约100.0s。10次研究前向合计{step1['total_seconds']/60:.2f}min；没有额外通用伴随矩阵。
- 研究样本最大主机 RSS {step1['peak_host_rss_MiB']/1024:.2f}GiB；采样最低主机余量{step1['sampled_min_host_available_MiB']/1024:.2f}GiB，设备峰值{step1['sampled_peak_device_MiB']/1024:.2f}GiB。通过1GiB／.5GiB余量门槛；采样不是瞬时严格峰值界。

这证明所选离散问题及输入处的导数和成本。小型耦合损失的自定义 VJP 还通过 N4 集成回归；没有把它称为网络训练证据。

## 2. 新几何、可分辨变化与参考限制

几何 c(X)=c0+ax cos(2πx)+ay cos(2πy)+az cos(2πz)，目标实体 |G|≤c(X)，基材均质。两锚点 a=(.035,0,0)／(0,0,.035)，几何积分求得相同 c0=0.5410380367934704。投影 Vf*=.3505280533，构造误差约3.81e-11；不是按力学响应标定。全局 c 范围 .506038～.576038。

| 锚点 | N48 K | N64 K | 网格变化 | N64 Vf |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

已验证均匀基准 N64 K={uniform_K64:.9f}。以上三种 N64 响应的跨度 **{span:.9f}**，约为基准的{span/uniform_K64*100:.3f}%。包括复用均匀 N48 基准的三倍最大背景绝对网格变化为 **{threshold:.9f}**，跨度未达门槛。实体网格指示量未全得到，该量只作为门槛的已有下界；补入实体指示量只会增加最大值。网格变化不是严格物理误差界，未通过不等于证明真实性能完全相同。

ax035 G32 已生成、datacheck、求解并提取成功：K={K32:.9f}，实体体积={accept['measured']['volume_solid']:.9f}，周期、加载面、反力平衡、积分能量及宏观功检查全部通过。N64 与该 G32 差约{partial_diff*100:.3f}%，**仅作粗参考诊断，不能代替规划的 G48／加密精度认证**。

G32 有{19972}/{135969}个畸变单元警告，最小正 detJ≈{prepared['geometry']['min_detJ']:.3e}，保留参考质量风险。ax035 G48 在 Gmsh 体网格生成后的 audit_linear/boundary_faces 检查出现 `Nonmanifold tetrahedral mesh`，未生成 INP、未提交分析；还没有分离输入界面构造与重网格环节的影响。按停止条件不继续 az035 的实体作业、不换网格器或扩大参数／数值扫描。

因此不能声称这两个新几何达到2%实体精度、整个参数域已认证、FEM失败、或学习路线不可行。下一轮科学问题需要基于这些结论重新选择；本报告不另立执行计划。

## 文件与实际成本

- 正式程序 `/home/xuehu/projects/tpms_jax`。本轮仅扩展 `design_fem.py`、`voxel_field.py`、`binary_gyroid.py`、`scripts/prepare_abaqus_binary.py`，新增3项必要检查于 `tests/test_stationary_width.py`。复用原求解器、材料与加载，无安装库修改或新依赖。
- 证据 `validation/learning_bridge_20261003_r3/`：step1 基准／梯度／差分／资源，step2 锚点／4个前向／G32提取／G48失败；execution 与 abaqus_execution 记录失败和重试成本；source_before/source_after、manifest 保留版本。
- Abaqus `E:/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus/learning_bridge_20261003_r3/ax035/G32/`：INP、expected、mesh及work中的ODB/dat/msg/sta/提取。`ax035/G48/FAILED_BEFORE_SUBMISSION.json`标明预处理失败；az035实体目录未创建。
- 科研工况：**14次高／中前向，0次科研小网格，0次新增科研通用伴随，1次Abaqus分析＋1次datacheck，0训练**。回归内微型求解及原伴随测试另计，未当作新物理认证算例。
- JAX科研累计{facts['JAX_research_seconds']/60:.2f}min；实体预处理、提交、提取及失败累计约{facts['Abaqus_preparation_run_and_retry_seconds']/60:.2f}min，首项原生提交用保守日志区间记时。均在规划预算内，未把上限做满。
- 原130项回归通过（82.25s），随后仅运行新增3项通过（6.95s），同一数值实现共133项覆盖；未重复整个测试集。
- 启动器旧PowerShell缺命令、提取包漏共用脚本已修复，失败日志保留；G32直接复用完成ODB，仅重新提取，没有重复分析。G48是另一个真实的网格审查失败。
- 第一、二轮原始输入／输出／清单保留；没有Git提交、推送或合并。未自动创建训练模块、模型、数据集或下一轮。
'''
(W/'ROUND3_REPORT.md').write_text(report);(R/'docs/ROUND3_REPORT.md').write_text(report)
plan='''# 近期主规划：第三轮已按停止条件收口

更新：2026-10-03。当前没有执行中的近期步骤。第三轮已形成通过／未达标／条件不执行的结论；下一轮尚未制定。长期问题见[背景](RESEARCH_BACKGROUND.md)，事实见[状态](PROJECT_OVERVIEW.md)，完整证据见[第三轮报告](ROUND3_REPORT.md)。

| 原步骤 | 实际状态 | 结论 |
|---|---|---|
| 1 N64专用 K/Vf 梯度 | 通过 | N32存量伴随核对、N64两方向两步长差分及资源通过 |
| 2 相近用料壁宽锚点 | 未达标，收口 | 4个背景解与网格检查通过；性能跨度未越过门槛；G32实体成功、G48拓扑失败，精度认证未完成 |
| 3 小型参数先验／网络链 | 条件不执行 | 第2步前置条件未满足，没有训练或网络链证据 |
| 4 一例性能引导／实体复核 | 条件不执行 | 无生成引导或随机对照结果 |

本轮14次高／中研究前向、1次Abaqus分析＋1次datacheck，零训练。没有重跑通用N64伴随；停止项不自动续跑。原第三轮合同与预算逐字保存在[归档](//wsl.localhost/Ubuntu-24.04/home/xuehu/projects/tpms_jax/docs/RESEARCH_PLAN_ROUND3_COMPLETED.md)，不得把其中将来时文字当作新待办。

取舍：N64目标尺度梯度这一关键接口已走通；本轮有限壁宽空间、相近用料约束下的性能分辨力和实体网格可靠性不足。它没有否定体素JAX-FEM或长期学习／构型逆设计路线。新问题、表示域或阶段须在下一轮另定不超过四步的唯一主规划，不从旧报告自动扩展网络、多TPMS族、大压缩、更多锚点或新网格器。
'''
(W/'RESEARCH_PLAN.md').write_text(plan);(R/'docs/RESEARCH_PLAN.md').write_text(plan.replace('(PROJECT_OVERVIEW.md)','(RESEARCH_STATUS.md)'))
status=(W/'PROJECT_OVERVIEW.md').read_text();status=status[:status.index('## 第三轮执行进度')]
status=status.replace('第三轮第一步进行中','第三轮已按停止条件收口').replace('；第三轮 N64 方向差分正在串行执行。','；第三轮见[报告](ROUND3_REPORT.md)，当前无运行或排队作业。').replace('本轮结果见 [第二轮报告]','第二轮结果见 [第二轮报告]').replace('本轮实际 9 个高','第二轮实际 9 个高').replace('本轮没有新 free','第二轮没有新 free')
status+='''## 第三轮最终状态

N64专用K/Vf梯度通过，两个方向两个步长最大相对差4.20e-7，N32与存量完整伴随向量差3.35e-14。10次前向，研究通用伴随0；N64基准值／梯度约100s，主机峰值12.15GiB、采样设备峰值4.95GiB。薄反向接口可用；尚无网络训练证据。

两个相近用料壁宽锚点4个背景解通过一致性和N48→N64≤1%检查。三设计性能跨度.006011，小于已有三倍背景网格指示量.021874。ax035 G32实体分析与提取通过；ax035 G48重网格后非流形审查失败、无INP/分析；az035实体未执行。新几何域的2%实体精度认证未完成。

第二步按条件停止；第三、四步未启动，不自动续跑。固定用料和有限四参数是本轮示例选择，不是长期问题的限制。本轮科研14次高／中前向、1次Abaqus分析＋1次datacheck、零训练；研究用通用伴随0，回归仍包含原伴随测试。原130回归与新增3项分别通过，无新依赖或Git操作。

正式证据 validation/learning_bridge_20261003_r3；报告[ROUND3_REPORT](ROUND3_REPORT.md)。本轮只扩展4个现有数值文件与一份3项测试，没有创建训练框架或复制FEM。两轮旧证据和环境保持原样。下一轮尚未制定；当前主规划仅记录收口和归档，历史停止项不是遗留待办。
'''
(W/'PROJECT_OVERVIEW.md').write_text(status);(R/'docs/RESEARCH_STATUS.md').write_text(status.replace('(NEAR_TERM_REPORT.md)','(../validation/near_term_20261003/README.md)'))
# Preserve the background before its first update in this execution turn.
bg=R/'docs/RESEARCH_BACKGROUND.md';backup=O/'source_before/docs/RESEARCH_BACKGROUND.md'
shutil.copy2(bg,backup);initial['source_before']['docs/RESEARCH_BACKGROUND.md']=sha(backup)
(O/'initial_manifest.json').write_text(json.dumps(initial,indent=2)+'\n')
background=(W/'RESEARCH_BACKGROUND.md').read_text()
background=background.replace('目标 N64 的实现与资源成本尚待验证。','第三轮 N64 专用 K/Vf 实现已通过两方向两步长差分和资源审查；没有计算通用 N64 伴随。')
background=background.replace('第三轮准备验证目标尺度的专用刚度梯度，并在条件通过后连接最小参数先验学习与性能引导。','第三轮专用 N64 梯度通过；有限相近用料壁宽示例性能跨度不足，G48 实体网格又未通过拓扑审查，因而未启动先验学习和引导。')
background=background.replace('小网格链通过，N32/M64 K 伴随与驻值导数一致；N64 通用伴随未过资源门槛。尚无训练或生成性能引导结果','小网格链、N32向量及N64专用K/Vf方向导数通过；N64通用伴随未执行。有限壁宽示例未过分辨／网格门槛，尚无训练或生成引导结果')
background+='''
### 第三轮取舍（2026-10-03）

目标尺度的专用刚度梯度及资源可行性已有证据。当前相近用料壁宽变化的性能跨度不足以支持预定学习示例，G48参考网格又在重网格后出现非流形，未完成新几何精度认证。第三、四步条件不执行；细节见[第三轮报告](ROUND3_REPORT.md)。这不等于背景FEM求解失败、整个学习路线不可行或真实几何没有性能差异。固定用料和四参数是本轮最小示例的选择；长期目标仍包含构型和参数学习。下一轮选择科学问题需另立近期规划，不自动延长当前训练、几何或数值扫描。
'''
(W/'RESEARCH_BACKGROUND.md').write_text(background);bg.write_text(background.replace('(PROJECT_OVERVIEW.md)','(RESEARCH_STATUS.md)'))
filemap=(W/'FILE_MAP.md').read_text()
filemap=filemap.replace('第三轮已授权执行，第一步进行中；后续条件步骤未执行。','第三轮按停止条件收口；没有运行作业或自动续跑待办。')
filemap=filemap.replace('现有周期壁宽四参数、响应与隐式伴随；可选 PETSc；未接训练模型','现有周期壁宽四参数、隐式伴随及分块专用K/Vf反向；N64专用导数通过，未接训练模型')
filemap=filemap.replace('目前常数 c 的二值域、贴体网格及审计；空间壁宽支持尚未实现','常数c与既有周期壁宽二值域；变量壁宽G32已完成，G48重网格后非流形审查失败，范围未认证')
filemap=filemap.replace('当前仅扩展 design_fem.py / voxel_field.py 的分块专用 K/Vf 反向；未新增 FEM 或训练平台。空间壁宽实体与先验仍待条件通过后添加。','本轮扩展4个现有数值文件及3项必要测试；没有FEM复制、训练平台、新依赖或空模块。先验和性能引导未启动。')
filemap=filemap.replace('复核；第一步执行中','复核的收口状态；不续跑归档步骤')
filemap=filemap.replace('docs/ROUND2_REPORT.md | 第二轮已完成报告，保留原文与哈希；Windows ROUND2_REPORT.md 为阅读副本','docs/ROUND2_REPORT.md / docs/ROUND3_REPORT.md | 第二轮冻结报告／第三轮通过与停止证据，Windows同名阅读副本')
filemap=filemap.replace('当前第三轮实验；step1 基准／差分、execution、源码前快照；不与旧结果混用','第三轮已冻结：step1通过，step2分辨筛查／网格失败，step3/4未启动；summary、预算、源码及失败日志')
filemap=filemap.replace('tests/ | 第二轮最新完整回归 130/130，84.49s；不是本次新测试结果','tests/ | 本轮原130回归通过82.25s，随后3新增检查通过6.95s；共133项，研究工况另计')
filemap=filemap.replace('分析／矩阵诊断共 27 项','分析／矩阵诊断共 28 项').replace('历史 23 加第二轮4','历史23加第二轮4、第三轮1')
filemap=filemap.replace('| geometry_interface_20261003_r2/{c048,c060}/{G32,G48} | 4 | 第二轮两个锚点；另4 datacheck |','| geometry_interface_20261003_r2/{c048,c060}/{G32,G48} | 4 | 第二轮两个锚点；另4 datacheck |\n| learning_bridge_20261003_r3/ax035/G32 | 1 | 新壁宽实体及1 datacheck；G48仅预处理失败，az035实体未创建 |')
filemap=filemap.replace('空间壁宽不能直接套用当前常数 c 生成器。','空间壁宽使用 --width-parameters，G48审查失败限制见本轮报告；不把该参数接口当作已认证域。')
filemap=filemap.replace('第三轮已有独立实验及第一步计算，尚无模型、训练、Abaqus 或 Git 提交。','第三轮已有14个研究前向、1项Abaqus分析及预处理失败；没有模型、训练或Git提交。')
filemap+='''
第三轮计划原文归档于 docs/RESEARCH_PLAN_ROUND3_COMPLETED.md，主规划只保留收口状态。新变量壁宽失败目录用 FAILED_BEFORE_SUBMISSION.json 标记；原始文件、旧输入／结果／日志不移动或删除。正式 source_before/source_after 是复现快照，实验辅助脚本不是另一套维护FEM。
'''
(W/'FILE_MAP.md').write_text(filemap);(R/'docs/FILE_MAP.md').write_text(filemap.replace('(PROJECT_OVERVIEW.md)','(RESEARCH_STATUS.md)'))
for p in [W/'AGENTS.md',R/'AGENTS.md']:
 lines=p.read_text().splitlines();lines=[line for line in lines if not line.startswith('- 第一、二轮已按判据收口')]
 paragraph='- 第一、二轮冻结；第三轮已按停止条件收口，报告为 '+('[第三轮报告](ROUND3_REPORT.md)' if p.parent==W else 'docs/ROUND3_REPORT.md')+'。N64专用K/Vf梯度通过；有限相近用料壁宽示例性能跨度不足，G48重网格拓扑失败，新域未认证。第3/4步条件不执行，没有训练或性能引导，不作为自动遗留。当前主规划只记录收口，下一轮尚未制定；不重跑历史M32/占据数组/通用N64伴随停止项。四参数与固定用料是临时示例，不是长期限制。'
 lines.insert(6,paragraph);p.write_text('\n'.join(lines)+'\n')
(W/'START_HERE.md').write_text('''# TPMS研究入口

长期目标：可信背景压缩与有效梯度，服务TPMS构型／参数学习、生成及性能驱动逆设计。

依次读[背景](RESEARCH_BACKGROUND.md)、[主规划](RESEARCH_PLAN.md)、[状态](PROJECT_OVERVIEW.md)、[文件地图](FILE_MAP.md)。第三轮已按停止条件收口：[报告](ROUND3_REPORT.md)。N64专用梯度通过；新壁宽示例分辨力不足、G48网格未合格，未训练或引导。当前无运行作业，下一轮尚未制定。旧停止项不自动续跑。

历史：[第一轮](NEAR_TERM_REPORT.md)、[第二轮](ROUND2_REPORT.md)；[方法依据](PAPER_ROUTE.md)。正式程序在WSL，Windows work/tpms_jax仍是历史传输副本。
''')
for p in [R/'README.md',R/'validation/README.md']:
 t=p.read_text();lines=t.splitlines();lines=[line for line in lines if not line.startswith('第一、二轮已冻结，见')]
 lines.insert(6,'第三轮已按停止条件收口，见 '+('[报告](docs/ROUND3_REPORT.md)' if p.parent==R else '[报告](../docs/ROUND3_REPORT.md)')+'：N64专用梯度通过，新壁宽示例性能跨度不足／G48网格审查失败，未启动训练与引导。无自动续跑项，下一轮未制定。')
 t='\n'.join(lines)+'\n';t=t.replace('第三轮已制定四步规划，尚未创建新实验或启动计算，执行以主规划为准。','第三轮已按停止条件收口，实际结果以第三轮报告及当前主规划为准。');p.write_text(t)
# Scientific implementation is unchanged after passing tests; clarify only its module description.
p=R/'design_fem.py';t=p.read_text().replace("Fixed-lateral Gyroid design derivatives using installed JAX-FEM's adjoint.","Fixed-lateral Gyroid responses, general adjoint and stationary K/Vf derivatives.");p.write_text(t)
for name in initial['source_before']:
 p=O/'source_after'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(R/name,p)
shutil.copy2(R/'tests/test_stationary_width.py',O/'source_after/tests/test_stationary_width.py')
(O/'README.md').write_text(report.replace('(ROUND3_REPORT.md)','(../../docs/ROUND3_REPORT.md)'))
print(json.dumps(facts,indent=2))

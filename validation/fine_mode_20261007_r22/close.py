"""Close r22 once; retain r21 and all prior science byte for byte."""
from pathlib import Path
import json,shutil,re
R=Path('/home/xuehu/projects/tpms_jax');D=R/'validation/fine_mode_20261007_r22'
W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
read=lambda p:json.loads(p.read_text());s=read(D/'analysis_summary.json');assert s['complete_result']
rows=s['rows'];c=rows[3];receipt=s['receipt'];cost=receipt['wall_seconds']+receipt.get('previous_attempt_seconds',0.)
old=read(R/'validation/critical_mode_20261007_r21/physical_metric/deformation_candidate.json')['rows'][0]
drop=100*(1-c['eigenvalue_s_minus2']/old['eigenvalue_s_minus2']);checks=read(D/'operator_checks.json')
assert not any(x['mode_converged'] for x in rows)
latest='2026-10-07 r22：第3步原12.1904%保存态N32矩阵自由模式细化已收口。200迭代四方向均未过模式残差门槛；第4变形候选仍正曲率0.7345Nmm、细空间残差0.7933，与原候选同家族而壳折叠相关性仍低。近似方向商下降不认证临界/前向精度，原因与可迁移修正未定位。原中断/未收敛保持，无新前向/Abaqus/设计AD；第4步未启动。详见[模式诊断报告](CRITICAL_MODE_PROGRESS.md)。'
next_action='原第3步的下一项改为一个有物理来源的方向诊断：复用r20快壳现有ODB中11.8632%→13.8245%两帧U/UR，先只读提取转角；元数据已确认UR存在，现有NPZ仅有位移。执行前写清有限转角/中面至厚度方向转移、宏观仿射分量、原XYZ周期及平移固定、原N32插值误差与虚域延拓，再在同一12.1904%JAX保存态评价这个完整域方向的切线曲率/材料分项和与现有候选的耦合。该有限壳增量不是特征模态，方向转移本身有近似；正曲率不证明全稳定，负方向也不直接定位静态临界点。只用它判断是否漏掉壳观察到的折叠家族/表示差异，不依赖壳来校准后续生产前向。不扫幅值、η、界面、积分或阻尼，不追加本轮谱迭代或新压缩路径；第4步仍需针对性证据。目前尚未提取UR、生成转移方向或提交新诊断。'
table='\n'.join(f"| {x['rank_lowest_block']+1} | {x['eigenvalue_s_minus2']:.6g} | {x['curvature_for_1mm_max_surface_mode_N_mm']:.5f} | {100*x['uniform_translation_inertia_fraction']:.4f}% | {x['original_fine_relative_residual']:.6f} | {x['mass_whitened_relative_residual']:.6f} |" for x in rows)
report=f'''# 周期折叠模式诊断：r22收口与下一项

2026-10-07。**原N32矩阵自由机械模式细化已执行，但没有得到收敛的临界模式。** 找到了更软的变形候选，仍未解释壳与JAX观察峰位置差；第4步完整20%没有启动。前一轮详见[r21冻结报告](history/before_fine_mode_20261007/CRITICAL_MODE_PROGRESS.md)，其科学原件与失败保持。

## 1. 问题、作用与固定条件

研究主线仍是可信薄壁TPMS背景压缩、保留有效梯度，服务后续构型/参数学习和逆设计；当前先解决跨构型前向。diverse_04快壳峰11.8241%、JAX峰15.9409%，相差4.1168个百分点，JAX峰力高约35.23%。速率对齐不足以解释差异。r21的较粗周期扰动空间找到一个正曲率变形候选，但它的原细空间残差约1、与壳折叠增量相关性很低。

本次只问：将这个候选放入原N32全部周期自由度细化，是否存在遗漏的更局部薄壁方向？使用原12.1904%动态保存态，固定宏观H；diverse_04中面、L=10mm、t=0.5mm、均匀NH E=10MPa/ν=0.3、XYZ周期波动/横向宏观应变0、HEX27 N32/27点、HRZ、η=1e−4、界面0.05mm和objective_void/C²核保持。没有改前向网格、积分、材料、质量或边界；未推进压缩时间，也未求设计AD。

## 2. 怎么算，什么才算通过

原周期自由度为786429。机械切线K仍来自同一材料能量的二阶位移导数；在原Gauss点构造dP/dF，再与原形函数梯度收缩得到K对向量的作用，不组装巨型全局矩阵。使用原HRZ对角质量M，将广义问题K v=λ M v写成质量白化算子A=M⁻¹ᐟ² K M⁻¹ᐟ²。

初始四方向为r21零附近前三个主平移方向与第4变形候选。原点固定保持，保留并分类平移，没有为了找期望模式增加前向约束。SciPy LOBPCG以这四方向为初块，求较小代数值；最多200迭代，用材料切线绝对行和/梯度所得正对角界预处理迭代。预处理只改善数值求解，不改变K或M。框架、收敛受初块与预条件影响的说明见[SciPy LOBPCG文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lobpcg.html)。

协议先规定：独立重算的质量白化残差和原广义细空间残差都≤0.001，才称该候选模式收敛；直接完整域曲率还须有限并与算子作用一致。程序正常退出不等于本征问题收敛。动态保存态不是已认证静态平衡态，有限初块的迭代也不认证全局最小或全部方向稳定。

## 3. 实际结果及数字解释

| 返回方向 | 近似方向商λ，s⁻² | 表面最大1mm归一化曲率，N·mm | 平移惯性比例 | 原细空间残差 | 白化残差 |
| --- | ---: | ---: | ---: | ---: | ---: |
{table}

四方向都未过模式残差门槛。前三个仍有99.96%以上惯性对应整体平移，不当作薄壁折叠。第4方向的平移惯性仅约0.001%，主要是变形；其近似质量方向商从r21约3.2083e7降到1.3114e7s⁻²（降低{drop:.2f}%）。这里λ尚不是收敛特征值，下降说明迭代找到较低方向商，不是反力误差降低了{drop:.2f}%。

同样把中面最大模式位移归一化到1mm，第4方向曲率由1.8825降至0.7345N·mm；模式形状/质量也有变化，不能把两者差直接称作实体弯曲刚度误差。没有给模型真的施加1mm扰动。

第4方向原细空间残差仍{c['original_fine_relative_residual']:.6f}、白化残差{c['mass_whitened_relative_residual']:.6f}，离0.001很远。两指标都在衡量K v与λM v的一致性，只是度量不同；它们**不是前向反力误差百分比**。本次没有通过临界模式提取，也不能用正曲率宣布全稳定。

该方向完整域曲率分项为深虚域{c['parts_N_mm']['deep_void']:.5f}、混合尾部{c['parts_N_mm']['mixed_tail']:.5f}、主要界面{c['parts_N_mm']['main_interface']:.5f}、几何核心{c['parts_N_mm']['geometric_core']:.5f}N·mm。深虚域φ≤0.001与混合尾部0.001<φ<0.01合计份额{100*c['continued_void_curvature_fraction']:.2f}%，φ为几何占据。这个方向仍以实体核心为主；份额不是总储能、反力误差或虚域的因果贡献，不据它调η。

第4方向与r21候选的中面面积加权绝对相关性{c['absolute_cosine_with_initial_r21_candidate']:.6f}，仍是同一形态家族；与快壳11.8632%→13.8245%折叠增量相关性仅{c['absolute_cosine_with_fast_shell_event_increment']:.5f}。相关性1表示同向、0表示正交；符号不影响模态。壳增量是有限动态变形，并不是壳特征模态，因此此比较是线索，不能作严格同模认证。

![原N32保存态返回方向，非真实变形或收敛模态](output/figures/FINE_MODE_modes.png)

图中四方向分别归一化，颜色是参考中面的法向模式分量；颜色不能直接比较真实刚度。所有原材料域保留，不用正J子域代替总能量。

## 4. 科研判断与下一项

本轮证明矩阵自由机械切线可以在原N32规模运行，也表明原受限候选可以继续软化；**没有定位真正临界模式，更没有定位主要反力差或完成可迁移修正。** 简单对角预条件下的200次迭代仍有大残差；不能把本征迭代未收敛当成JAX前向积分不可靠或TPMS路线不可行。继续只加次数/换任意初块，目前缺少直接科研依据。

候选仍与壳折叠方向差异明显。初块/迭代可能没有覆盖相应变形家族，求解条件也可能不足；它们是待判断的因素，不宣称已证明对称性、锁定或虚域是唯一原因。下一项使用实际壳折叠信息作为诊断方向来源，可以更直接检查当前遗漏方向，而不是挑扰动去拟合峰值。

{next_action}

这仍是原四步的第3步。diverse_28三个厚度已有完整20%范围保持；diverse_04跨构型约10%目标仍未过，r18仍只到17.5363%，新核完整20%路径梯度未认证。建议继续有边界的仿真研究，不承诺通用替代，不把这轮算子导数当成设计梯度；训练后置。原准静态失败、壳人工能量与漂移不确定性保留，高KE单独不阻塞动态研究。

## 5. 执行事实、成本与核对

首次准备调用约{receipt['previous_attempt_seconds']:.2f}s后手动中断，未进入特征迭代；中断前刷新日志已有前两列材料切线。原日志/源码/退出−2保存在startup_attempt01，不改写为成功。它不能单凭中断位置定位之前耗时的唯一原因。

一次准备恢复补充阶段计时，并将批量3×3通用行列式换为r21已有的等价三重积表达式，16个真实点对通用行列式核对；实际F/J不夹断，科学输入/后端/谱参数保持。恢复约{receipt['wall_seconds']:.2f}s完成200迭代及材料解释，程序退出0，但四模仍未收敛。两次调用共{cost:.2f}s（{cost/60:.2f}min），从原900s总预算扣除首次成本，没有另开900s。这是机械诊断成本，不是新前向/反向训练成本；本轮不再谱重试。

质量投影与原HRZ相对差{checks['same_original_mass_relative_error']:.3g}，原N8切线投影作用相对差{checks['projected_action_vs_original_N8_K_relative_error']:.3g}，16点JVP/jacfwd相对差{checks['H_JVP_vs_jacfwd_relative_error']:.3g}。材料切线对称性与完整域直接曲率核对有原数据；实现检查不认证机械精度。原NH必须正J区域的该态最小J约{checks['original_positive_J_min']:.4f}，只是该保存态采样事实，不认证全过程无接触/无翻转。

已有快壳INP请求U/UR/RF，实际帧元数据有UR；当前NPZ提取只保存U相关量。shell_data_inventory只读记录其可用性，没有新Abaqus作业或ODB修改。

## 6. 文件位置与整理

- 正式实验：`/home/xuehu/projects/tpms_jax/validation/fine_mode_20261007_r22/`；protocol为执行前范围，startup_attempt01保留中断，startup_recovery为成本恢复边界；result与iteration_history为返回候选及过程，analysis_summary/modes.png为解释，operator_checks为原算子核对。
- 当前报告：Windows本文件、正式docs同名和r22/REVIEW。r21原报告留r21/REVIEW及整理前备份，科学原件不改。背景/唯一规划/状态/地图同步r22。
- 本轮完成调用工具归档`history/completed_tools_20261007/fine_mode_20261007/`，不是第二套FEM；科研原件/ODB/用户论文不移动或删除。
- 旧r15～r21共490科学文件及共享核/入口/距离/PBC保持原字节。当前无运行中或排队作业；无新前向/Abaqus/设计AD，没有Git提交、推送或合并。
'''
(W/'CRITICAL_MODE_PROGRESS.md').write_text(report,encoding='utf-8')
for name in ['TPMS_RESEARCH_REVIEW.md','MECHANISM_ANALYSIS.md']:
 p=W/name;parts=p.read_text(encoding='utf-8').split('\n\n',2);p.write_text(parts[0]+'\n\n'+latest+'\n\n'+parts[2],encoding='utf-8')
p=W/'RESEARCH_PLAN.md';t=p.read_text(encoding='utf-8').replace('更新2026-10-07，r21。','更新2026-10-07，r22。',1)
t=t.replace('第3步本轮完成r21受限模式提取','第3步r21受限模式和r22原细空间细化收口')
a=t.index('## 3.');b=t.index('## 4.',a)
t=t[:a]+'''## 3. 模式诊断收口；原因与可迁移修正仍未定位

r20快壳峰只迟0.4080个百分点，速率非充分解释。r20固定跳跃方向五态曲率正，但不是特征方向。r21零附近模式识别主平移与第4变形候选，细空间残差大；失败原件保持。

'''+latest+'\n\n'+next_action+'\n\n执行前协议和实际中断/残差/成本以r22目录为准，不在本页重复堆叠。未收敛不延长原迭代或换参数凑过，不自动加入接触/塑性/稳定化。第4步仍需有依据的针对性条件。\n\n'+t[b:];p.write_text(t,encoding='utf-8')
p=W/'PROJECT_OVERVIEW.md';t=p.read_text(encoding='utf-8').replace('更新2026-10-07，r21。','更新2026-10-07，r22。',1)
t=t.replace('r20快壳及本轮r21受限模式调用已结束；','r20快壳、r21受限模式及r22原细空间细化已结束；')
a=t.index('| 梯度 |');t=t[:a]+'| r22原N32模式细化 | 12.1904%原态四方向200迭代完成，原算子作用核对；总5.32min | 四模未收敛；第4正曲率0.7345Nmm、残差0.7933、仍与壳折叠相关性低，不认证临界/原因 |\n'+t[a:]
a=t.index('近期四步：');b=t.index('\n\n最新结果：',a)
t=t[:a]+'近期四步：①现有动态证据已完成；②一次快壳速率诊断已完成；③r21受限模式与r22原N32细化收口，临界/原因未定位，下一项只用现有壳U/UR准备实际折叠方向诊断；④完整20%JAX条件未成立，未启动。执行只看唯一主规划，旧历史不生成待办。'+t[b:];p.write_text(t,encoding='utf-8')
p=W/'RESEARCH_BACKGROUND.md';t=p.read_text(encoding='utf-8').replace('r21已有受限变形候选，但细空间残差约1、壳相关性低，下一项仍需原N32机械模式诊断；','r22原N32细化后候选仍未收敛、壳相关性低，下一项用现有壳U/UR作有来源的折叠方向诊断；');p.write_text(t,encoding='utf-8')
p=W/'FILE_MAP.md';t=p.read_text(encoding='utf-8').replace('更新2026-10-07，r21。','更新2026-10-07，r22。',1)
t=t.replace('r21周期模式、平移分类、候选/限制；docs同名、r21/REVIEW.md','当前r22原N32细化、残差/域解释/下一方向；docs同名、r22/REVIEW.md；r21原件留r21/REVIEW')
t=t.replace('| validation/critical_mode_20261007_r21/ |','| validation/fine_mode_20261007_r22/ | 一态矩阵自由模式细化；原协议/中断/恢复/迭代/返回候选/原算子核对；四模未收敛，无新路径/AD |\n| validation/critical_mode_20261007_r21/ |',1)
t=t.replace('原N32模式细化尚未生成或排队。','此为r21时点状态；当前r22已执行收口，未收敛保持。')
t+='\n## r22位置与状态\n\n正式validation/fine_mode_20261007_r22/含490原科学文件冻结清单、输入/源码、startup_attempt01和startup_recovery、实际result/iteration_history、模式图、材料域解释及壳数据库存。旧r21与共享核/入口不改。Windows备份history/before_fine_mode_20261007，正式docs/history同名；完成调用工具归档history/completed_tools_20261007/fine_mode_20261007。下一壳U/UR转移诊断只记录为主规划，未提取/未排队；现有快壳ODB可复用，不另算壳。\n';p.write_text(t,encoding='utf-8')
p=W/'AGENTS.md';lines=p.read_text(encoding='utf-8').splitlines()
for i,line in enumerate(lines):
 if line.startswith('- 当前2026-10-07 r21唯一规划：'):
  lines[i]='- 当前2026-10-07 r22唯一规划：第1/2步完成；第3步r21受限模式与r22一态原N32矩阵自由细化收口，四方向200迭代均未过残差。第4变形候选正曲率0.7345Nmm、细空间/白化残差0.7933/0.9662；与原候选同家族、壳折叠相关性0.02755，非临界/全稳定认证，原因及可迁移修正未定位。原中断249.83s与一次同算子准备恢复69.17s共5.32min，失败/未收敛保留，不追加本轮谱迭代。下一项仅复用r20现有11.8632%→13.8245%壳U/UR准备物理折叠方向转移，再在12.1904%原JAX态做完整域方向曲率/耦合诊断；先明确有限转角/宏观仿射/周期/插值/虚域延拓协议，有限增量非特征模态，不用于曲线校准。UR元数据确认存在但尚未提取。第4步未成立，无新前向/Abaqus/设计AD/排队；r15～r21共490科学文件、共享核/入口/HRZ冻结。r20快壳峰11.8241%、JAX15.9409%，速率非充分解释，r18仅17.5363%预算停；高KE不单独阻塞动态，旧准静态失败和壳不确定性保持。动态KE选项未实现，控制默认关闭，block可选dt默认AST等价；详见CRITICAL_MODE_PROGRESS.md。'
p.write_text('\n'.join(lines)+'\n',encoding='utf-8')
start='# TPMS研究阅读入口\n\n依次读[研究背景](RESEARCH_BACKGROUND.md)、[唯一近期主规划](RESEARCH_PLAN.md)、[当前状态](PROJECT_OVERVIEW.md)、[文件地图](FILE_MAP.md)。方法/范围见[综合说明](TPMS_RESEARCH_REVIEW.md)。\n\n'+latest+'\n\n正式程序仅WSL `/home/xuehu/projects/tpms_jax`，Windows阅读/历史；无运行中或排队作业，训练后置。\n'
(W/'START_HERE.md').write_text(start,encoding='utf-8')
def formal_text(t):
 t=t.replace('(PROJECT_OVERVIEW.md)','(RESEARCH_STATUS.md)').replace('(GEOMETRY_TRANSFER_REVIEW.md)','(../validation/geometry_transfer_review_20261006_r16/REVIEW.md)')
 t=t.replace('(history/before_fine_mode_20261007/CRITICAL_MODE_PROGRESS.md)','(../validation/critical_mode_20261007_r21/REVIEW.md)')
 t=t.replace('[归档索引](history/completed_tools_20261006/README.md)','`Windows history/completed_tools_20261006/README.md`')
 for label,path in [('VOID_CONTINUATION_response.png','void_continuation_20261006_r12/response.png'),('DYNAMIC_BRANCH_response_energy.png','dynamic_branch_review_20261007_r19/response_energy.png'),('SHELL_RATE_response.png','shell_rate_20261007_r20/analysis/response.png'),('SHELL_RATE_modes.png','shell_rate_20261007_r20/analysis/modes.png'),('CRITICAL_MODE_modes.png','critical_mode_20261007_r21/analysis/modes.png'),('FINE_MODE_modes.png','fine_mode_20261007_r22/modes.png')]:t=t.replace('(output/figures/'+label+')','(../validation/'+path+')')
 return t
for name in ['RESEARCH_BACKGROUND.md','RESEARCH_PLAN.md','PROJECT_OVERVIEW.md','FILE_MAP.md','TPMS_RESEARCH_REVIEW.md','MECHANISM_ANALYSIS.md','CRITICAL_MODE_PROGRESS.md']:
 (R/'docs'/('RESEARCH_STATUS.md' if name=='PROJECT_OVERVIEW.md' else name)).write_text(formal_text((W/name).read_text(encoding='utf-8')),encoding='utf-8')
(D/'REVIEW.md').write_text(report.replace('(output/figures/FINE_MODE_modes.png)','(modes.png)').replace('(history/before_fine_mode_20261007/CRITICAL_MODE_PROGRESS.md)','(../critical_mode_20261007_r21/REVIEW.md)'),encoding='utf-8')
t=(W/'AGENTS.md').read_text(encoding='utf-8')
for name in ['RESEARCH_BACKGROUND.md','RESEARCH_PLAN.md','PROJECT_OVERVIEW.md','FILE_MAP.md','TPMS_RESEARCH_REVIEW.md']:t=t.replace('('+name+')','(docs/'+('RESEARCH_STATUS.md' if name=='PROJECT_OVERVIEW.md' else name)+')')
(R/'AGENTS.md').write_text(t,encoding='utf-8');t=start
for name in ['RESEARCH_BACKGROUND.md','RESEARCH_PLAN.md','PROJECT_OVERVIEW.md','FILE_MAP.md','TPMS_RESEARCH_REVIEW.md','CRITICAL_MODE_PROGRESS.md']:t=t.replace('('+name+')','(docs/'+('RESEARCH_STATUS.md' if name=='PROJECT_OVERVIEW.md' else name)+')')
(R/'README.md').write_text(t,encoding='utf-8')
vp=R/'validation/README.md';vp.write_text('2026-10-07最新r22：原N32模式细化200迭代未过残差，不认证临界或原因；下一仅主规划的壳折叠方向诊断，零新路径/AD。见[本轮报告](fine_mode_20261007_r22/REVIEW.md)及[唯一规划](../docs/RESEARCH_PLAN.md)。\n\n'+vp.read_text(encoding='utf-8'),encoding='utf-8')
(D/'decision.json').write_text(json.dumps({'plan_step':3,'four_returned_modes_converged':0,'true_critical_mode_not_identified':True,'cause_not_uniquely_identified':True,'step4_not_started':True,'no_more_r22_spectrum_retries':True,'next_action':next_action,'new_forward':0,'new_Abaqus':0,'new_design_AD':False,'shared_core_changes':False,'current_running_or_queued_jobs':0,'Git_commit_push_merge':False,'total_launch_seconds':cost},indent=2,ensure_ascii=False))
shutil.copy2(W/'work/fine_mode_20261007/close.py',D/'close.py')
print(json.dumps({'report':'CRITICAL_MODE_PROGRESS.md','total_seconds':cost,'returned_modes_converged':0,'no_new_forward_or_AD':True}))

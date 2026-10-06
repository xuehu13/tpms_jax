"""Record the bounded failed transfer attempt; retain every original scientific byte."""
from pathlib import Path
import json,shutil
R=Path('/home/xuehu/projects/tpms_jax');O=R/'validation/geometry_transfer_20261006_r15'
W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
D=W/'work/geometry_forward_20261006'
m=json.loads((O/'step2_summary.json').read_text());j=m['JAX'];s=m['shell'];p=j['last_observed_progress']
assert not m['step2_successful_completion'] and m['status']=='attempted_not_accepted_at_20pct'
def sub(text,old,new):
    assert text.count(old)==1,old
    return text.replace(old,new)
detail=f"""本轮第2步已执行，但未达到双方20%可信前向的目标，不标成功完成。固定diverse_04原中面、0.50mm厚度、NH/XYZ/HEX27/HRZ/C²及全部原数值常数；仅使用已有CPU参考几何/GPU推进选项匹配CPU缓存，未改内力、质量、算法或匹配容差。

JAX首次非有限拒绝块的前一有效监测点为{100*j['first_nonfinite_rejected_block_previous_compression']:.3f}%压缩，当时动能/内能仅{100*j['first_rejection_last_valid_KE_over_U']:.3f}%；之后3次拒绝减步，从原dt降为1/8。最后监测到{100*p['compression']:.3f}%压缩、动能/内能{100*p['KE_over_U']:.2f}%，稀疏接受记录中最高约{100*j['maximum_KE_over_U_in_sparse_logged_rows']:.2f}%。按本规划实际惯性/完成疑点的停止条件中止，完整入口耗时{j['wall_seconds']/60:.2f}min；不是达到15min预算限制、不是已证明所有时间步/材料点都失败。监测的未续接NH域无失效点，不能据此认定拒绝原因已定位；虚域实际翻转保持。

Abaqus Explicit完成20%及保载，调用耗时{s['wall_seconds']/60:.2f}min，末半段保载反力约{s['hold_mean_Fz_N']:.6f}N。但加载时长中动能/内能≤5%的比例仅{100*s['loading_time_fraction_KE_below_5pct']:.2f}%（门槛95%），能量漂移/输入功{100*s['energy_drift_relative_work']:.2f}%（门槛1%），人工能量/输入功最大{100*s['artificial_energy_relative_max_work']:.2f}%（门槛5%）。末态动能/内能{100*s['terminal_KE_over_IE']:.2f}%虽通过，不能覆盖加载过程的三项失败；该壳结果只作诊断参照，不能直接当可信准静态标准或认定JAX误差。

原JAX入口中断未被Exception保存分支捕获，未留有效末态q/v及完整接受路径；只有{j['sparse_logged_observations']}条原日志/拒绝块last_valid中的稀疏接受观察、最新progress和3份拒绝场。拒绝场不能冒充有效末态，不能做原定完整20%曲线误差、末态模式或27/125点有效域验收。这个记录缺口明确保留，不重构伪状态、不自动重跑补数据。第一次Gauss设备路径启动失败及壳提取漏input.json的日志亦保留；后者只修正只读提取，ODB未改，壳零重跑。

当前结论：diverse_04迁移目标未通过；旧diverse_28已验证范围不改，整体路线没有由单一新工况被否定，也不能承诺普遍替代。下一项仅第3步基于现有记录定位失稳/非有限来源及壳质量，先区分动态阶段、数值问题与几何/模式问题；当前原因尚未认证。必要修复/速率/记录保全或新作业必须先形成针对性方案并更新唯一规划，不自动改参数、重跑、AD或训练。
"""
x=W/'RESEARCH_PLAN.md';t=x.read_text();t=sub(t,
 '更新2026-10-06；第1步匹配输入已完成，依据仓库7e9c642及新r15输入/诊断。当前第2/3/4步待执行，下一项仅第2步一次双方20%前向。本轮零位移求解/Abaqus作业/完整AD/训练；新输入与旧科学证据分别留存。',
 '更新2026-10-06；依据86d55b2输入及r15第2步记录。第1步完成；第2步已执行但未达到双方20%可信前向目标：JAX因实际质量疑点停止，壳完成但三项质量未过。第3/4步待执行，下一项仅第3步现有记录限定诊断，零新增作业/完整AD/训练。')
t=t.replace('已建立r15输入目录；未提交压缩作业','r15保留一次JAX中止和一次完整壳作业，未追加作业')
t=t.replace('## 2. 各完成一次20%前向：待执行','## 2. 各尝试一次20%前向：已执行，目标未通过')
t=sub(t,'## 3. 对照并定位实际差异：待执行',detail+'\n## 3. 对照并定位实际差异：待执行（限定现有证据）')
t=t.replace('科研问题：新构型的整体响应一致性是否仍达到工作目标，差别是量级还是变形机制？',
 '科研问题：本次在何阶段失去可信前向条件，现有证据支持哪些原因、还缺哪些证据？由于JAX未完成、有效末态/完整接受路径缺失，壳三项质量未过，下表保留原验收口径但当前不能进行完整迁移验收。先用真实稀疏接受记录、拒绝场和完整壳记录诊断；不将插值或拒绝场冒充有效20%结果。')
t=t.replace('下一次推进从本轮第2步开始；输入就绪不是反力/屈曲响应通过，尚无新20%结果。',
 '下一次推进仅第3步限定诊断；当前没有可信的双方20%对照或新构型梯度认证。若需要新增修复/计算，先以实际问题更新本规划。')
x.write_text(t)
x=W/'PROJECT_OVERVIEW.md';t=x.read_text();t=sub(t,
 '更新2026-10-06；r15第1步匹配输入已完成，依据基线7e9c642。第2/3/4步待执行，下一项仅[唯一主规划](RESEARCH_PLAN.md)第2步双方20%前向。新增Gauss占据/质量/XYZ壳INP，零位移求解/Abaqus作业/完整AD/训练。',
 '更新2026-10-06；r15第2步已执行但目标未通过：JAX最后监测17.44%后按质量条件停止，壳完成20%及保载但三项质量未过。下一项仅[唯一主规划](RESEARCH_PLAN.md)第3步现有证据限定诊断；零追加作业/完整AD/训练。')
t=sub(t,'| 新第2/3/4步 | 待执行：一次双方20%前向→对照实际差异→迁移范围与继续条件；不存在新仿真结果 |',
 '| 新第2步 | 已执行，未达到双方20%可信前向目标；JAX中止、壳三项质量未过；未认证新构型精度 |\n| 新第3/4步 | 待执行：现有稀疏接受/拒绝场及完整壳诊断→迁移范围/继续条件；不自动重跑 |')
t=t.replace('不同几何已完成输入，第2步待给实际响应证据；','不同几何前向尝试显示当前迁移限制，第3步待定位问题；')
t+='\n'+detail
x.write_text(t)
x=W/'RESEARCH_BACKGROUND.md';t=x.read_text().replace(
 '第1步限定输入就绪已完成，下一项为一次双方20%前向，当前没有新几何响应结果。',
 '第1步输入完成；第2步已尝试但diverse_04迁移目标未通过，JAX因质量疑点停止、壳三项质量未过；下一项仅利用现有记录诊断，不改写旧范围或自动追加作业。')
x.write_text(t)
x=W/'START_HERE.md';t=x.read_text().replace('下一项仅第2步双方20%前向，尚无新压缩结果。',
 '第2步已执行但迁移目标未通过，JAX中止、壳质量未过；下一项仅第3步现有记录限定诊断，不自动追加作业。')
x.write_text(t)
x=W/'AGENTS.md';t=x.read_text();t=sub(t,
 'r15第1步限定输入就绪已完成，下一项仅第2步；真实Gauss/HRZ/XYZ S3R INP已建立但零压缩作业。',
 'r15第1步限定输入就绪已完成，第2步已执行但双方20%可信前向目标未通过；下一项仅第3步现有记录限定诊断。JAX约15.97%前后首次非有限拒绝，3次减步后最后监测17.44%、KE/U19.74%，按实际质量停止；无有效末态或完整接受路径，稀疏14观察和3拒绝场不冒充有效20%。壳完成20%但加载惯性/能量漂移/人工能量三项未过。step2_summary/STEP2保留所有事实，旧范围不改；不自动新增速率/补场/修复作业或AD。首CPU缓存设备检查失败无时间推进，已有CPU参考/GPU推进路径后原检查通过，源/常数/容差不改；壳只修正只读提取漏参，零重跑。')
x.write_text(t)
x=W/'FILE_MAP.md';t=x.read_text().replace('第1步输入已就绪，第2步待执行','第1步完成，第2步目标未通过，第3步限定诊断待执行')
t=t.replace('diverse_04输入/Gauss占据/HRZ/XYZ壳包；原4条法向失败、首出口诊断、step1_decision限定就绪；零压缩/AD',
 'diverse_04匹配输入及第2步JAX中止/壳完整诊断；step1_decision与STEP2/step2_summary分别留输入与执行事实；零完整AD')
t=t.replace('当前没有新ODB或压缩响应。','当前有完整壳ODB/响应及JAX稀疏接受观察/拒绝场，没有有效JAX20%末态。')
t=t.replace('；无ODB。科学数组、首次日志/源码原字节保留，计时未知不补造。',
 '；有一次壳ODB/日志。第1步科学数组/首次日志原字节保留，首次预处理计时未知不补造。')
t+='\n第2步：r15的T0p004为空输出目录及缓存启动失败收据；实际JAX尝试为T0p004_cpu_reference，含input/progress/rejected_blocks、3拒绝场和source_at_run，无result/有效末态。operator_stop.json保留停止前监测；accepted_logged_observations.json只提取14条真实稀疏接受观察，不等于完整曲线。abaqus/explicit_T0p040/results保留壳JSON/场/日志/retention，ODB留原生不重复复制。step2_summary.json和STEP2.md分别为结构化与可读事实。Windows work/geometry_forward_20261006仅本次调用/提取/留存/发布工具，非第二套FEM。\n'
x.write_text(t)
x=W/'TPMS_RESEARCH_REVIEW.md';t=x.read_text().replace(
 'diverse_04单一几何迁移第1步已达限定输入就绪，未提交位移/Abaqus/AD作业。',
 'diverse_04第1步输入完成，第2步已尝试但迁移目标未通过：JAX质量中止、壳完成但三项质量未过。第3步限定诊断待执行，零完整AD。')
t=t.replace('第1步输入已完成，第2步双方前向待执行。',
 '第1步输入完成，第2步目标未通过，第3步现有证据限定诊断待执行。')
t=t.replace('真正的壳响应尚未求解。','壳已完成20%及保载但三项质量未过；JAX最后监测17.44%后质量中止，未形成可信双方20%对照。')
t+='\n'+detail
x.write_text(t)
mapping={'RESEARCH_BACKGROUND.md':'docs/RESEARCH_BACKGROUND.md','RESEARCH_PLAN.md':'docs/RESEARCH_PLAN.md',
 'PROJECT_OVERVIEW.md':'docs/RESEARCH_STATUS.md','FILE_MAP.md':'docs/FILE_MAP.md','TPMS_RESEARCH_REVIEW.md':'docs/TPMS_RESEARCH_REVIEW.md',
 'START_HERE.md':'README.md','AGENTS.md':'AGENTS.md'}
for src,dest in mapping.items():
    t=(W/src).read_text().replace('PROJECT_OVERVIEW.md','RESEARCH_STATUS.md').replace('output/figures/','figures/')
    if dest=='docs/FILE_MAP.md':t=t.replace('[归档索引](history/completed_tools_20261006/README.md)','Windows history/completed_tools_20261006/README.md（Windows归档索引）')
    if dest=='docs/TPMS_RESEARCH_REVIEW.md':t=t.replace('figures/VOID_CONTINUATION_response.png','../validation/void_continuation_20261006_r12/response.png')
    if dest in ['AGENTS.md','README.md']:
        for name in ['RESEARCH_BACKGROUND.md','RESEARCH_PLAN.md','RESEARCH_STATUS.md','FILE_MAP.md','TPMS_RESEARCH_REVIEW.md','THICKNESS_RANGE_PROGRESS.md','VOID_CONTINUATION_PROGRESS.md','PAPER_ROUTE.md']:
            t=t.replace(']('+name+')','](docs/'+name+')')
    (R/dest).write_text(t)
x=R/'scripts/README.md';t=x.read_text().replace(
 'diverse_04第1步限定输入就绪已完成，下一项仅[主规划](../docs/RESEARCH_PLAN.md)第2步一次双方前向',
 'diverse_04第2步已执行但目标未通过，下一项仅[主规划](../docs/RESEARCH_PLAN.md)第3步现有证据限定诊断')
t+='\nr15 CPU参考缓存需沿已有--geometry-on-cpu及JAX_PLATFORMS=cuda,cpu读取，原逐位检查不放宽；实际内力/推进仍在CUDA。本轮中断缺少有效末态，入口未处理KeyboardInterrupt保存，不把拒绝场当末态；必要保全改动仅经主规划另行安排。\n'
x.write_text(t)
x=R/'validation/README.md';t=x.read_text().replace(
 'diverse_04的r15第1步匹配输入已达限定就绪，下一项仅[主规划](../docs/RESEARCH_PLAN.md)第2步；没有新位移求解/Abaqus作业/完整AD。',
 'diverse_04的r15第2步已执行但目标未通过，下一项仅[主规划](../docs/RESEARCH_PLAN.md)第3步现有记录诊断；零追加作业/完整AD。')
t=t.replace('diverse_04新Gauss/HRZ/XYZ壳输入；原法向末点失败保留、首出口诊断后限定输入就绪；零压缩/AD',
 'diverse_04匹配输入及一次JAX中止/壳诊断；[第2步记录](geometry_transfer_20261006_r15/STEP2.md)保留限制，未认证迁移')
x.write_text(t)
assert not (O/'STEP2.md').exists()
(O/'STEP2.md').write_text('# diverse_04：第2步已执行，20%迁移目标未通过\n\n科研问题：固定背景方法能否迁移到不同中面？本步给真实执行与质量事实，未做完整响应误差/模式分析。\n\n'+detail+'\n路径见[地图](../../docs/FILE_MAP.md)，执行只看[唯一规划](../../docs/RESEARCH_PLAN.md)。第1步原件、r6–r14均冻结；step2_summary给新事实，不改写旧失败和旧认证。\n')
shutil.copy2(__file__,O/'step2_tools/update_docs.py')
print(json.dumps({'step2':'attempted target not passed','next_only':'step3 existing evidence diagnosis'},ensure_ascii=False))

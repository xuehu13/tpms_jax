from pathlib import Path
import hashlib,json,shutil
W=Path(r'C:\Users\xuehu\Documents\Codex\2026-10-01\referenced-chatgpt-conversation-this-is-an')
R=Path(r'\\wsl.localhost\Ubuntu-24.04\home\xuehu\projects\tpms_jax')
B=R/'validation/platen_xy_diverse04_20261012_r54';O=W/'output/r54_platen_xy_diverse04_20261012'
H=W/'history/execution_diverse04_20261012_r54';H.mkdir(parents=True,exist_ok=True)
assert not (B/'checks/pair_started.json').exists()
for name in ['run_pair.py','compare_peaks.py','bootstrap.py','install.py']:
 p=W/'work/r54_stage'/name;compile(p.read_text(encoding='utf-8'),str(p),'exec');shutil.copy2(p,B/'tools'/name)
mapping={'AGENTS.md':'AGENTS.md','RESEARCH_BACKGROUND.md':'docs/RESEARCH_BACKGROUND.md','RESEARCH_PLAN.md':'docs/RESEARCH_PLAN.md',
 'PROJECT_OVERVIEW.md':'docs/RESEARCH_STATUS.md','FILE_MAP.md':'docs/FILE_MAP.md','START_HERE.md':'README.md','output/README.md':'validation/README.md'}
manifest={}
for a,b in mapping.items():
 for root,rel,label in [(W,a,'windows'),(R,b,'wsl')]:
  p=root/rel;dest=H/'before'/label/rel;dest.parent.mkdir(parents=True,exist_ok=True)
  assert not dest.exists();shutil.copy2(p,dest);manifest[label+'/'+rel]=hashlib.sha256(p.read_bytes()).hexdigest()
(H/'before_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
link='output/r54_platen_xy_diverse04_20261012/README.md'
notice='2026-10-12新增授权：仅执行diverse_04一组20%共同中面端缘压板／XY周期／横向自由对照，准备后顺序运行Abaqus与JAX，比较总力、峰位、功、保载及模式。总预算100 min（JAX65、Abaqus25、其余10），原参数不拟合、每端一次尝试；数值失败或预算停止保留部分路径。梯度研究继续暂停，heartbeat tpms保持PAUSED；不扩展AD／FD、训练或Git。'
p=W/'AGENTS.md';s=p.read_text(encoding='utf-8');s=s.replace('用户已选择先看总结、不启动新完整反向，并要求停止每30分钟自动检查。heartbeat id=tpms为PAUSED；当前无活动求解、AD或FD。没有新的明确研究指示前不新增计算、不恢复队列，不宣布四步完成。训练、新Abaqus、P10/12和Git提交后置。',notice+' 原梯度四步未全部完成，门槛与失败不改。');p.write_text(s,encoding='utf-8')
p=W/'RESEARCH_BACKGROUND.md';s=p.read_text(encoding='utf-8').replace('当前按用户要求暂停新计算与自动检查，只整理已有证据。','梯度与自动检查继续暂停；当前仅按用户新指示执行diverse_04共同端缘压板20%成对前向对照。');p.write_text(s,encoding='utf-8')
p=W/'RESEARCH_PLAN.md';old=p.read_text(encoding='utf-8');tail=old[old.index('## 原四步实际完成程度'):];tail=tail.replace('当前范围仍仅t、w及一个a；','原梯度范围仍仅t、w及一个a；').replace('P10/12、新Abaqus、大压缩反向、训练和Git提交继续后置。','P10/12、梯度相关新Abaqus、大压缩反向、训练和Git提交继续后置。')
s='# 唯一主规划：diverse_04共同端缘压板20%成对对照\n\n'+notice+'\n\n1. 复用原中面和N32占据场，生成与G/P/D同口径的新输入；冻结参数、峰定义、分母和预算。\n2. 完成一组Abaqus／JAX的4 ms加载至20%及0.2 ms保载；保留数值失败与实际成本。\n3. 比较原始全局峰、预定平滑首个显著峰、完整／分段曲线、加载功、保载、能量和变形模式；不平移曲线或挑峰。\n4. 汇总可比范围、结论与限制，更新已有导航及r54独立报告；完成即停止，不自动恢复旧梯度规划。\n\n当前阶段：输入检查完成，尚未提交动态计算。报告位置：[r54对照]('+link+')。原XYZ与新加载边界、横向自由度及历史时长可能不同，不能把新旧差全部归因于某一个因素。\n\n'+tail;p.write_text(s,encoding='utf-8')
p=W/'PROJECT_OVERVIEW.md';s=p.read_text(encoding='utf-8');s=s.replace('# 当前状态：梯度证据已整理，计算与自动检查暂停','# 当前状态：新增diverse_04前向对照；梯度继续暂停',1).replace('当前无活动求解／AD／FD任务；每30分钟heartbeat id=tpms已暂停。','当前仅授权diverse_04新压板模型成对前向对照，输入准备中；每30分钟heartbeat id=tpms继续暂停，AD／FD不启动。',1)
s=s.replace('## 前向响应：限定加载条件下G/P结果积极','## 当前新增对照r54\n\n'+notice+' [独立报告]('+link+')。待实际求解后才给峰位及误差结论。\n\n## 前向响应：限定加载条件下G/P结果积极',1);p.write_text(s,encoding='utf-8')
p=W/'FILE_MAP.md';s=p.read_text(encoding='utf-8').replace('当前计算与自动检查暂停。','当前只有r54成对前向任务获新授权，自动检查及梯度继续暂停。',1)
s=s.replace('## 当前梯度批次r53','## 当前前向批次r54\n\n正式`R/validation/platen_xy_diverse04_20261012_r54`；阅读`W/output/r54_platen_xy_diverse04_20261012`；原生`A/r54_diverse04_midsurface_end_20pct_20261012`。独立保存协议、输入、原曲线、峰规则／CSV、图和成本。复用r15的surface_geometry.npz及gauss_field.npz，不改原数组；轻量适配复用r51/r52程序。\n\n## 暂停的梯度批次r53',1);p.write_text(s,encoding='utf-8')
p=W/'START_HERE.md';s=p.read_text(encoding='utf-8').replace('**当前暂停新计算及每30分钟自动检查，先阅读梯度总结。**','**当前仅新增diverse_04共同端缘压板20%前向对照；梯度与自动检查继续暂停。**',1).replace('- [梯度研究完整回顾','- [diverse_04最新模型20%对照]('+link+')：本轮协议、实际峰位和曲线；未完成前不提前给结论。\n- [梯度研究完整回顾',1);p.write_text(s,encoding='utf-8')
p=W/'output/README.md';s=p.read_text(encoding='utf-8').replace('当前计算和每30分钟自动检查均暂停。','梯度和每30分钟自动检查均暂停；用户另授权[r54 diverse_04新压板20%对照](r54_platen_xy_diverse04_20261012/README.md)。',1).replace('当前无活动任务，原Word','该旧批次已结束，原Word').replace('当前无活动作业。','该批次已结束。');p.write_text(s,encoding='utf-8')
# Correct relative links for the established formal docs layout.
for a,b in mapping.items():
 if a=='output/README.md':continue
 s=(W/a).read_text(encoding='utf-8')
 s=s.replace('output/r54_platen_xy_diverse04_20261012/','validation/platen_xy_diverse04_20261012_r54/').replace('output/r53_platen_xy_gradients_20261011/','validation/platen_xy_gradients_20261011_r53/').replace('output/r52_platen_xy_PD_20261010/','validation/platen_xy_PD_20261010_r52/').replace('output/r51_platen_xy_G_20261010/','validation/platen_xy_G_20261010_r51/experiments/../')
 for w,f in [('PROJECT_OVERVIEW.md','RESEARCH_STATUS.md')]:s=s.replace(w,f)
 if b.startswith('docs/'):
  s=s.replace('](validation/','](../validation/').replace('](output/README.md)','](../validation/README.md)')
 else:
  for d in ['RESEARCH_BACKGROUND.md','RESEARCH_PLAN.md','RESEARCH_STATUS.md','FILE_MAP.md','TPMS_RESEARCH_REVIEW.md']:s=s.replace(']('+d+')','](docs/'+d+')')
 (R/b).write_text(s,encoding='utf-8')
p=R/'validation/README.md';s=p.read_text(encoding='utf-8').replace('当前计算与每30分钟自动检查暂停，无活动求解／AD／FD。','当前仅执行新授权[r54 diverse_04共同端缘压板20%](platen_xy_diverse04_20261012_r54/README.md)前向对照；梯度和每30分钟自动检查继续暂停。',1);p.write_text(s,encoding='utf-8')
text='# diverse_04共同端缘压板20%对照r54\n\n'+notice+'\n\n当前输入检查完成，接下来生成INP及接触缓存并提交一次成对动态任务。尚无新数值结果，不作误差或峰位结论。参数见protocol.json，几何预检见geometry_precheck.json，运行状态见status.json。\n'
(O/'README.md').write_text(text,encoding='utf-8');(B/'README.md').write_text(text.replace('geometry_precheck.json','checks/geometry_precheck.json'),encoding='utf-8')
shutil.copytree(H,B/'checks/navigation_before',dirs_exist_ok=True)
print('Installed adapters and navigation; no numerical job launched.')

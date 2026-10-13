"""Publish actual r54 outcome after coordinator exits; no solver or AD calls."""
from pathlib import Path
import hashlib,json,shutil
W=Path(r'C:\Users\xuehu\Documents\Codex\2026-10-01\referenced-chatgpt-conversation-this-is-an')
R=Path(r'\\wsl.localhost\Ubuntu-24.04\home\xuehu\projects\tpms_jax')
B=R/'validation/platen_xy_diverse04_20261012_r54';E=B/'diverse_04';O=W/'output/r54_platen_xy_diverse04_20261012'
H=W/'history/execution_diverse04_20261012_r54'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
s=read(B/'status.json');assert not s['active_jobs'] and (B/'checks/source_guard_after.json').exists()
guard=read(B/'checks/source_guard_before.json');assert all(sha(R/p)==v for p,v in guard.items())
m=read(E/'analysis/peak_metrics.json');cost=read(E/'analysis/costs.json');full=read(E/'analysis/metrics.json')
assert s['complete_pair'] and m['paired_complete20_and_hold'],'Incomplete pair needs an explicit partial-range report'
assert sha(E/'jax/result.json')==m['source_sha256']['JAX'] and sha(E/'abaqus/result.json')==m['source_sha256']['Abaqus']
shutil.copy2(E/'analysis/metrics.json',B/'checks/original_derived_metrics_before_context_cleanup.json')
removed={k:full.pop(k) for k in ['contextual_original2pct_peak_N','curve_RMS_over_old_fixed_peak'] if k in full}
for p in [E/'analysis/metrics.json',O/'metrics.json']:p.write_text(json.dumps(full,indent=2)+'\n',encoding='utf-8')
(B/'checks/context_cleanup.json').write_text(json.dumps({'removed_unrelated_G_template_fields':removed,'reason':'r54 uses its own new Abaqus full-loading peak only; no response or criterion changed','original_derived_sha256':sha(B/'checks/original_derived_metrics_before_context_cleanup.json')},indent=2)+'\n',encoding='utf-8')
summary=f"diverse_04两端完整20%及保载完成；曲线RMS／壳峰{100*m['primary_curve_RMS_over_shell_peak']:.3f}%，加载功差{100*m['loading_work_relative_bias']:+.3f}%，保载均力差{100*m['hold_mean_relative_bias']:+.3f}%。总墙钟{cost['overall_minutes']:.2f} min。"
nopeak=all(not p['peaks'] for p in m['smoothed_peaks'].values())
raw=m['raw_global_peak']
peaknote=(f"两端均无符合预定规则的显著内部降载峰；原始最大力位置为JAX {raw['JAX']['compression_percent']:.4f}%、Abaqus {raw['Abaqus']['compression_percent']:.4f}%。不能以全局最大值代替同一屈曲事件的峰位核对。" if nopeak else '显著峰的全部位置、峰力及差异见独立报告，不以端点最大值代替屈曲峰。')
interpretation='本次把原diverse_04体内参数保持为既有N32连续占据模型，换成与G/P一致的共同中面端缘压板、XY周期与自由横向伸缩。在这套限定边界下，宏观响应接近与否按本次三项指标判断。相较旧XYZ模型，接触、Z约束及横向宏观自由度同时改变；这是加载条件迁移验证，不能据此把旧误差全部归因于单一因素。没有显著内部峰时，也不能声称已证明复杂跳跃屈曲的峰位预测准确。'
health=full['JAX_sampled_health'];fields=full['field_comparison']
lastfield=next(x for x in fields if x['target_pct']==20)
localnote=f"20%附近中面位移相对RMS差为{100*lastfield['relative_displacement_RMS']:.2f}%，高于宏观力曲线差；尚不能宣布局部响应完全一致。共同端缘接触没有阻挡后续进入板面的其他壁点：同一保存时间的中面越板最大值JAX {lastfield['JAX_all_midsurface_outside_platens_max_mm']:.4f} mm、Abaqus {lastfield['Abaqus_all_midsurface_outside_platens_max_mm']:.4f} mm；JAX有限宽切口φ≥0.01点越板最大{lastfield['JAX_finite_width_cut_face_outside_platen_max_mm']['0.01']:.4f} mm。这些位置不同于受选端缘罚穿透，不能混作接触精度。没有完整壁面或自接触。"
ft='\n'.join(f"| {x['target_pct']}%附近 | {x['displacement_RMS_mm']:.6f} | {100*x['relative_displacement_RMS']:.3f}% |" for x in fields)
extra=f'''\n## 综合判断与数值健康\n\n**{summary}** 三个预定约10%宏观工作目标：{'均达到' if m['three_working_targets_within10pct'] else '未全部达到'}。{peaknote}\n\n{interpretation}\n\nJAX实际推进{health['steps']}步，无效状态回退{health['rejected_blocks']}次；已接受保存样本中材料域实际J最小{health['minimum_required_actual_J']:.6f}，全域最小{health['minimum_all_actual_J']:.6f}。采样功平衡缺口／最大采样功{100*health['energy_work_defect_over_max_sampled_work']:.4f}%；末端动能／内能{100*health['terminal_KE_over_IE']:.4f}%。这些是有限样本的健康检查，不是全过程无自接触或全局稳定性证明。Abaqus能量、周期残差与端缘穿透见metrics.json。\n\n中面位移比较在同一实际时间进行，JAX用Q2插值，Abaqus在相邻输出帧间插值，没有刚体对齐或位移放大。相对位移RMS以对应Abaqus中面位移RMS为分母；总位移一致不等于局部应变或局部分支均已认证。\n\n| 保存检查点 | 位移差RMS mm | 相对位移RMS |\n|---|---:|---:|\n{ft}\n\n本轮至此停止。每30分钟自动检查保持暂停；r53梯度失败、门槛和未启动的三参数反向保持，不由本次前向通过代替。无Git提交。\n'''
extra=extra.replace('\n本轮至此停止。','\n'+localnote+'\n\n执行墙钟从本次输入生成起计，包含准备、编译、求解、提取和自动分析；前置代码审阅、适配开发和文档维护不计入求解执行成本。复用G绘图器带出的旧G归一化字段已从本轮指标表中移除，原派生表原字节另存checks；本次唯一分母是新diverse_04壳加载峰，原始两端曲线不改。\n\n本轮至此停止。')
for p in [E/'README.md',E/'analysis/README.md',O/'README.md']:
 p.write_text(p.read_text(encoding='utf-8')+extra,encoding='utf-8')
(B/'README.md').write_text('# diverse_04共同端缘压板20%对照r54\n\n'+summary+' '+peaknote+'\n\n[完整结果、峰定义、图和限制](diverse_04/README.md)。协议在protocol.json，状态在status.json；原始输入与路径在diverse_04。原材料与几何数据未修改，梯度与自动检查保持暂停。\n',encoding='utf-8')
for root,paths in [(W,{'plan':'RESEARCH_PLAN.md','status':'PROJECT_OVERVIEW.md','background':'RESEARCH_BACKGROUND.md','map':'FILE_MAP.md','start':'START_HERE.md','agents':'AGENTS.md'}),(R,{'plan':'docs/RESEARCH_PLAN.md','status':'docs/RESEARCH_STATUS.md','background':'docs/RESEARCH_BACKGROUND.md','map':'docs/FILE_MAP.md','start':'README.md','agents':'AGENTS.md'})]:
 for key,rel in paths.items():
  p=root/rel;t=p.read_text(encoding='utf-8')
  if key=='plan':
   t=t.replace('当前阶段：输入检查完成，尚未提交动态计算。','本轮r54四步已完成并停止：'+summary+' '+peaknote+' 原r53梯度四步依然未全部完成。')
  elif key=='status':
   t=t.replace('# 当前状态：新增diverse_04前向对照；梯度继续暂停','# 当前状态：diverse_04新加载对照已完成；梯度继续暂停',1).replace('当前仅授权diverse_04新压板模型成对前向对照，输入准备中；','diverse_04新压板20%成对对照已结束，当前无活动计算；',1).replace('待实际求解后才给峰位及误差结论。',summary+' '+peaknote)
  elif key=='background':t=t.replace('当前仅按用户新指示执行diverse_04共同端缘压板20%成对前向对照。','diverse_04共同端缘压板20%成对前向对照已完成；其范围见当前状态。')
  elif key=='map':t=t.replace('当前只有r54成对前向任务获新授权，自动检查及梯度继续暂停。','r54成对前向已结束，当前无计算队列；自动检查及梯度继续暂停。')
  elif key=='start':t=t.replace('**当前仅新增diverse_04共同端缘压板20%前向对照；梯度与自动检查继续暂停。**','**diverse_04共同端缘压板20%前向对照已完成，当前无活动任务；梯度与自动检查继续暂停。**').replace('本轮协议、实际峰位和曲线；未完成前不提前给结论。','本轮完整结果与边界限制。'+summary+' '+peaknote)
  elif key=='agents':t=t.replace('2026-10-12新增授权：仅执行diverse_04一组20%共同中面端缘压板／XY周期／横向自由对照，准备后顺序运行Abaqus与JAX，比较总力、峰位、功、保载及模式。','2026-10-12新增diverse_04共同端缘压板20%成对前向已完成并停止。'+summary+' '+peaknote+' 当前无活动计算。')
  p.write_text(t,encoding='utf-8')
for root,rel,link in [(W,'output/README.md','r54_platen_xy_diverse04_20261012/README.md'),(R,'validation/README.md','platen_xy_diverse04_20261012_r54/README.md')]:
 p=root/rel;t=p.read_text(encoding='utf-8');t=t.replace('当前仅执行新授权','已完成新授权').replace('前向对照；梯度和每30分钟自动检查继续暂停。','前向对照，当前无计算队列；梯度和每30分钟自动检查继续暂停。');t=t.replace('用户另授权[r54','已完成[r54');t+='\n最新r54实际结果：'+summary+' '+peaknote+' [详细报告]('+link+')。\n';p.write_text(t,encoding='utf-8')
s.update(stage='complete_pair_delivered',active_jobs=[],next_queue=[],heartbeat='PAUSED; not re-enabled',gradient_r53_unchanged=True)
for p in [B/'status.json',O/'status.json']:p.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
stage=W/'work/r54_stage';dest=H/'stage';assert stage.resolve().is_relative_to(W.resolve()) and dest.resolve().is_relative_to(W.resolve()) and not dest.exists()
manifest={str(p.relative_to(stage)):sha(p) for p in stage.rglob('*') if p.is_file()}
for name,digest in manifest.items():
 src=stage/name;shutil.copy2(src,B/'tools'/name);assert sha(B/'tools'/name)==digest
shutil.move(str(stage),str(dest));assert all(sha(dest/n)==v for n,v in manifest.items())
receipt={'source_guard_count':len(guard),'protected_sources_unchanged':True,'stage_archived_sha256':manifest,'summary':summary,'peak_note':peaknote,'no_new_gradient_or_Git':True}
for p in [H/'final_receipt.json',B/'checks/final_receipt.json']:p.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(summary+' '+peaknote)

"""Summarize saved paired evidence only; never submits or retries a solve."""
from pathlib import Path
import csv,hashlib,json,shutil,sys,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
B=Path(__file__).resolve().parents[1];R=B.parents[1]
W=Path(r'C:\Users\xuehu\Documents\Codex\2026-10-01\referenced-chatgpt-conversation-this-is-an');O=W/'output/r55_platen_xy_diverse_batch_20261012'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
cases=read(B/'protocol.json')['cases'];audit=read(B/'checks/geometry_precheck.json');status=read(B/'status.json');final='--final' in sys.argv
out=B/'analysis';out.mkdir(exist_ok=True);rows=[];detailed=[]
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
fig,axes=plt.subplots(1,len(cases),figsize=(15,4.6),constrained_layout=True)
figall,axall=plt.subplots(figsize=(10.5,6.5),constrained_layout=True)
colors=['#285c8e','#c45b25','#46854b']
for i,kind in enumerate(cases):
 E=B/kind;op=O/kind;op.mkdir(exist_ok=True);cfg=read(E/'input/protocol.json') if (E/'input/protocol.json').exists() else {}
 peak=read(E/'analysis/peak_metrics.json') if (E/'analysis/peak_metrics.json').exists() else None
 m=read(E/'analysis/metrics.json') if (E/'analysis/metrics.json').exists() else {}
 cost=read(E/'analysis/costs.json') if (E/'analysis/costs.json').exists() else {}
 row={'case':kind,'paired_complete':peak['paired_complete20_and_hold'] if peak else False,
  'common_compression_percent':peak['common_compression_max']*100 if peak else None,
  'curve_RMS_percent_of_shell_peak':100*peak['primary_curve_RMS_over_shell_peak'] if peak else None,
  'work_bias_percent':100*peak['loading_work_relative_bias'] if peak else None,
  'hold_force_bias_percent':100*peak['hold_mean_relative_bias'] if peak and 'hold_mean_relative_bias' in peak else None,
  'first_prominent_peak_location_difference_pp':peak.get('first_prominent_peak_difference_percentage_points') if peak else None,
  'raw_global_peak_JAX_percent':peak['raw_global_peak']['JAX']['compression_percent'] if peak else None,
  'raw_global_peak_Abaqus_percent':peak['raw_global_peak']['Abaqus']['compression_percent'] if peak else None,
  'working_targets_pass':peak.get('three_working_targets_within10pct') if peak else None,
  'wall_minutes':cost.get('overall_minutes'),'field_relative_RMS_20pct_percent':None}
 field=read(E/'analysis/field_comparison.json') if (E/'analysis/field_comparison.json').exists() else {}
 for f in field.get('states',[]):
  if f['target_pct']==20:row['field_relative_RMS_20pct_percent']=100*f['relative_displacement_RMS']
 rows.append(row)
 for solver,folder,style in [('Abaqus','abaqus','-'),('JAX','jax','--')]:
  q=E/folder/'result.json'
  if not q.exists():continue
  res=read(q);path=res['path'];t=np.array([p['time'] for p in path]);eps=np.array([p['compression'] for p in path]);force=np.array([p['force_N'] for p in path]);mask=t<=.004+1e-12
  axes[i].plot(eps[mask]*100,force[mask],style,color='#285c8e' if solver=='Abaqus' else '#c45b25',lw=1.3,label=solver+(' complete' if res['complete'] else ' partial'))
  axall.plot(eps[mask]*100,force[mask]/100,style,color=colors[i],label=kind+' / '+solver,lw=1.35)
 axes[i].set(xlabel='Engineering compression (%)',ylabel='Total upper-platen force (N)',title=kind,xlim=(0,20));axes[i].grid(alpha=.2)
 if axes[i].lines:axes[i].legend(fontsize=8)
 else:axes[i].text(.5,.5,'No saved response yet',ha='center',transform=axes[i].transAxes)
 if peak:
  src=E/'checks/README_analysis_base.md'
  if not src.exists() and (E/'analysis/README.md').exists():shutil.copy2(E/'analysis/README.md',src)
  text=src.read_text(encoding='utf-8') if src.exists() else '# '+kind+'\n'
  health=m.get('JAX_sampled_health',{});ah=m.get('Abaqus_health',{})
  text+='\n## 实际成本、健康指标与局部限制\n\n'
  text+=f"本组成对总墙钟{cost.get('overall_minutes',float('nan')):.2f} min（包括准备／编译／检查／求解／提取／自动分析）；预算100 min，不自动重算。是否完整：JAX {peak['JAX_complete']}、Abaqus {peak['Abaqus_complete']}。\n\n"
  if health:text+=f"JAX步数{health.get('steps')}；采样主体最小实际J={health.get('minimum_required_actual_J')}；全域最小实际J={health.get('minimum_all_actual_J')}；拒绝块{health.get('rejected_blocks')}；能量／功缺口={health.get('energy_work_defect_over_max_sampled_work')}；末端KE/IE={health.get('terminal_KE_over_IE')}。负J弱域保留，不删除其能量；采样并非全过程证明。\n\n"
  if ah:text+=f"Abaqus XY位移方程误差{ah.get('XY_U_error_mm')} mm；末端KE/IE={ah.get('terminal_KE_over_IE')}；全局能量漂移／功={ah.get('energy_drift_relative_work')}。能量比是无量纲比值，跳跃附近的动能不能单独用于否决响应。\n\n"
  if field.get('states'):
   text+='|压缩量|中面位移相对RMS差|JAX全部中面越板mm|Abaqus全部中面越板mm|\n|---|---:|---:|---:|\n'
   for f in field['states']:text+=f"|{f['target_pct']}%|{f['relative_displacement_RMS']*100:.2f}%|{f['JAX_all_midsurface_outside_platens_max_mm']:.5f}|{f['Abaqus_all_midsurface_outside_platens_max_mm']:.5f}|\n"
   text+='\n位移差采用同时间JAX Q2中面插值与Abaqus输出时间插值。全部中面越板、JAX有限宽切面越板、选中端缘罚穿透是不同位置的量；后两者详见field_comparison.json与原路径。共同端缘简化不会阻挡所有其他壁点进入板面。宏观工作目标通过不能代替局部模式、有限厚度接触或梯度认证。\n'
  gp=read(E/'checks/gauss_preparation.json');text+=f"\n自身高斯积分占据体积分数{gp['weighted_occupancy_volume_fraction']:.6f}；壳中面面积×厚度给出的体积分数{gp['shell_area_times_t_volume_fraction']:.6f}，相对差{gp['volume_relative_difference']*100:+.3f}%。它是几何表达差异，不进行体积或模量拟合。\n"
  # The reused analyzer already wrote correct case-relative links; enrich both copies.
  (E/'analysis/README.md').write_text(text,encoding='utf-8');(E/'README.md').write_text(text.replace('](','](analysis/'),encoding='utf-8')
  for q in (E/'analysis').glob('*'):
   if q.is_file():shutil.copy2(q,op/q.name)
  (op/'README.md').write_text(text,encoding='utf-8')
  for q in [E/'checks/gauss_preparation.json',E/'checks/input_audit.json']:
   if q.exists():shutil.copy2(q,op/q.name)
  detailed.append({'case':kind,'metrics':peak,'health':m,'field':field,'cost':cost})
fig.suptitle('Same new loading model | 4 ms to20% +0.2 ms hold | no fitted material or geometry')
fig.savefig(out/'all_pairs_force.png');plt.close(fig)
axall.set(xlabel='Engineering compression (%) = platen travel /10 mm',ylabel='Nominal engineering compressive stress (MPa) = force /100 mm²',xlim=(0,20),title='Geometry-dependent responses under one fixed protocol');axall.grid(alpha=.2)
if axall.lines:axall.legend(ncol=2,fontsize=9)
figall.savefig(out/'all_shapes_stress.png');plt.close(figall)
write(out/'batch_metrics.json',{'cases':rows,'complete_batch_attempts':final,'details':detailed,'metrics_frozen_before_execution':True,'new_design_AD':False,'heartbeat':'PAUSED'})
with (out/'batch_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
fmt=lambda x:'—' if x is None else f'{x:+.3f}'
tab='\n'.join(f"|[{q['case']}]({q['case']}/README.md)|{'完整' if q['paired_complete'] else '未完整／待运行'}|{fmt(q['curve_RMS_percent_of_shell_peak'])}|{fmt(q['work_bias_percent'])}|{fmt(q['hold_force_bias_percent'])}|{fmt(q['first_prominent_peak_location_difference_pp'])}|{fmt(q['wall_minutes'])}|" for q in rows)
text=f'''# 三种diverse构型：共同中面端缘压板／XY周期20%对照

2026-10-12。{'三组成对尝试已结束，不自动重算或扩展。' if final else '本批次正在按队列推进，表中仅列实际保存的结果。'} 比较diverse_28、diverse_05和diverse_02，考察同一方法能否迁移到不同的周期薄壁构型。原diverse_04的结果仍单独保存在r54，本轮不重算。

|构型|20%＋保载|曲线RMS／壳峰%|加载功差%|保载均力差%|首个显著峰差／百分点|总墙钟min|
|---|---|---:|---:|---:|---:|---:|
{tab}

差值方向为JAX−Abaqus；功／保载相对差分别除以本组新壳的功／均力。RMS是两条曲线之差的均方根除以本组新壳加载峰，不能理解为每一点误差均在该百分比内。未完整路径只能评价实际共同窗口；完整20%工作目标需要两端均完成并且曲线RMS、功、保载三个量均约10%以内。局部模式和数值健康必须另读。

![三组成对总力](all_pairs_force.png)

![跨构型名义应力](all_shapes_stress.png)

名义应力=上压板总压缩反力／100 mm²，工程压缩量=上板位移绝对值／10 mm；不是壁材料局部应力或局部应变。每张单独报告还包含原曲线、固定显著峰、横向伸缩、能量与实际J，以及可提取的10%／20%变形。全局最大值落在20%端点时，不能称为已捕捉屈曲峰；没有合格内部峰则不强行核对峰位。

## 固定条件与选样依据

三组均L10 mm／t0.5 mm、Neo-Hookean E10 MPa／ν0.3／密度1e−9 tonne/mm³；JAX N32 HEX27／27点／HRZ、η1e−4、占据过渡宽0.05 mm、objective_void/C²，原虚域和实际J保护不变。Abaqus同一三角中面S3R壳，厚度0.5 mm、截面5点；从G/r54逐字复用材料、压板、接触、加载和输出控制。无质量缩放，两端物理加载4 ms到20%、保载0.2 ms，横向宏观伸缩自由、宏观剪切零，仅XY周期。

几何来自用户原模型。diverse_28和05为Fine/T0p02/MS9；02为r3_smoke/batch_001，原网格的周期性与拓扑经本轮重新检查。各自新占据场在其实际HEX27积分点生成，壳与背景使用同一来源中面及物理厚度，不复用其他构型的占据。旧曲线仅用于选样：其峰约8.14%、4.39%、1.87%，但材料／加载／质量等不一致，不能用来预言或验收新曲线。

共同端缘简化只在初始中面上下端缘施加载荷：JAX占据切面面积权重聚合到对应壳端缘位置、Q2插值；壳端缘节点接触厚度为零，截面厚度仍保留。这减少端部定义干扰，但不是完整有限厚度壁面接触，也没有自接触。壳／实体单元运动学、罚接触和惯性离散仍不同。约10%只是限定研究工作目标，壳不是精确真值；前向接近不能证明梯度或任意构型均可靠。

## 输入、证据和执行边界

正式批次`/home/xuehu/projects/tpms_jax/validation/platen_xy_diverse_batch_20261012_r55`。sources分构型保存原中面与实际高斯场；各构型独立保存input/protocol.json、abaqus/platen_20pct.inp、abaqus/result.json、jax/result.json和analysis；原生ODB在`E:/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus/r55_<kind>_midsurface_end_20pct_20261012/platen.odb`。Windows各构型阅读文件夹含参数、INP、曲线CSV、图和报告。

每组100 min总预算：JAX65、Abaqus25、其余10。顺序、各端一次尝试，数值失败或超时保留有效部分与原因，不根据差异大小提前停算，不自动变参数重试。本轮不做新AD／FD、训练或Git，heartbeat保持PAUSED。受保护旧科学文件见checks/source_guard_before/after.json，原r53失败与门槛不改。
'''
if final:
 passed=[q['case'] for q in rows if q['working_targets_pass']];complete=[q['case'] for q in rows if q['paired_complete']]
 text+=f"\n## 本轮可作的结论\n\n完整成对路径：{', '.join(complete) or '无'}。满足三个限定宏观工作目标：{', '.join(passed) or '无'}。未通过或未完整的样例均保留，不用其余样例的成功覆盖。对峰位须结合各自合格内部峰及变形；宏观结果不能代替局部模式的判断。\n"
 guard=read(B/'checks/source_guard_before.json');changed=[p for p,h in guard.items() if sha(R/p)!=h]
 originalchanged=[p for a in audit.values() for p,h in a['original_readonly_source_sha256'].items() if sha(Path(p))!=h]
 write(B/'checks/source_guard_after.json',{'protected_files':len(guard),'changed':changed,'original_user_files_changed':originalchanged});assert not changed and not originalchanged
 notice='2026-10-12新增授权：r55正在准备／执行diverse_28、diverse_05、diverse_02三组20%共同端缘压板对照；每组100 min，一次尝试。唯一执行队列见RESEARCH_PLAN.md；梯度与heartbeat仍暂停。旧“无活动／已停止”描述仅指r54结束时。\n\n'
 new=f"2026-10-12 r55三组成对尝试已结束：完整路径{', '.join(complete) or '无'}；三个宏观工作目标通过{', '.join(passed) or '无'}。当前无新计算队列，梯度和heartbeat继续暂停。详见[本轮报告](output/r55_platen_xy_diverse_batch_20261012/README.md)。\n\n"
 for n,rn in [('PROJECT_OVERVIEW.md','docs/RESEARCH_STATUS.md'),('FILE_MAP.md','docs/FILE_MAP.md'),('START_HERE.md','README.md'),('AGENTS.md','AGENTS.md')]:
  q=W/n;s=q.read_text(encoding='utf-8');s=s.replace(notice,new);q.write_text(s,encoding='utf-8');(R/rn).write_text(s,encoding='utf-8')
 plan=W/'RESEARCH_PLAN.md';s=plan.read_text(encoding='utf-8');s+='\n## r55实际执行收口\n\n'+new+'每组成本与未通过范围均见报告，本轮停止，不自动恢复原梯度规划。\n';plan.write_text(s,encoding='utf-8');shutil.copy2(plan,R/'docs/RESEARCH_PLAN.md')
 for q in [W/'output/README.md',R/'validation/README.md']:
  with q.open('a',encoding='utf-8') as f:f.write('\n- r55：三种diverse共同端缘压板20%对照，见'+('r55_platen_xy_diverse_batch_20261012/README.md' if q.parent.name=='output' else B.name+'/README.md')+'；失败保留，梯度暂停。\n')
 write(B/'checks/final_receipt.json',{'delivered_to_reading_folder':str(O),'no_new_solve_in_summary':True,'protected_sources_unchanged':True,'batch_rows':rows,'time_unix':time.time()})
(B/'README.md').write_text(text,encoding='utf-8');(O/'README.md').write_text(text,encoding='utf-8')
for q in out.glob('*'):
 if q.is_file():shutil.copy2(q,O/q.name)
print(json.dumps({'final':final,'cases':rows},ensure_ascii=False))

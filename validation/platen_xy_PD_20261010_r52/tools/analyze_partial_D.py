"""Analyze the original D attempt with a failed shell reference; no FEM calls/retry."""
from pathlib import Path
import csv,hashlib,json,re,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
B=Path(__file__).resolve().parents[1];R=B.parents[1];E=B/'D';D=R/'validation/platen_xy_G_20261010_r51'
sys.path[:0]=[str(D/'tools'),str(R)]
O=E/'analysis';O.mkdir(exist_ok=True)
def load(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def arr(res,key):return np.asarray([x.get(key) for x in res['path']],float)
def integrate(y,x):return float(np.sum((y[1:]+y[:-1])*.5*np.diff(x)))
cfg=load(E/'input/protocol.json');T=cfg['load_time_s'];L=cfg['L_mm']
j=load(E/'jax/result.json') if (E/'jax/result.json').exists() else {'complete':False,'status':'interrupted_saved_path','path':load(E/'jax/path_latest.json')}
a=load(E/'abaqus/result_raw.json');jt=arr(j,'time');at=arr(a,'time')
def loading(res):
    t=arr(res,'time');mask=t<T-1e-13;end=min(t[-1],T)
    eps=np.r_[arr(res,'compression')[mask],np.interp(end,t,arr(res,'compression'))]
    force=np.r_[arr(res,'force_N')[mask],np.interp(end,t,arr(res,'force_N'))]
    if end>=T-1e-13 and abs(eps[-1]-.2)<1e-8:eps[-1]=.2
    ue,rev=np.unique(eps[::-1],return_index=True);return ue,force[len(eps)-1-rev]
ej,fj=loading(j);ea,fa=loading(a);fixed=4.123236290287
metrics={'scope':cfg['stage'],'JAX_status':j['status'],'JAX_complete':j['complete'],'Abaqus_complete':False,
    'paired_full20_and_hold':False,'full20_acceptance':'not_evaluable_reference_failed',
    'primary_shell_full20_peak_N':None,'curve_RMS_N':None,'curve_RMS_over_new_shell_peak':None,
    'common_loading_work_N_mm':{'JAX':None,'Abaqus':None,'relative_bias':None},
    'three_scalar_targets_within_10pct':False,'new_design_AD':False,'physical_finite_width_contact_validated':False,
    'Abaqus_failure':{'element':16329,'history_last_time_s':float(at[-1]),'history_last_compression':float(ea[-1]),
        'fatal_field_time_s':float(a['frame_times_s'][-1]),'fatal_STA_time_s_rounded':.003090,
        'fatal_STA_dt_s':1e-14,'fatal_STA_KE_N_mm':6.154e8,'fatal_STA_ETOTAL_N_mm':6.154e8,
        'fatal_increment_not_in_ODB_history':True,'reference_is_not_full20':True},
    'partial_diagnostic_windows':[],'sample_force':[]}
for hi in [.02,.10,.15]:
    if min(ej[-1],ea[-1])<hi-1e-9:continue
    grid=np.linspace(0,hi,1001);J=np.interp(grid,ej,fj);A=np.interp(grid,ea,fa)
    den=float(np.max(abs(A)));rms=float(np.sqrt(np.mean((J-A)**2)));wj=integrate(J,grid*L);wa=integrate(A,grid*L)
    metrics['partial_diagnostic_windows'].append({'compression':[0,hi],'RMS_N':rms,'own_partial_shell_peak_N':den,
        'RMS_over_own_partial_shell_peak':rms/den,'RMS_over_fixed_G_peak_context_only':rms/fixed,
        'loading_work_JAX_N_mm':wj,'loading_work_Abaqus_N_mm':wa,'loading_work_relative_bias':wj/wa-1,
        'full20_acceptance':False,'window_fixed_before_D_JAX_submission':True})
emax=min(.15,ej[-1],ea[-1]);metrics['compared_compression_max']=float(emax)
grid=np.linspace(0,emax,1001);J=np.interp(grid,ej,fj);A=np.interp(grid,ea,fa)
with (O/'force_comparison.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['engineering_compression','JAX_upper_force_N','Abaqus_upper_force_N','diagnostic_difference_over_fixed_G_peak']);w.writerows(zip(grid,J,A,(J-A)/fixed))
for val in [.02,.05,.10,.15]:
    if val>emax+1e-9:continue
    vj=float(np.interp(val,ej,fj));va=float(np.interp(val,ea,fa));metrics['sample_force'].append({'compression':val,'JAX_N':vj,'Abaqus_N':va,'relative_bias':vj/va-1})
for label,res in [('JAX',j),('Abaqus_raw_failed',a)]:
    keys=list(res['path'][0])
    with (O/(label+'_path.csv')).open('w',newline='') as f:
        w=csv.writer(f);w.writerow(keys);w.writerows([[json.dumps(x.get(k)) if isinstance(x.get(k),(list,dict)) else x.get(k) for k in keys] for x in res['path']])
energy=arr(j,'energy_N_mm')+arr(j,'contact_energy_N_mm')+arr(j,'KE_N_mm');force=arr(j,'force_N');travel=arr(j,'compression')*L
work=np.r_[0,np.cumsum((force[1:]+force[:-1])*.5*np.diff(travel))]
momentum=arr(j,'momentum_z_N_s');net=arr(j,'bottom_force_N')-force;impulse=np.r_[0,np.cumsum((net[1:]+net[:-1])*.5*np.diff(jt))]
metrics['JAX_sampled_health']={'minimum_required_actual_J':float(np.min(arr(j,'required_positive_J_min'))),
    'minimum_all_actual_J':float(np.min(arr(j,'all_J_min'))),'max_negative_void_J_samples':int(np.max(arr(j,'negative_void_J_points'))),
    'energy_work_defect_over_max_sampled_work':float(np.max(abs(energy-energy[0]-work))/max(np.max(abs(work)),1e-30)),
    'impulse_defect_over_max_momentum':float(np.max(abs(momentum-momentum[0]-impulse))/max(np.max(abs(momentum)),1e-30)),
    'terminal_KE_over_IE':float(arr(j,'KE_over_IE')[-1]),'steps':j.get('steps'),'wall_seconds':j.get('wall_seconds'),
    'rejected_blocks':len(j.get('rejections',[])),'initial_dt_s':j.get('initial_dt_s'),'terminal_dt_s':j.get('terminal_dt_s'),
    'validation_scope':'Accepted block samples and selected tangent checks, not every-step or a design total gradient.'}
plt.rcParams.update({'font.size':10,'figure.dpi':160,'savefig.dpi':180,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
for ax in axs[0]:
    ax.plot(ea*100,fa,color='#285c8e',label='Abaqus raw ODB history (failed run)');ax.plot(ej*100,fj,color='#c45b25',ls='--',label='JAX HEX27 / Verlet')
    ax.axvline(15,color='gray',ls=':',label='Paired diagnostic limit: 15%');ax.grid(alpha=.2);ax.legend(fontsize=8)
    ax.set(xlabel='Engineering compression (%)',ylabel='Upper-platen compressive force (N)')
axs[0,0].set(xlim=(0,20),title='D: reference fails near 18.4%; no full20 acceptance');axs[0,1].set(xlim=(0,5),title='Initial segment, original 4 ms ramp')
for key,ls in [('ax_mm','-'),('ay_mm','--')]:
    axs[1,0].plot(jt*1000,arr(j,key),ls,color='#c45b25',label='JAX '+key[:2]);axs[1,0].plot(at*1000,arr(a,key),ls,color='#285c8e',label='Abaqus '+key[:2])
axs[1,0].set(xlabel='Physical time (ms)',ylabel='Macro displacement jump (mm)',title='Free transverse response; failed history preserved')
axs[1,1].plot(jt*1000,force,color='#c45b25',label='JAX upper force');axs[1,1].plot(at*1000,arr(a,'force_N'),color='#285c8e',label='Abaqus upper force');axs[1,1].axvline(4,color='gray',ls=':')
axs[1,1].set(xlabel='Physical time (ms)',ylabel='Compressive force (N)',title='Only JAX can reach the planned hold in this attempt')
for ax in axs[1]:ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('D | common midsurface-end platen / free XY | L10 mm, t0.5 mm, NH E10 MPa')
fig.savefig(O/'force_and_lateral.png');plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
for res,t,label,ie,ke in [(j,jt,'JAX','energy_N_mm','KE_N_mm'),(a,at,'Abaqus','ALLIE','ALLKE')]:
    axs[0,0].plot(t*1000,arr(res,ie),label=label+' internal');axs[0,0].plot(t*1000,arr(res,ke),'--',label=label+' kinetic')
axs[0,0].set(ylabel='Energy (N mm)',title='ODB history stops before the fatal energy explosion')
axs[0,1].plot(jt*1000,energy-energy[0],label='JAX internal + contact + kinetic');axs[0,1].plot(jt*1000,work,'--',label='JAX sampled external work');axs[0,1].set(ylabel='Energy (N mm)',title='JAX sampled work balance')
axs[1,0].plot(jt*1000,arr(j,'required_positive_J_min'),label='Required actual J min');axs[1,0].plot(jt*1000,arr(j,'all_J_min'),label='All-domain J min');axs[1,0].set(ylabel='Jacobian determinant',title='Virtual-domain flips retained')
sta=[]
for line in (E/'abaqus/platen.sta').read_text(errors='replace').splitlines():
    cols=line.split()
    if len(cols)==8 and cols[0].isdigit() and ':' in cols[3]:
        try:sta.append([float(cols[1]),float(cols[4]),float(cols[6]),float(cols[7])])
        except ValueError:pass
sta=np.asarray(sta);assert sta.shape[1]==4 and sta[-1,1]==1e-14 and sta[-1,2]==6.154e8
np.savetxt(O/'Abaqus_STA_energy_and_dt.csv',sta,delimiter=',',header='time_s,stable_dt_s,kinetic_energy_N_mm,total_energy_N_mm',comments='')
axs[1,1].plot(sta[:,0]*1000,sta[:,2],label='STA kinetic');axs[1,1].plot(sta[:,0]*1000,sta[:,3],'--',label='STA total energy');axs[1,1].set_yscale('symlog',linthresh=.01)
axs[1,1].set(ylabel='Energy (N mm), symlog',title='Fatal increment retained from native STA (6.154e8)')
for ax in axs.flat:ax.set_xlabel('Physical time (ms)');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.savefig(O/'energy_and_J.png');plt.close(fig)
# Reuse the frozen G Q2 field interpolation and plotting, restricted to saved
# 5/10/15% states. Never extrapolate the failed shell to 20%.
source=D/'experiments/midsurface_rim_20pct_20261010/tools/analyze_rim20.py';original=source.read_text();code=original[original.index('# Field comparison at the same time,'):]
subs=[("R/'validation/canonical_tpms_20261009_r50/G/surface_shell_quality.npz'",f"Path({cfg['source_shell_surface']!r})"),
    ("D/'input/cut_face_contact.npz'","E/'input/cut_face_contact.npz'"),('for target in [5,10,15,20]:','for target in [5,10,15]:'),
    ("# Reuse the exact periodic equivalence classes; no model rebuild.","assert ts <= ft[-2], 'No extrapolation beyond the final regular shell frame'\n    # Reuse the exact periodic equivalence classes; no model rebuild."),
    ('if x[0] in [10,20]','if x[0] in [10,15]'),('deformation_10_20pct.png','deformation_10_15pct.png')]
for old,new in subs:assert code.count(old)==1,(old,code.count(old));code=code.replace(old,new)
write(E/'checks/partial_postprocess_recipe.json',{'source':str(source),'source_sha256':hashlib.sha256(original.encode()).hexdigest(),'substitutions':subs,'fixed_windows':[.02,.10,.15],'full20_denominator':None,'new_FEM_calls':False})
exec(compile(code,str(source),'exec'),globals())
jax20=''
if (E/'jax/state_20pct.npz').exists():
    with np.load(E/'jax/state_20pct.npz') as z:q=z['q'];macro=z['macro'];ts=float(z['time'])
    nodal=q[np.asarray(classes)];u=np.einsum('cn,cnj->cj',sh,nodal[ids])*L;u[:,:2]+=b*macro*L;pos=vertices*L+u
    vmax=float(np.max(np.linalg.norm(u,axis=1)));colors=plt.cm.viridis(np.mean(np.linalg.norm(u,axis=1)[tri],axis=1)/max(vmax,1e-12))
    fig=plt.figure(figsize=(7,6));ax=fig.add_subplot(111,projection='3d');ax.add_collection3d(Poly3DCollection(pos[tri],facecolors=colors,edgecolors='none'))
    ax.set(xlim=(-1,12),ylim=(-1,12),zlim=(-1,11),xlabel='X (mm)',ylabel='Y (mm)',zlabel='Z (mm)',title='D JAX near 20% only | actual displacement scale 1\nNo Abaqus 20% reference available')
    ax.set_box_aspect((1,1,1));ax.view_init(20,-60);fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0,vmax),cmap='viridis'),ax=ax,shrink=.65,pad=.1,label='|U| (mm)')
    fig.savefig(O/'JAX_deformation_20pct_only.png',bbox_inches='tight');plt.close(fig)
    plate=L*(1-cfg['target_compression']);cz=cap['points'][:,2]*L+np.einsum('cn,cn->c',cap['shape'],nodal[cap['node_ids'],2])*L
    pen=np.maximum(np.where(cap['side']==0,-cz,cz-plate),0)
    metrics['JAX_20pct_alone']={'time_s':ts,'finite_width_cut_face_max_overrun_phi_ge_0p5_mm':float(np.max(pen[cap['phi']>=.5])),'not_a_paired_field_comparison':True}
    jax20='\n![JAX单独20%实际尺度变形；没有壳20%参考](analysis/JAX_deformation_20pct_only.png)\n'
windowrows='\n'.join(f"| 0–{x['compression'][1]*100:.0f}% | {x['own_partial_shell_peak_N']:.5f} | {100*x['RMS_over_own_partial_shell_peak']:.3f}% | {100*x['loading_work_relative_bias']:+.3f}% |" for x in metrics['partial_diagnostic_windows'])
samples='\n'.join(f"| {x['compression']*100:.0f}% | {x['JAX_N']:.5f} | {x['Abaqus_N']:.5f} | {100*x['relative_bias']:+.3f}% |" for x in metrics['sample_force'])
fieldrows='\n'.join(f"| {x['target_pct']}% | {x['displacement_RMS_mm']:.5f} | {100*x['relative_displacement_RMS']:.2f}% |" for x in metrics['field_comparison'])
text=f'''# D共同中面端部压板：参考路径失败，保留完整证据

**Abaqus在约18.4%压缩时因壳单元16329过度畸变停止；JAX状态为{j['status']}，完整20%与保载完成={j['complete']}。本轮D的完整20%成对验收不可评价。** 没有重算Abaqus，没有修改材料／网格／加载参数。D/JAX是原计划尚未提交的第四条路径，沿用原D总预算时钟。

L10 mm、t0.5 mm、NH E10 MPa/ν0.3/密度1e−9 tonne/mm³；N32 HEX27/27点、HRZ、连续占据η1e−4、界面0.05 mm、objective_void/C²。Abaqus S3R壳截面5点。4 ms平滑下压20%及0.2 ms保载、XY周期与自由横向伸缩。每端254中面节点（含2相切点），JAX自身占据参考面积每端26.750596 mm²聚合到同一初始位置，kn320 MPa/mm。没有Z周期、全壁面／自接触或新增设计总梯度。

![原始加载历史与JAX路径](analysis/force_and_lateral.png)

在D/JAX提交前固定0–2%、0–10%、0–15%三个诊断窗口。下表RMS以**各窗口自己的Abaqus峰力**为分母，不能与G/P全20%RMS直接排名，也不能代替D全20%验收。超过15%的原ODB历史仍在图中展示，但不以失效前末段作为完整可靠参考。主指标的完整20%分母、功与保载差均留空，避免用失效尖峰稀释误差。

| 预先固定诊断窗口 | 本窗口壳峰 N | 曲线RMS／窗口峰 | 窗口加载功偏差 |
| --- | ---: | ---: | ---: |
{windowrows}

| 压缩量 | JAX上板总反力 N | Abaqus上板总反力 N | 相对偏差 |
| --- | ---: | ---: | ---: |
{samples}

初段2%、5%、10%的总力差约−2.03%、−2.24%、−1.29%，但15%时已增至约+9.51%。**响应分歧先于壳最终的数值失败，不能把全部中后段差异归咎于致命增量。** 累积窗口RMS可能被接近的初段稀释，应同时看15%点差、曲线形状和位移模式。现有证据不足以单独归因于JAX内力、壳网格、实体／壳运动学、接触离散或耗散；本轮不通过调参拟合这段曲线。

![能量、实际J及原生STA致命增量](analysis/energy_and_J.png)

ODB历史最后为{at[-1]*1000:.6f} ms／{ea[-1]*100:.4f}%，未保存致命增量的历史能量；最后特殊位移场约3.08961 ms。原生STA在3.090 ms列出稳定步长1e−14 s、动能及总能量6.154e8 Nmm，明确未完成。ODB、原始历史／位移数组及STA均保留，不能仅凭ODB历史有限而忽略最终数值失败。这个失败尚不能归因于某一个物理因素，也不能作为JAX不准确的证明。

![仅10%和15%同时间变形对照](analysis/deformation_10_15pct.png)

{jax20}

| 保存态附近压缩量 | 同中面节点位移差RMS mm | 相对壳位移RMS |
| --- | ---: | ---: |
{fieldrows}

位移差未作刚体对齐或面积加权。D的10%／15%场差约13.05%／26.70%，不能由小累计力RMS推出局部变形准确。完整接触与自接触仍未验证。

变形采用原壳节点位置的JAX Q2插值，并在相邻Abaqus正常输出帧之间按时间插值；没有刚体对齐、曲线缩放或位移放大，也没有将Abaqus场外推到20%。实际F/J不夹断、低占据负J能量保留；有限采样与局部稳定步长JVP不等于全过程稳定、自接触或设计总梯度验证。

正式输入、原路径、有效态、失败证据与启动收据分别保存在本目录input、jax、abaqus、checks；总成本另见analysis/costs.json。当前停止追加求解，后续应先分析D失效单元位置与变形模式，再决定是否另立壳参考修复方案。
'''
(E/'README.md').write_text(text,encoding='utf-8')
write(O/'metrics.json',metrics)
print(json.dumps({'JAX_complete':j['complete'],'Abaqus_complete':False,'partial_diagnostic_windows':metrics['partial_diagnostic_windows']},indent=2))

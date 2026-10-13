"""Read r50 outcomes; common-window comparisons, no new FEM or AD."""
from pathlib import Path
import sys,json,csv,hashlib
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
R=Path('/home/xuehu/projects/tpms_jax');D=R/'validation/canonical_tpms_20261009_r50'
sys.path.insert(0,str(R))
T=.004;L=10.
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
COLORS={'JAX':'#235c91','Abaqus':'#a75030'}
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def table(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def loading(rows):
    a=np.array([r['compression'] for r in rows]);t=np.array([r['time'] for r in rows]);force=-np.array([r['Fz_N'] for r in rows])
    # Interpolate the force at the exact loading/hold boundary if history brackets it.
    boundary=float(np.interp(T,t,force)) if len(t)>1 and t[0]<=T<t[-1] else None
    mask=t<=T+1e-12;a,t,force=a[mask],t[mask],force[mask]
    if boundary is not None and (not len(t) or t[-1]<T-1e-12):
        a=np.r_[a,.2];t=np.r_[t,T];force=np.r_[force,boundary]
    if len(a) and a[0]>1e-12:a=np.r_[0.,a];t=np.r_[0.,t];force=np.r_[0.,force]
    unique=np.r_[True,np.diff(a)>1e-13];return a[unique],force[unique],t[unique]
def at(a,f,x):return float(np.interp(x,a,f)) if len(a)>1 and x<=a[-1]+1e-9 else None
def history_time_mean(rows,a=.0042,b=.0044):
    t=np.array([r['time'] for r in rows]);force=-np.array([r['Fz_N'] for r in rows])
    if len(t)<2 or t[0]>a or t[-1]<b-1e-10:return None
    tt=np.r_[a,t[(t>a)&(t<b)],b]
    return float(np.trapezoid(np.interp(tt,t,force),tt)/(b-a))
def fmt(x,n=2):return '—' if x is None else f'{x:.{n}f}'
def peak(a,f):
    i=int(np.argmax(f));return {'compression_pct':float(a[i]*100),'force_N':float(f[i]),
        'observed_interior_peak':bool(i<len(a)-2 and np.min(f[i+1:])<.99*f[i]),
        'note':'global maximum over recorded loading; rebound maximum can differ from the first load drop; endpoint maximum is not an instability peak'}

all_metrics=[];point_rows=[];overview=[]
ready=[k for k in ['G','P','D'] if (D/k/'abaqus/shell_result.json').exists() and ((D/k/'jax/result.json').exists() or (D/k/'jax/failure.json').exists())]
if not ready:raise SystemExit('No completed pair available yet')
for kind in ready:
    c=D/kind;ji=read(c/'jax_result.json');si=read(c/'initial_shell/shell_result.json')
    jfile=c/'jax/result.json';failure=c/'jax/failure.json'
    j=read(jfile) if jfile.exists() else read(failure)
    jr=j.get('path',j.get('accepted_path',[]));s=read(c/'abaqus/shell_result.json');sr=s['force_path']
    ja,jf,jt=loading(jr);sa,sf,st=loading(sr)
    if min(len(ja),len(sa))<2:raise ValueError('Insufficient dynamic comparison samples '+kind)
    end=float(min(ja[-1],sa[-1]));xx=np.linspace(0,end,2001)
    fj=np.interp(xx,ja,jf);fs=np.interp(xx,sa,sf);delta=fj-fs
    shell_peak=float(np.max(abs(sf)));shell_complete=bool(s['checks']['complete'])
    rms=float(np.sqrt(np.trapezoid(delta**2,xx)/end));wj=float(np.trapezoid(fj,xx)*L);ws=float(np.trapezoid(fs,xx)*L)
    jrec=read(c/(kind+'_dynamic_jax_receipt.json'));srec=read(c/'abaqus/receipt.json')
    geometric=read(c/'geometry_audit.json')
    quality=read(c/'shell_quality_adopted.json') if (c/'shell_quality_adopted.json').exists() else None
    shellvolume=quality['shell_nominal_volume_mm3'] if quality else geometric['nominal_shell_volume_mm3']
    with np.load(c/'gauss_field.npz') as cache:
        phi=cache['rho'];weights=cache['JxW'];v=cache.get('surface_vertices',None)
        volume=float(np.sum(phi*weights)*1000);weighted=float(np.sum((1e-4+(1-1e-4)*phi)*weights)*1000)
    last=jr[-1];required=[r['required_positive_J_min'] for r in jr if r.get('required_positive_J_min') is not None]
    final_time=float(last['time']);complete_j=bool(jfile.exists() and final_time>=1.1*T*(1-1e-9))
    hold=[r['Fz_N'] for r in jr if r['time']>=T+.0002]
    holdj=history_time_mean(jr);holds=history_time_mean(sr)
    raw_holdj=-float(np.mean(hold)) if hold else None
    raw_holds=-s['hold_mean_Fz_N'] if s.get('hold_mean_Fz_N') is not None else None
    jload=[r for r in jr if r['time']<=T+1e-12];loads_end=jload[-1]
    full_energy_work=float(np.trapezoid(jf,ja)*L)
    ledger=float((full_energy_work-loads_end['energy_N_mm']-loads_end['KE_N_mm'])/max(abs(full_energy_work),1e-30))
    met={'kind':kind,'initial_K_JAX_N_per_mm':ji['stiffness_N_per_mm'],'initial_K_shell_N_per_mm':si['stiffness_N_per_mm'],
        'initial_K_bias_pct':100*(ji['stiffness_N_per_mm']/si['stiffness_N_per_mm']-1),
        'JAX_complete20_and_hold':complete_j,'Abaqus_complete20_and_hold':shell_complete,
        'JAX_max_compression_pct':float(ja[-1]*100),'Abaqus_max_compression_pct':float(sa[-1]*100),
        'common_window_end_pct':end*100,'common_curve_RMS_N':rms,
        'RMS_pct_matching_shell_observed_loading_peak':100*rms/shell_peak,
        'matching_shell_denominator_N':shell_peak,'matching_shell_peak_full_path':shell_complete,
        'RMS_pct_legacy_fixed_2p258914N':100*rms/2.258914,
        'common_work_JAX_N_mm':wj,'common_work_shell_N_mm':ws,'common_work_bias_pct':100*(wj/ws-1),
        'observed_JAX_maximum':peak(ja,jf),'observed_shell_maximum':peak(sa,sf),
        'hold_mean_JAX_N':holdj,'hold_mean_shell_N':holds,
        'hold_mean_definition':'time-weighted trapezoid mean over fixed 4.2 to 4.4 ms, saved force histories; not a steady-state certification',
        'hold_raw_sample_mean_JAX_N':raw_holdj,'hold_raw_sample_mean_shell_N':raw_holds,
        'hold_mean_bias_pct':100*(holdj/holds-1) if holdj is not None and holds is not None else None,
        'jax_wall_minutes':jrec['wall_seconds']/60,'abaqus_wall_minutes':(srec['wall_seconds']+srec.get('extraction_seconds',0))/60,
        'Abaqus_extraction_seconds':srec.get('extraction_seconds'),
        'Abaqus_current_launch_solver_minutes':srec.get('current_launch_wall_seconds',srec['wall_seconds'])/60,
        'Abaqus_prior_mesh_startup_reserved_seconds':srec.get('prior_aborted_launch_budget_reservation_seconds',0),
        'shell_quality_reconnected':quality is not None,'shell_quality':quality,
        'JAX_stop_type':'completed' if complete_j else j.get('type','external_budget_stop'),
        'JAX_stop_message':j.get('message'),'JAX_rejected_blocks':len(j.get('rejected_blocks',[])),
        'JAX_budget_stop':jrec['budget_stop'] or j.get('type')=='TimeoutError' or (c/'budget_interrupt.json').exists(),
        'JAX_min_sampled_required_positive_J':min(required),
        'JAX_min_sampled_all_J':min(r['J_min'] for r in jr),
        'JAX_max_negative_J_points':max(r['negative_J_points'] for r in jr),
        'JAX_terminal_KE_over_U':last['KE_over_U'],
        'JAX_loading_max_KE_over_U_after1pct':max([r['KE_over_U'] for r in jload if r['compression']>=.01],default=0.),
        'JAX_loading_energy_ledger_relative_work':ledger,
        'shell_checks':s['checks'],'shell_core_valid':all(s['checks'][k] for k in ['finite','physical_node_count','XYZ_periodic_mm_le_1e-6','XYZ_rotation_rad_le_1e-6']),
        'occupancy_volume_mm3':volume,'stiffness_mass_weighted_volume_mm3':weighted,
        'shell_area_times_thickness_volume_mm3':shellvolume,
        'JAX_total_mass_tonne':weighted*1e-9,'shell_nominal_mass_tonne':shellvolume*1e-9,
        'geometry_triangles':geometric['triangles'],
        'translation_gauge':{'JAX':'q=0 at background periodic class (0,0,0)',
            'Abaqus':'u=0 at first non-slave physical shell node',
            'Abaqus_pin_xyz_mm':{'G':[0.,0.,5.],'P':[0.,2.5,5.],'D':[0.,0.,2.5]}[kind],
            'same_discrete_dynamics':False,'isolated_dynamic_effect_tested':False},'no_contact':True,'new_design_AD':False,
        'scope':'Common-window model comparison. Approx10pct exploratory target, not physical truth, time convergence, contact or full-path gradient certification.'}
    for a in [.05,.10,.15,.20]:
        fjx=at(ja,jf,a);fsx=at(sa,sf,a)
        point_rows.append({'kind':kind,'compression_pct':a*100,'JAX_force_N':fjx,'shell_force_N':fsx,
            'signed_bias_pct_shell_point':100*(fjx/fsx-1) if fjx is not None and fsx is not None and abs(fsx)>1e-12 else None,
            'signed_error_N':fjx-fsx if fjx is not None and fsx is not None else None})
    met['point_forces']=[x for x in point_rows if x['kind']==kind]
    jp,sp=met['observed_JAX_maximum'],met['observed_shell_maximum']
    met['peak_force_bias_pct']=100*(jp['force_N']/sp['force_N']-1)
    met['both_observed_interior_peaks']=jp['observed_interior_peak'] and sp['observed_interior_peak']
    met['peak_position_difference_percentage_points']=jp['compression_pct']-sp['compression_pct'] if met['both_observed_interior_peaks'] else None
    met['Abaqus_terminal_KE_over_IE']=s.get('KE_fraction_of_IE_terminal')
    met['Abaqus_global_energy_drift_relative_work']=s.get('global_energy_drift_relative_work')
    met['Abaqus_artificial_energy_relative_max_work']=s.get('artificial_energy_relative_max_work')
    all_metrics.append(met)
    overview.append({k:met[k] for k in ['kind','initial_K_JAX_N_per_mm','initial_K_shell_N_per_mm','initial_K_bias_pct',
        'JAX_complete20_and_hold','Abaqus_complete20_and_hold','common_window_end_pct','common_curve_RMS_N',
        'RMS_pct_matching_shell_observed_loading_peak','common_work_bias_pct','hold_mean_bias_pct',
        'jax_wall_minutes','abaqus_wall_minutes','JAX_stop_type']})
    fig,axes=plt.subplots(1,2,figsize=(11.4,4.2),layout='constrained')
    for ax,limit in zip(axes,[20,5]):
        ax.plot(sa*100,sf,label='Abaqus S3R / Explicit',color=COLORS['Abaqus'],lw=1.8)
        ax.plot(ja*100,jf,label='JAX HEX27 N32 / 27 GP',color=COLORS['JAX'],lw=1.7)
        ax.set(xlabel='Compression (%)',ylabel='Compressive force (N)',xlim=(0,limit),ylim=(0,None));ax.grid(alpha=.2)
    axes[0].set_title(f'Common-window RMS / shell peak = {100*rms/shell_peak:.2f}%')
    axes[1].set_title(f'Zero-inertia initial stiffness bias = {met["initial_K_bias_pct"]:+.2f}%')
    axes[0].legend(fontsize=9)
    fig.suptitle(f'{kind} | L=10 mm, t=0.50 mm, NH E=10 MPa, XYZ periodic\nBoth T=4 ms; JAX continuous occupancy, eta=1e-4, interface=0.05 mm; no contact',fontsize=11)
    for ext in ['png','svg']:fig.savefig(c/('response.'+ext),dpi=180)
    plt.close(fig)
    # Both complete and partial fields are shown with their own strain labels.
    statefile=c/'jax/field.npz'
    if not statefile.exists():statefile=c/'jax/last_valid_field.npz'
    if statefile.exists() and (c/'abaqus/shell_field.npz').exists():
        import basix
        from jax_fem.basis import get_elements
        with np.load(c/'surface.npz') as geo:v,f=geo['surface_vertices'],geo['surface_triangles']
        fshell=f
        if quality:
            with np.load(c/'surface_shell_quality.npz') as geo:fshell=geo['surface_triangles']
        with np.load(c/'linear_state.npz') as cache:classids=cache['class_ids']
        with np.load(statefile) as cache:q=cache['q'];time=float(cache['time'])
        x=np.clip(time/T,0,1);compression=.2*(10*x**3-15*x**4+6*x**5)
        family,cell,_,_,degree,order=get_elements('HEX27');el=basix.create_element(family,cell,degree)
        local=np.rint(2*el.points[order]).astype(int);origins=2*np.indices((32,)*3).reshape(3,-1).T
        idx=origins[:,None,:]+local[None,:,:];cells=(idx[:,:,0]*65+idx[:,:,1])*65+idx[:,:,2]
        ci=np.minimum(np.floor(v*32).astype(int),31);ref=v*32-ci
        tab=el.tabulate(0,ref)[0,:,:,0][:,order];index=(ci[:,0]*32+ci[:,1])*32+ci[:,2]
        ju=np.einsum('vn,vni->vi',tab,q[classids[cells[index]]])*10;ju[:,2]-=compression*v[:,2]*10
        with np.load(c/'abaqus/shell_field.npz') as cache:su=cache['u_mm']
        if abs(compression-s.get('field_compression',sa[-1]))<1e-6:ju-=np.mean(ju-su,axis=0)
        fig=plt.figure(figsize=(10,4.6),layout='constrained')
        for panel,(label,u,a) in enumerate([('JAX',ju,compression),('Abaqus',su,s.get('field_compression',sa[-1]))]):
            ax=fig.add_subplot(1,2,panel+1,projection='3d');deformed=v*10+u
            pc=Poly3DCollection(deformed[f if label=='JAX' else fshell],edgecolor='none',facecolor=COLORS[label],alpha=.9)
            ax.add_collection3d(pc);ax.set(xlim=(-1,11),ylim=(-1,11),zlim=(-1,11),
                xlabel='x (mm)',ylabel='y (mm)',zlabel='z (mm)',title=f'{label}: {a*100:.2f}% compression')
            ax.set_box_aspect((1,1,1));ax.view_init(22,-55)
        fig.suptitle(f'{kind}: actual displacement scale; geometry shown as midsurface')
        fig.savefig(c/'deformation.png',dpi=160);plt.close(fig)

table(D/'summary.csv',overview);table(D/'point_errors.csv',point_rows);write(D/'metrics.json',all_metrics)

fig,axes=plt.subplots(1,3,figsize=(13.2,4.2),layout='constrained')
for ax,kind in zip(axes,['G','P','D']):
    if kind not in ready:
        ax.set_title(kind+': pending');ax.axis('off');continue
    c=D/kind;met=next(m for m in all_metrics if m['kind']==kind)
    j=read(c/'jax/result.json') if (c/'jax/result.json').exists() else read(c/'jax/failure.json')
    for label,rows in [('JAX',j.get('path',j.get('accepted_path',[]))),('Abaqus',read(c/'abaqus/shell_result.json')['force_path'])]:
        a,f,t=loading(rows);ax.plot(a*100,f,color=COLORS[label],label=label,lw=1.6)
        pp=peak(a,f)
        if pp['observed_interior_peak']:ax.plot(pp['compression_pct'],pp['force_N'],'o',color=COLORS[label],ms=3)
    ax.set(xlabel='Compression (%)',ylabel='Compressive force (N)',xlim=(0,20),ylim=(0,None),
        title=f"{kind}: curve RMS / own shell peak {met['RMS_pct_matching_shell_observed_loading_peak']:.2f}%")
    ax.grid(alpha=.2)
axes[0].legend(fontsize=9)
fig.suptitle('Canonical TPMS: same L=10 mm, t=0.50 mm, NH E=10 MPa, nu=0.3, XYZ periodic, T=4 ms\nJAX HEX27 N32 / 27 GP, continuous occupancy; Abaqus S3R / Explicit; no contact\nDots: global dynamic maxima, which may occur after the first load drop',fontsize=10)
for ext in ['png','svg']:fig.savefig(D/('all_response.'+ext),dpi=180)
plt.close(fig)

fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
for ax,kind in zip(axes,['G','P','D']):
    if kind not in ready:
        ax.set_title(kind+': pending');ax.axis('off');continue
    c=D/kind;j=read(c/'jax/result.json') if (c/'jax/result.json').exists() else read(c/'jax/failure.json')
    jr=j.get('path',j.get('accepted_path',[]));sr=read(c/'abaqus/shell_result.json')['force_path']
    for label,rows in [('JAX',jr),('Abaqus',sr)]:
        t=np.array([r['time'] for r in rows]);ratios=np.array([r['KE_over_U'] if label=='JAX' else abs(r['ALLKE'])/max(abs(r['ALLIE']),1e-30) for r in rows])
        mask=t>1e-6;ax.semilogy(t[mask]*1000,np.maximum(ratios[mask],1e-8),label=label,color=COLORS[label])
    ax.axhline(.05,color='gray',ls=':',lw=1);ax.axvline(4,color='gray',ls='--',lw=.8)
    ax.set(title=kind,xlabel='Time (ms)',ylabel='Kinetic / internal energy',xlim=(0,4.4));ax.grid(alpha=.15)
axes[0].legend();fig.suptitle('Dynamics diagnostic: 5% line is historical quality screen, not a numerical failure threshold')
fig.savefig(D/'energy.png',dpi=180);plt.close(fig)

batch_state='本批六个物理动态尝试已结束' if (D/'batch_receipt.json').exists() else '本批仍在执行，当前为中间结果'
headline='；'.join(f"{m['kind']}初始差{m['initial_K_bias_pct']:+.2f}%、共同曲线差{m['RMS_pct_matching_shell_observed_loading_peak']:.2f}%" for m in all_metrics)
lines=['# G/P/D常见薄壁TPMS对照 r50','',f'已具备成对提取数据：{", ".join(ready)}。{batch_state}，下表仅覆盖实际提取的路径范围。','',headline+'。初始匹配与大压缩全路径精度分别评价，曲线差的定义及实际覆盖见下表。','',
    '本批在固定厚度和物理条件下检验方法对常见构型的适用范围。JAX采用原C²连续占据显式入口，Abaqus采用同解析中面的S3R/Explicit（G/P三角重连差异见下文）；壳是工程参考，不是真值。','',
    '本批三组均完整到20%及保载。5%压缩同点反力差为G+2.11%、P+1.81%、D+1.60%；初始阶段一致性较好。G在首次明显降载时较早，P/D较晚，峰后振幅也不同；现有设置不能普遍满足约10%的全程目标。D保载时间均力仍高约18%，不能把另外两组末端接近外推全部构型。以上针对本批快速动态对照，不是实验真值或准静态认证。','',
    '## 计算结果','',
    '| 构型 | 初始K偏差 | JAX/壳覆盖 | 共同曲线RMS/匹配壳峰 | 共同功偏差 | JAX/壳墙钟分钟（含首启预留） |',
    '| --- | ---: | --- | ---: | ---: | --- |']
for m in all_metrics:
    lines.append(f'| {m["kind"]} | {m["initial_K_bias_pct"]:+.2f}% | {m["JAX_max_compression_pct"]:.2f}% / {m["Abaqus_max_compression_pct"]:.2f}% | {m["RMS_pct_matching_shell_observed_loading_peak"]:.2f}% | {m["common_work_bias_pct"]:+.2f}% | {m["jax_wall_minutes"]:.2f} / {m["abaqus_wall_minutes"]:.2f} |')
lines += ['', '## 峰和数值诊断','', '| 构型 | 观察最大力位置 JAX/壳 | 峰力差 | 最小采样实体J | 回退块 | 末端动能比 JAX/壳 | 壳能量漂移/功 |', '| --- | --- | ---: | ---: | ---: | --- | ---: |']
for m in all_metrics:
    jp,sp=m['observed_JAX_maximum'],m['observed_shell_maximum']
    label='已观察内部峰' if m['both_observed_interior_peaks'] else '含端点最大值，不作失稳峰位比较'
    lines.append(f"| {m['kind']} | {jp['compression_pct']:.2f}% / {sp['compression_pct']:.2f}%（{label}） | {m['peak_force_bias_pct']:+.2f}% | {m['JAX_min_sampled_required_positive_J']:.4f} | {m['JAX_rejected_blocks']} | {100*m['JAX_terminal_KE_over_U']:.2f}% / {100*m['Abaqus_terminal_KE_over_IE']:.2f}% | {100*m['Abaqus_global_energy_drift_relative_work']:.2f}% |")
lines += ['', '上表位置是加载段观察全局最大力，峰后反弹可能超过首次降载前的峰，不能自动当作首屈曲点。P和D尤其需要区分：P图中壳约8–9%开始明显降载，JAX约12–13%；D壳约8–9%，JAX约10–11%。全局最大力可能出现在后段振荡。这里的区间是曲线图读数，未据此新设验收判据。','', '实体J为主要占据域采样点的局部体积比，正值是本批接受规则的一部分，不能证明全域无接触或每个时间步都无失效。低占据虚域的实际翻转数量与最小全域J保存在metrics.json；未夹断实际F/J。末端动能比例和壳能量漂移单列：完成动态路径与通过准静态/能量质量筛查不是同一结论，原checks全部保留。','', 'RMS是共同压缩窗口内两条反力曲线差的均方根，单位N；百分比以该构型的壳加载段观察最大力为分母。部分壳路径不冒称完整峰。功是反力对宏观压缩位移的积分。保载均力统一在4.2–4.4 ms固定窗口按时间梯形积分除以窗长；原逐行均值附存metrics，不修改提取器原记录，短窗口均值不等于稳态反力。曲线比较采用保存历史的线性插值，观察峰位受历史输出间隔限制，未经临界应变收敛认证。相同厚度不代表相同用料量；三种构型不按反力大小直接排名。','',
    '## 输入和执行边界','',
    'L10 mm、t0.50 mm；NH E10 MPa/ν0.3、基材密度1e−9 tonne/mm³；XYZ周期、宏观横向应变零；双方4 ms五次平滑加载到20%，保载0.4 ms。JAX N32 HEX27/27点、HRZ，η1e−4、10–90%界面宽0.05 mm；占据固定于参考高斯材料点。无接触/塑性/质量缩放，Abaqus默认体积黏性明列，JAX无附加阻尼。','',
    '宏观XYZ周期关系一致，但平移固定点的具体实现不同：JAX在背景周期类(0,0,0)固定波动q=0；壳固定首个非从属实体节点u=0，G/P/D位置分别为(0,0,5)/(0,2.5,5)/(0,0,2.5) mm。两端不是完全相同的离散动力学；本批未单独量化这一差异的动态影响，不据此直接归因全部峰差。固定点从现有代码和实际输入核对，计算中未改动。','',
    '启动前选择复用r12已完成20%的adaptive显式路径，代替草案的state-step-control：后者带历史高KE硬停，与本批允许动态跳跃的范围不一致。原生产程序未改，有限回退下限仍为初始dt/16，实际F/J不夹断。此选择在任何新动态结果前冻结。','',
    '几何采用每轴64区间提取，属于中面三角化分辨率，不是N64背景求解。去除孤立顶点并按周期等价同步处理0.01 mm以内短边；所有保留顶点仍在解析零等值面，独立商空间拓扑闭合。D与切面的相切内部点不属于开边界接缝；接缝按实际开边界边和周期边角检查。准备阶段错误和修正保留在preparation_attempt*，未在这些尝试提交动态计算。','',
    'G壳首次启动出现极细三角形，稳定时间步约3e−10s，未形成有意义加载路径即停机。G/P随后保持原解析零面顶点、仅采用内蕴Delaunay重连，周期商空间闭合和定向均通过；D原网格未改。G/P两侧三角面不完全相同，双向全部面心抽样距离最大分别0.003849/0.001433 mm（非全局Hausdorff界）；原JAX距离/占据字段不重算。修复前后Standard初始刚度变化均小于0.01%，不是按动态结果拟合网格；初始刚度接近不证明失稳峰位对三角化扰动不敏感。G原ODB及首启记录保留，320s保守成本预留计入其25分钟总预算，新求解上限1080s；P/D求解1400s预留退出及提取。六个物理动态算例，其中G壳因前处理问题多一次启动。','',
    '初始启动曾把JAX平台限制为cuda，排除了CPU预处理；修正为cuda,cpu，复用三项已完成的Standard初始结果。environment_launch_attempt01记录零JAX Problem/零动态任务，不计作力学失败或额外动态算例。','',
    '## 结果文件','',
    '- metrics.json：完整指标、峰/末端、能量、J、质量、停止原因。','- summary.csv、point_errors.csv：汇总和5/10/15/20%定点差。','- G/P/D各目录response.png/svg、deformation.png、geometry.png、几何审查及原结果。','- energy.png：双方动能比例，包含保载。','- protocol.json、batch_receipt.json、budget_summary.json：配置、启动记录和逐任务预算；六项累计动态预算记账约102.00 min，几何/初始检查另计。','- scientific_integrity.json：旧2243项记录、九项生产环境文件及原Word核对均未变。','',
    '这批曲线比较同时包含实体背景与壳运动学、占据几何、集中质量/耗散和动态分岔差异，不能把曲线差唯一归因于JAX程序或某一个因素；G/P初始复查也不证明峰区对微小三角化扰动不敏感。','',
    '约10%是探索性目标，不能只看平均误差忽略峰区差异。初始匹配不外推失稳/接触；未进行新设计AD或训练，原完整20%梯度限制保持。根据本批结果决定后续，不自动增加IWP、N64或调参扫描。']
report='\n'.join(lines)+'\n'
report=report.replace('## 计算结果','## 对照图\n\n![三种构型的完整加载曲线](all_response.png)\n\n[无惯性初始刚度图](initial_response.png) · [双方动能比例图](energy.png) · [逐点误差CSV](point_errors.csv)\n\n| 构型 | 初始几何 | 全程及初始段放大 | 20%保载末态形变 |\n| --- | --- | --- | --- |\n| G | [中面](G/geometry.png) | [反力曲线](G/response.png) | [形变](G/deformation.png) |\n| P | [中面](P/geometry.png) | [反力曲线](P/response.png) | [形变](P/deformation.png) |\n| D | [中面](D/geometry.png) | [反力曲线](D/response.png) | [形变](D/deformation.png) |\n\n## 计算结果',1).replace('## 输入和执行边界\n\n','## 输入和执行边界\n\n中面使用文献常见三角函数近似（不是严格解析极小曲面），令X=2πx/L、Y=2πy/L、Z=2πz/L：G为sinX cosY+sinY cosZ+sinZ cosX；P为cosX+cosY+cosZ；D为cosX cosY cosZ−sinX sinY sinZ。三者等值均为0、相位为0，厚度按到周期中面的距离定义，不用同一隐函数阈值冒充恒厚。\n\n',1).replace('## 结果文件\n\n','## 结果文件\n\n正式原始记录目录：`/home/xuehu/projects/tpms_jax/validation/canonical_tpms_20261009_r50`；Windows阅读副本：`output/r50_canonical_tpms_20261009`。原ODB在`E:/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus/`，当前动态目录为`canonical_tpms_20261009_r50_G_dynamic_quality`、`..._P_dynamic_quality`、`..._D_dynamic`；G首启原ODB保留在`..._G_dynamic`，不被覆盖。\n\n',1)
(D/'README.md').write_text(report)
print(json.dumps(overview,indent=2))

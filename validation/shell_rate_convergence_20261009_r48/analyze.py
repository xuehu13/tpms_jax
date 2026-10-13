"""Read-only r48 shell loading-rate comparison; never starts a solver."""
from pathlib import Path
import csv, hashlib, json, math, shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

R=Path('/home/xuehu/projects/tpms_jax')
D=R/'validation/shell_rate_convergence_20261009_r48'
W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an/output/r48_shell_rate_convergence_20261009')

def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()
def output(name,rows):
    (D/(name+'.json')).write_text(json.dumps(rows,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    if isinstance(rows,list) and rows:
        keys=list(dict.fromkeys(k for r in rows for k in r))
        with (D/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)

def energy(rows,lo,hi):
    q=[r for r in rows if lo<=r['compression']<=hi+1e-12]
    if not q:return None
    t=np.array([r['time'] for r in q]);ratio=np.array([r['ALLKE']/max(abs(r['ALLIE']),1e-30) for r in q])
    scale=max(max(abs(r['ALLWK']) for r in q),1e-30)
    dt=np.r_[0.,np.diff(t)]
    return {'compression_from':float(q[0]['compression']),'compression_to':float(q[-1]['compression']),'requested_from':lo,'requested_to':hi,'samples':len(q),
        'KE_over_IE_max':float(ratio.max()),'KE_over_IE_at_end':float(ratio[-1]),
        'time_fraction_KE_le_5pct':float(np.dot(dt,ratio<=.05)/dt.sum()) if dt.sum()>0 else None,
        'AE_max_over_window_max_WK':max(abs(r['ALLAE']) for r in q)/scale,
        'VD_max_over_window_max_WK':max(abs(r['ALLVD']) for r in q)/scale,
        'ETOTAL_drift_over_window_max_WK':max(abs(r['ETOTAL']-rows[0]['ETOTAL']) for r in q)/scale}

def main():
    protocol=read(D/'protocol.json');catalog=read(R/'validation/simulation_atlas_20261009_r47/curve_catalog.json')
    cases={};summaries=[];energies=[];sources={}
    for item in catalog:
        if item['case']=='diverse_04' and item['role']=='壳速率':
            cases[item['T_ms']]=(R/item['source'],False)
    for item in protocol['cases']:
        p=D/'abaqus'/('explicit_T'+item['tag'])/'results/shell.json'
        if not p.exists():p=p.with_name('partial_shell.json')
        if not p.exists():raise SystemExit('Waiting for '+str(p))
        cases[1000*item['T_seconds']]=(p,True)
    series={}
    for Tms,(p,new) in sorted(cases.items()):
        j=read(p);T=Tms/1000;rows=j['force_path'];sources[str(p.relative_to(R))]=sha(p)
        raw=np.array([[r['time'],r['compression'],-r['Fz_N']] for r in rows]);assert np.isfinite(raw).all()
        load=raw[raw[:,0]<T]
        full_loading=raw[-1,0]>=T-1e-10
        if full_loading:load=np.vstack([load,[T,np.interp(T,raw[:,0],raw[:,1]),np.interp(T,raw[:,0],raw[:,2])]])
        assert np.all(np.diff(load[:,1])>=-1e-10)
        k=int(np.argmax(load[:,2]));peak=load[k]
        observed=0<k<len(load)-1 and load[k+1:,2].min()<.9*peak[2]
        drop=None
        for n in range(k+1,len(load)-2):
            if np.all(load[n:n+3,2]<=.9*peak[2]):drop=float(load[n,1]);break
        rec={'T_ms':Tms,'new_job':new,'observed_peak_with_later_10pct_drop':bool(observed),
             'loading_complete_20pct':bool(full_loading),'covered_compression_pct':100*float(load[-1,1]),
             'recorded_max_compression_pct':100*float(peak[1]),'recorded_max_force_N':float(peak[2]),
             'peak_neighbor_spacing_pp':100*float(max(peak[1]-load[max(0,k-1),1],load[min(len(load)-1,k+1),1]-peak[1])),
             'first_sustained_10pct_drop_compression_pct':100*drop if drop is not None else None,
             'force_at_10pct_N':float(np.interp(.1,load[:,1],load[:,2])) if load[-1,1]>=.1 else None,
             'loading_work_N_mm':float(10*np.trapezoid(load[:,2],load[:,1])),
             'original_status':j['status'],'original_failed_checks':'; '.join(n for n,v in j['checks'].items() if not v),
             'source':str(p.relative_to(R)),'source_sha256':sources[str(p.relative_to(R))]}
        pr=min(rows,key=lambda z:abs(z['time']-peak[0]));pscale=max(abs(pr['ALLWK']),1e-30)
        rec.update(peak_KE_over_IE=pr['ALLKE']/max(abs(pr['ALLIE']),1e-30),
            peak_AE_over_WK=pr['ALLAE']/pscale,peak_VD_over_WK=pr['ALLVD']/pscale,
            peak_ETOTAL_drift_over_WK=abs(pr['ETOTAL']-rows[0]['ETOTAL'])/pscale)
        summaries.append(rec);series[Tms]={'load':load,'raw':raw,'rows':rows,'summary':rec}
        for name,lo,hi in [('pre_1_to_10pct',.01,.1),('pre_1pct_to_own_peak',.01,float(peak[1])),('post_peak_to_20pct',float(peak[1]),.2)]:
            stats=energy([z for z in rows if z['time']<=T+1e-12],lo,hi)
            energies.append({'T_ms':Tms,'window':name,**(stats if stats else {'status':'not covered'})})
    ref=series[80.]['load'];den=series[80.]['summary']['recorded_max_force_N']
    for rec in summaries:
        x=series[rec['T_ms']]['load'];rec['peak_shift_vs_80ms_pp']=rec['recorded_max_compression_pct']-series[80.]['summary']['recorded_max_compression_pct']
        rec['peak_force_relative_vs_80ms']=rec['recorded_max_force_N']/den-1
        for hi,tag in [(.1,'0_10pct'),(.2,'0_20pct')]:
            if hi>min(x[-1,1],ref[-1,1])+1e-7:
                rec['RMS_'+tag+'_N']=None;rec['RMS_'+tag+'_over_80ms_peak']=None;continue
            g=np.linspace(0,hi,2001);diff=np.interp(g,x[:,1],x[:,2])-np.interp(g,ref[:,1],ref[:,2])
            rec['RMS_'+tag+'_N']=float(np.sqrt(np.mean(diff**2)))
            rec['RMS_'+tag+'_over_80ms_peak']=rec['RMS_'+tag+'_N']/den if series[80.]['summary']['loading_complete_20pct'] else None
        rec['loading_work_relative_vs_80ms']=rec['loading_work_N_mm']/series[80.]['summary']['loading_work_N_mm']-1 if rec['loading_complete_20pct'] and series[80.]['summary']['loading_complete_20pct'] else None
    pairs=[]
    for fast,slow in [(20.,40.),(40.,80.)]:
        a=series[fast]['summary'];b=series[slow]['summary'];dp=b['recorded_max_compression_pct']-a['recorded_max_compression_pct'];df=b['recorded_max_force_N']/a['recorded_max_force_N']-1
        pairs.append({'fast_T_ms':fast,'slow_T_ms':slow,'peak_shift_pp':dp,'peak_force_relative_change':df,
            'both_observed_peak':a['observed_peak_with_later_10pct_drop'] and b['observed_peak_with_later_10pct_drop'],
            'peak_position_working_gate_pass':abs(dp)<=.2 and a['observed_peak_with_later_10pct_drop'] and b['observed_peak_with_later_10pct_drop'],'peak_force_working_gate_pass':abs(df)<=.02 and a['observed_peak_with_later_10pct_drop'] and b['observed_peak_with_later_10pct_drop'],
            'scope':'Same discrete shell only; not physical truth or complete quasistatic certification'})
    output('rate_summary',summaries);output('slow_pairs',pairs);output('energy_windows',energies)
    # Existing actual midsurface fields, with no geometric or spatial fitting.
    folders={40.:R/'validation/mechanism_20261007_r17/shell_frames'}
    for c in protocol['cases']:folders[1000*c['T_seconds']]=D/'abaqus'/('explicit_T'+c['tag'])/'results/shell_frames'
    with np.load(folders[40.]/'reference.npz') as z:
        xyz=z['xyz_mm'];tri=z['triangles'];weights=z['reference_area_weights_mm2'];labels=z['labels']
    field={};modes=[]
    for T,f in folders.items():
        with np.load(f/'reference.npz') as z:
            np.testing.assert_array_equal(z['labels'],labels);np.testing.assert_array_equal(z['triangles'],tri)
            np.testing.assert_allclose(z['xyz_mm'],xyz,rtol=0,atol=1e-12)
        meta=read(f/'modes.json');sources[str((f/'modes.json').relative_to(R))]=sha(f/'modes.json')
        for requested in [.1,.114,.12,.14]:
            row=min(meta['rows'],key=lambda z:abs(z['requested_compression']-requested));p=f/row['file']
            with np.load(p) as z:field[T,requested]={'w':z['fluctuation_mm'],'u':z['u_mm'],'normal':z['normal_fluctuation_mm'],'compression':float(z['compression'])}
            sources[str(p.relative_to(R))]=sha(p)
    for ta,tb in [(20.,40.),(40.,80.),(20.,80.)]:
        for target in [.1,.114,.12,.14]:
            a=field[ta,target];b=field[tb,target];va=a['w'];vb=b['w']
            aa=float(np.sum(weights[:,None]*va*va));bb=float(np.sum(weights[:,None]*vb*vb));ab=float(np.sum(weights[:,None]*va*vb))
            modes.append({'T_a_ms':ta,'T_b_ms':tb,'requested_compression_pct':target*100,
                'actual_a_pct':a['compression']*100,'actual_b_pct':b['compression']*100,
                'area_weighted_direction_cosine':ab/math.sqrt(aa*bb),
                'fluctuation_difference_RMS_mm':float(np.sqrt(np.sum(weights[:,None]*(va-vb)**2)/weights.sum())),
                'scope':'Nearest saved frames, subtract affine and area-mean translation; not eigenmodes or exact critical location'})
    output('mode_comparison',modes)
    # Scientific figures: raw histories, consistent colours, explicit time units.
    for f in ['msyh.ttc','simhei.ttf']:
        p=Path('/mnt/c/Windows/Fonts')/f
        if p.exists():font_manager.fontManager.addfont(str(p))
    plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'font.size':10,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'svg.fonttype':'none'})
    palette={.25:'#89929b',1.:'#9365a4',2.:'#279180',4.:'#d05b24',20.:'#1764a0',40.:'#343e49',80.:'#ac2646'}
    def save(fig,name):
        fig.savefig(D/(name+'.png'),dpi=180);fig.savefig(D/(name+'.svg'));plt.close(fig)
    def footer(fig,text):fig.text(.06,.026,text,fontsize=9,color='#374151')
    fig,axs=plt.subplots(2,2,figsize=(13.4,9.2));fig.subplots_adjust(left=.07,right=.97,bottom=.13,top=.88,hspace=.4,wspace=.22)
    fig.suptitle('diverse_04：同一Abaqus壳的加载时长与峰值响应',x=.06,y=.965,ha='left',fontsize=17,fontweight='bold')
    fig.text(.06,.915,'新增20/80ms；复用0.25/1/2/4/40ms。原曲线不平移、不平滑，20%后保载不参与峰值。',fontsize=10)
    for T,c in series.items():
        q=c['load'];label=f'T={T:g}ms'+('（新）' if T in [20,80] else '')
        if not c['summary']['loading_complete_20pct']:label+=f'，止{q[-1,1]*100:.2f}%'
        for ax in axs[0]:
            if ax is axs[0,1] and T==.25:continue
            ax.plot(q[:,1]*100,q[:,2],color=palette[T],lw=1.8 if T in [20,80] else 1.1,ls='-' if T in [20,80] else '--',label=label)
            if not c['summary']['loading_complete_20pct']:ax.plot(q[-1,1]*100,q[-1,2],'x',color=palette[T],ms=6)
        axs[0,1].plot(c['summary']['recorded_max_compression_pct'],c['summary']['recorded_max_force_N'],'o',color=palette[T],ms=3) if T!=.25 else None
    axs[0,0].set(xlabel='宏观压缩量 (%)',ylabel='压缩宏观反力 (N)',xlim=(0,20));axs[0,0].legend(fontsize=8,ncol=2)
    axs[0,1].set(xlabel='宏观压缩量 (%)',ylabel='压缩宏观反力 (N)',xlim=(10.5,13.5));axs[0,1].legend(fontsize=8,ncol=2)
    normal=[r for r in summaries if r['T_ms']>=1]
    for ax,key,ylab in [(axs[1,0],'recorded_max_compression_pct','加载最大力位置 (%)'),(axs[1,1],'recorded_max_force_N','加载最大压缩力 (N)')]:
        ax.plot([r['T_ms'] for r in normal],[r[key] for r in normal],'-',color='#8a929d',lw=.9)
        for r in summaries:
            T=r['T_ms'];ax.plot(T,r[key],'o' if T>=1 else 'x',color=palette[T],ms=6)
            ax.annotate(f'{T:g}ms\n{r[key]:.4f}',(T,r[key]),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
        ax.set(xscale='log',xlabel='加载时长 T (ms，对数坐标)',ylabel=ylab);ax.margins(y=.25)
    footer(fig,'L10mm，t0.5mm，NH E10MPa/ν0.3/ρ10⁻⁹ tonne/mm³，S3R/5厚向点，XYZ周期。0.25ms全局最大值不等同慢路径跳跃事件。')
    save(fig,'rate_response')
    fig,axs=plt.subplots(2,2,figsize=(13.4,9.2));fig.subplots_adjust(left=.075,right=.97,bottom=.13,top=.88,hspace=.4,wspace=.24)
    fig.suptitle('diverse_04：峰前质量与跳跃后的动力效应分别查看',x=.06,y=.965,ha='left',fontsize=17,fontweight='bold')
    fig.text(.06,.915,'KE=动能，IE=内能，AE=人工能量，VD=粘性耗散，WK=外功；这些图不修改原有质量检查。',fontsize=10)
    for T in [1.,4.,20.,40.,80.]:
        c=series[T];rr=[z for z in c['rows'] if z['time']<=T/1000 and z['compression']>=.01];a=np.array([z['compression'] for z in rr])*100
        ratio=np.array([z['ALLKE']/max(abs(z['ALLIE']),1e-30) for z in rr]);scale=max(abs(z['ALLWK']) for z in rr)
        axs[0,0].plot(a,ratio*100,color=palette[T],label=f'{T:g}ms');axs[0,1].plot(a,np.array([z['ALLAE'] for z in rr])/scale*100,color=palette[T])
        axs[1,0].plot(a,np.array([z['ETOTAL']-c['rows'][0]['ETOTAL'] for z in rr])/scale*100,color=palette[T]);axs[1,1].plot(a,np.array([z['ALLVD'] for z in rr])/scale*100,color=palette[T])
    axs[0,0].set(yscale='log',ylabel='KE / IE (%)');axs[0,0].axhline(5,color='#6d7581',ls=':',lw=.9);axs[0,0].legend(fontsize=8)
    axs[0,1].set_ylabel('AE / 各自加载最大|WK| (%)');axs[1,0].set_ylabel('ETOTAL变化 / 各自加载最大|WK| (%)');axs[1,1].set_ylabel('VD / 各自加载最大|WK| (%)')
    for ax in axs.flat:ax.set(xlabel='宏观压缩量 (%)',xlim=(1,20))
    footer(fig,'峰前1–10%、1%至各自峰以及峰后分窗数值见energy_windows.csv。跳跃KE升高单列，能量漂移和人工能量仍需报告。')
    save(fig,'rate_energy')
    fig=plt.figure(figsize=(13.4,9.2));fig.suptitle('diverse_04：同一中面在不同T下的实际变形',x=.06,y=.965,ha='left',fontsize=17,fontweight='bold')
    fig.text(.06,.915,'位移不放大；颜色为参考法向波动位移。每幅标注实际最近保存帧压缩量，场帧不用于精确定位峰。',fontsize=10)
    vmax=max(float(np.max(np.abs(field[T,target]['normal']))) for T in [20.,40.,80.] for target in [.12,.14]);norm=colors.Normalize(-vmax,vmax);cmap=plt.get_cmap('RdBu_r')
    for i,target in enumerate([.12,.14]):
        for j,T in enumerate([20.,40.,80.]):
            f=field[T,target];ax=fig.add_subplot(2,3,i*3+j+1,projection='3d');points=xyz+f['u'];face=cmap(norm(f['normal'][tri].mean(axis=1)))
            mesh=Poly3DCollection(points[tri],facecolors=face,edgecolors='none',rasterized=True);ax.add_collection3d(mesh)
            ax.set(xlim=(-.5,10.5),ylim=(-.5,10.5),zlim=(-.5,10.5),xlabel='X (mm)',ylabel='Y (mm)',zlabel='Z (mm)');ax.set_box_aspect((1,1,1));ax.view_init(elev=20,azim=-55)
            ax.set_title(f'T{T:g}ms，压缩{f["compression"]*100:.3f}%',fontsize=10);ax.tick_params(labelsize=8)
    fig.subplots_adjust(left=.02,right=.9,bottom=.1,top=.86,hspace=.2,wspace=.02);cax=fig.add_axes([.93,.25,.015,.45]);fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,label='参考法向波动位移 (mm)')
    footer(fig,'相同节点、三角连接及面积权重；扣除宏观仿射位移和平移计算方向余弦。模式差不等于数值错误，也不直接认证真实屈曲模式。')
    save(fig,'rate_modes')
    changed=[n for n,h in read(D/'frozen_before.json').items() if sha(R/n)!=h];assert not changed,changed
    batch=read(D/'batch_receipt.json');stable=all(p['peak_position_working_gate_pass'] and p['peak_force_working_gate_pass'] for p in pairs)
    result={'new_shell_jobs':2,'new_JAX_forward':0,'AD':0,'two_slow_pairs_pass_frozen_peak_gates':stable,
            'slow_pairs':pairs,'all_original_protected_unchanged':not changed,'protected_files':len(read(D/'frozen_before.json')),
            'old_scientific_records_349_unchanged':True,'source_hashes':sources,'batch_receipt':batch,
            'scope':'T sensitivity of this fixed S3R model, not physical accuracy or complete 20% JAX accuracy certification'}
    output('result',result)
    def fmt(x,d=4):return 'n.a.' if x is None else f'{x:.{d}f}'
    text='''# r48：Abaqus壳加载时长与峰值稳定性

本轮直接检验diverse_04壳参考自身的速率敏感性。新增T20/80ms两项壳分析，复用T0.25/1/2/4/40ms；无新JAX前向、N32/N64接续或AD。T是物理加载时长，不是计算耗时或稳定时间步dt。

## 研究判断怎样区分

Abaqus/Explicit求解动力平衡。给定模型的动力解可以数值自洽，而峰值仍随加载时长变化；不能把某个T的动力峰直接当静态屈曲真值。当前材料没有引入应变率本构，惯性、变形历史和默认体积粘性仍可带来T效应。[SIMULIA的准静态说明](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEGSARefMap/simagsa-m-Quasi-sb.htm)要求分辨加速引入的动力效应。

屈曲还可能受微小扰动和触发模式影响。官方[柱屈曲示例](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEBMKRefMap/simabmk-c-sscxsec.htm)说明高度对称模型可延迟触发，速率也可能改变模式。本轮没有植入缺陷、改阻尼或换网格；T稳定只支持当前离散壳的内部一致性，不证明真实薄壁或任意构型准确。

## 同模型对照与实际执行

沿用diverse_04中面、L10mm、t0.5mm、8372节点/15914个S3R、5个厚向点；NH E10MPa/ν0.3/密度10⁻⁹ tonne/mm³，XYZ周期波动和转角周期、横向宏观应变0。每作业4CPU、double=both、默认体积粘性，无质量缩放、接触或塑性。新INP只改幅值时间、总步时间、历史输出间隔三行，反向替换逐字节恢复原T4ms输入。完整目标为20%+0.1T保载，共同墙钟40min，单项20ms/80ms上限15/30min，预算不足不自动延期。

'''
    for c in batch['cases']:
        text+=f"- T={c['T_seconds']*1000:g}ms：{'求解完成' if c.get('analysis_completed_successfully') else '未完整完成'}，求解墙钟{fmt(c.get('solver_wall_seconds',c['wall_seconds'])/60,2)}min，总含提取{c['wall_seconds']/60:.2f}min；原生目录 `{c['native_directory']}`。\n"
        if c.get('exception_message'):text+=f"  原停止：{c['exception_message']}。有效路径范围见表，不补成完整20%或完整保载。\n"
    text+=f"\n两作业并行，总UTC墙钟{batch['UTC_wall_seconds']/60:.2f}min；CPU时间与墙钟不同，不能把两个单项墙钟相加当本轮总耗时。\n\n## 峰值与覆盖范围\n\n| T (ms) | 记录最大力位置 (%) | 记录最大压缩力 (N) | 后续持续降载10%位置 (%) | 峰邻点最大间距 (百分点) | 加载完整20% |\n| ---: | ---: | ---: | ---: | ---: | :---: |\n"
    for q in summaries:text+=f"| {q['T_ms']:g} | {q['recorded_max_compression_pct']:.4f} | {q['recorded_max_force_N']:.5f} | {fmt(q['first_sustained_10pct_drop_compression_pct'])} | {q['peak_neighbor_spacing_pp']:.4f} | {'是' if q['loading_complete_20pct'] else '否，至'+fmt(q['covered_compression_pct'])+'%'} |\n"
    text+='\n0.25ms的加载全局最大值来自明显不同的快速动力响应，不能直接与慢路径跳跃事件等同。所有位置来自实际保存历史；没有曲线平移、拟合、平滑或外推。最大值后有三条连续低于90%的记录才报告持续降载位置；有限窗口内的观察峰不保证未知未来全局最大值。\n\n![反力、峰位与峰力](rate_response.png)\n\n## 运行前固定的放慢判据\n\n| 放慢序列 | 峰位变化 (百分点) | 峰力变化 (%) | 峰位≤0.2个百分点且峰力≤2% |\n| --- | ---: | ---: | :---: |\n'
    for q in pairs:text+=f"| {q['fast_T_ms']:g}→{q['slow_T_ms']:g}ms | {q['peak_shift_pp']:+.4f} | {q['peak_force_relative_change']*100:+.4f} | {'通过' if q['peak_position_working_gate_pass'] and q['peak_force_working_gate_pass'] else '未通过'} |\n"
    text+='\n'+('两次逐次放慢均满足本轮工作标准，支持当前离散壳的慢加载观察峰对T相对稳定。' if stable else '本轮未获得两次逐次放慢都满足预设工作标准的证据，不能宣布壳峰对T稳定。')
    text+='该标准不是行业精度认证，也不将峰位的百分点差写成相对百分比。\n\n## 峰前与跳跃后质量分开看\n\n| T (ms) | 峰处KE/IE (%) | 峰处AE/|WK| (%) | 峰处VD/|WK| (%) | 峰处ETOTAL漂移/|WK| (%) |\n| ---: | ---: | ---: | ---: | ---: |\n'
    for T in [20.,40.,80.]:
        q=series[T]['summary'];text+=f"| {T:g} | {q['peak_KE_over_IE']*100:.5f} | {q['peak_AE_over_WK']*100:.5f} | {q['peak_VD_over_WK']*100:.5f} | {q['peak_ETOTAL_drift_over_WK']*100:.5f} |\n"
    text+='\nKE为动能、IE为内能、AE为人工能量、VD为粘性耗散、WK为外功。完整1–10%、1%至各自峰及峰后加载的统计见energy_windows.csv，窗口取实际覆盖范围。原提取器全部检查与失败保存在rate_summary.csv，未因峰前较好而改判全20%质量。跳跃后动能上升可以是动力现象，但人工能量、能量漂移及变形模式仍需报告。\n\n![能量随压缩变化](rate_energy.png)\n\n## 变形场与模式范围\n\n用同一节点/三角连接/面积权重，扣除宏观仿射位移和面积平均平移。方向余弦接近1表示位移方向相近，不是准确率；它不验证临界特征模态。实际场帧比历史反力稀疏，实际压缩量写在图与表里，不能用它精确定位峰。\n\n| 实际最近目标 | T比较 (ms) | 两实际压缩量 (%) | 面积加权方向余弦 | 波动位移差RMS (mm) |\n| --- | --- | --- | ---: | ---: |\n'
    for q in modes:
        if q['T_a_ms']!=40.:continue
        text+=f"| {q['requested_compression_pct']:g}% | 40/80 | {q['actual_a_pct']:.4f}/{q['actual_b_pct']:.4f} | {q['area_weighted_direction_cosine']:.6f} | {q['fluctuation_difference_RMS_mm']:.6f} |\n"
    text+='\n![实际变形与法向波动](rate_modes.png)\n\n## 对当前JAX差异意味着什么\n\n本轮没有改变JAX结果。原二值N32 T4ms峰15.2387%，同T壳11.8241%，仍差3.4146个百分点；慢壳稳定与否都不能抹去这项同T事实。若慢加载壳峰趋稳，加载时长不是全部错位的充分解释；若不趋稳，当前壳参照的不确定性需要先界定。两者都不能仅据不吻合宣布整个JAX方法失效，或认定Abaqus算法错误。\n\n当前壳仍有实体/壳运动学、离散化、缺陷触发和未建接触等模型范围；本轮只隔离T，不认证这些方面。下一步先讨论本轮结果，不自动新增T160ms、JAX路径、质量缩放、缺陷或训练。\n\n## 文件与保护\n\n正式目录 `/home/xuehu/projects/tpms_jax/validation/shell_rate_convergence_20261009_r48`；Windows阅读镜像为当前目录。原生INP/ODB在表列E盘新目录，原作业不动。protocol.json是运行前协议；batch_receipt.json和各case/launch_receipt.json是实际执行；rate_summary.csv、slow_pairs.csv、energy_windows.csv、mode_comparison.csv为可追踪数值；三图都有PNG/SVG。531项保护记录（含原349项及r47图集）字节保持，生产源码零修改。准备阶段一次反向字符串检查拒绝已保存在preparation_attempt01.json，修复为精确整行替换；它发生于作业提交前，不是FEM失败。\n'
    text=text.replace('## 对当前JAX差异意味着什么','## 综合判断与采样限制\n\n40→80ms峰位变化0.0410个百分点，与峰附近相邻历史记录约0.041个百分点的间距相当。因此支持“慢加载观察峰已相对稳定”，不把11.3750%当作无限精度静态临界点。20/40/80ms峰处KE/IE分别约0.0322%/0.00235%/0.000519%，峰处人工能量/外功均小于0.1%；峰前质量明显优于1ms。\n\n峰后则不能同样认证：20/40/80ms实际覆盖的峰后窗口内，人工能量最大值除以该窗口最大绝对外功约11.03%/10.06%/8.55%，ETOTAL漂移相对同分母约5.95%/6.76%/6.38%。这些数值保留原失败，不能仅用加载变慢宣布全20%曲线准确。粘性耗散也在跳跃后显著累积；本轮没有改变默认体积粘性，不能由此唯一归因峰位差。峰后动能跃升不单独用作否决，数值能量问题也不归为“正常跳跃”而忽略。\n\n同约9.813%保存帧的40/80ms波动位移方向余弦接近1；约11.863%帧为0.80755，约13.823%为0.95812。跳跃后时相/变形差异仍存在，不能把峰位趋稳写成整个动力路径或模式完全一致。\n\n'+'## 对当前JAX差异意味着什么',1)
    (D/'README.md').write_text(text)
    for p in D.iterdir():
        if p.is_file() and p.name not in ['run_shell.py','analyze.py']:shutil.copy2(p,W/p.name)
    print(json.dumps({'peak_table':[{k:q[k] for k in ['T_ms','recorded_max_compression_pct','recorded_max_force_N','loading_complete_20pct','covered_compression_pct']} for q in summaries],'slow_pairs':pairs,'stable':stable},ensure_ascii=False,indent=2))

if __name__=='__main__':main()

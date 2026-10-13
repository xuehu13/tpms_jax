"""Read-only atlas of existing TPMS evidence. Never imports or runs a solver."""
from pathlib import Path
import json, hashlib, csv, sys

R = Path('/home/xuehu/projects/tpms_jax')
O = Path(sys.argv[sys.argv.index('--output')+1]) if '--output' in sys.argv else Path(__file__).resolve().parent

def read(p):
    return json.loads(p.read_text())

def compact(x):
    if isinstance(x, list):
        if len(x) > 8:
            return {'length':len(x), 'first':compact(x[0]), 'last':compact(x[-1])}
        return [compact(y) for y in x]
    if isinstance(x, dict):
        return {k:compact(v) for k,v in x.items() if k not in ['nodes','elements','nodal_displacements','samples','source_sha256','input_sha256','field_sha256']}
    return x

def inventory():
    found=[]
    roots=[R/'validation', R/'results']
    for root in roots:
        for p in sorted(root.rglob('*.json')):
            if p.stat().st_size > 30_000_000: continue
            try: j=read(p)
            except (ValueError,UnicodeError):continue
            curves=[]
            if isinstance(j,list) and j and isinstance(j[0],dict) and 'Fz_N' in j[0] and 'compression' in j[0]:curves=[('',j)]
            elif isinstance(j,dict):
                for key in ['path','force_path','accepted_path','valid_replay_observations']:
                    a=j.get(key)
                    if isinstance(a,list) and a and isinstance(a[0],dict) and 'Fz_N' in a[0] and 'compression' in a[0]:curves.append((key,a))
                if p.name=='accepted_logged_observations.json' and j.get('rows'):curves.append(('rows',j['rows']))
            if not curves:continue
            inputs=[]
            for d in [p.parent,*p.parents]:
                if d==R:break
                for name in ['input.json','protocol.json']:
                    f=d/name
                    if f.exists() and f!=p:
                        inputs.append({'file':str(f.relative_to(R)),'data':compact(read(f))})
                if len(inputs)>=3:break
            for k,a in curves:
                rec={'file':str(p.relative_to(R)),'key':k,'rows':len(a),'metadata':compact({z:v for z,v in j.items() if z not in ['path','force_path','accepted_path','valid_replay_observations']}) if isinstance(j,dict) else {},'first':a[0],'last':a[-1],'inputs':inputs}
                found.append(rec)
    (O/'inventory.json').write_text(json.dumps(found,indent=2,ensure_ascii=False))
    for i,x in enumerate(found):
        print(i,x['file'],x['key'],x['rows'],x['last'].get('compression'),x['last'].get('time'))

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

# Explicit inclusion recipe, keyed to the frozen inventory. Replays and probes
# have their own role; they are never joined to an independent loading path.
SPECS = [
 (0,'04-B32','04','主路径','二值；原预算检查点精确接续，完整20%+保载'),
 (4,'04-S025','04','壳速率','T0.25ms；原能量/惯性/人工能筛查未过'),
 (5,'04-S1','04','壳速率','T1ms；原能量/惯性/人工能筛查未过'),
 (8,'04-B64','04','主路径','预算停止13.5877%，仍增载，无峰'),
 (10,'04-S40','04','壳速率','T40ms；保留原质量检查'),
 (11,'04-J15','04','稀疏日志','仅14条已打印接受记录；不是完整接受历史'),
 (12,'28-Q64','28','主路径','HEX27/64点；20%完成但密采样材料域失败'),
 (13,'28-P2','28','资源探针','N2短路径；用于传输/资源检查'),
 (15,'28-S20','28','壳速率','T20ms'),(16,'28-S40','28','壳速率','T40ms'),
 (17,'28-STD20','28','Standard路径','20%完成；原能量/人工能检查未过；时间为伪时间'),
 (18,'28-H8','28','主路径','HEX8 N64；原NH自适应完整路径'),
 (19,'28-P64','28','资源探针','N64初始32步；未覆盖实质压缩'),
 (21,'28-Dminus','28','梯度扰动','t0.4975mm；原NH总导数20%符号失败'),
 (22,'28-Dplus','28','梯度扰动','t0.5025mm；原NH总导数20%符号失败'),
 (23,'28-Dreplay','28','固定时间表','原基准接受时间表独立重放；20%梯度未过'),
 (25,'28-NH4','28','主路径','HEX27/27点 NH，T4ms'),
 (26,'28-NH8','28','主路径','HEX27/27点 NH，T8ms'),
 (27,'28-P32','28','资源探针','HEX27 N32准备后32步'),
 (30,'04-J17','04','主路径','原控制；15.9662%末个块接受态后数值失败'),
 (32,'04-Rfail','04','局部重放','首错前逐步有效记录；15.9682–16.0229%，非从零路径'),
 (33,'04-Rdt','04','局部重放','减步短重放；15.9662–16.2139%，非完整路径'),
 (35,'04-S2','04','壳速率','T2ms；原质量检查未过'),
 (36,'04-J64','04','主路径','平滑N64 T2ms；14.7320%触原步长底线'),
 (38,'28-OV','28','主路径','旧objective_void；20%完成但混合尾域6点失败'),
 (40,'04-Fill','04','壳填料','嵌入S3R+C3D8R近零填料；8.35%失败；非JAX等价'),
 (41,'04-S4','04','壳速率','T4ms；原能量/惯性/人工能筛查未过'),
 (42,'04-J18','04','主路径','状态步长控制；17.5363%预算停止'),
 (44,'28-t45','28','厚度范围','t0.45mm；C²完整20%+保载'),
 (45,'28-S45','28','厚度壳','t0.45mm T40ms'),
 (46,'28-t55','28','厚度范围','t0.55mm；C²完整20%+保载'),
 (47,'28-S55','28','厚度壳','t0.55mm T40ms'),
 (48,'28-ST1','28','早期Standard','1%静力路径'),
 (49,'28-ST5','28','早期Standard','5%静力路径'),
 (50,'28-ST10','28','早期Standard','10%路径；原能量检查未过'),
 (51,'28-ST1old','28','早期Standard','1%旧输出；若完全重复仅在索引保留'),
 (52,'28-ST5old','28','早期Standard','5%旧粗增量；原检查未过'),
 (53,'28-ST5mid','28','早期Standard','5%中间版本；原检查未过'),
 (54,'28-ST498old','28','早期Standard','t0.498mm旧输出；若重复仅在索引保留'),
 (55,'28-ST498','28','早期Standard','t0.498mm；1%路径'),
 (56,'28-ST502','28','早期Standard','t0.502mm；1%路径'),
 (57,'28-C4','28','主路径','当前C²核 T4ms；完整20%+保载'),
 (58,'28-C8','28','主路径','当前C²核 T8ms；完整20%+保载'),
 (59,'04-Support','04','支撑干预','深虚域能量×0.1，质量不变；13.0777%预算停止'),
]

def build():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.backends.backend_pdf import PdfPages
    for f in ['msyh.ttc','simhei.ttf']:
        p=Path('/mnt/c/Windows/Fonts')/f
        if p.exists():font_manager.fontManager.addfont(str(p))
    plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'font.size':10,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2,'svg.fonttype':'none','pdf.fonttype':42})
    inv=read(O/'inventory.json'); curves={}; records=[]; omissions=[]; files={}
    def loadrows(x):
        j=read(R/x['file']);return j if not x['key'] else j[x['key']]
    # Record every old source consumed, including metadata, without changing it.
    def source(p):
        p=Path(p);files[str(p.relative_to(R))]=sha(p);return read(p)
    for idx,rid,case,role,note in SPECS:
        x=inv[idx];rows=loadrows(x);source(R/x['file'])
        d={}
        for z in reversed(x['inputs']):d.update(z['data']);source(R/z['file'])
        typ='Abaqus' if ('/abaqus/' in x['file'] or x['file'].endswith('shell.json') or rid=='04-Fill') else 'JAX'
        static=typ=='Abaqus' and not ('explicit_' in x['file'] or rid=='04-Fill')
        n=d.get('cells_per_axis',d.get('N')); elem=d.get('element_type',d.get('element','HEX8' if rid in ['28-H8','28-P64'] else 'HEX27'))
        qp=d.get('Gauss_points_per_cell',8 if elem=='HEX8' else 27)
        T=d.get('load_time_seconds',d.get('JAX_load_time_seconds')) if not static else None
        t=d.get('thickness_mm',.5);w=d.get('interface_10_90_mm',.05)
        kernel='C²' if 'virtual_continuation' in d else ('旧客观虚域' if d.get('material_model')=='objective_void' else 'NH')
        if rid=='04-J15':
            # r15's readiness input predates explicit continuation metadata.
            # Its actual execution record identifies the same C² candidate.
            p=R/'validation/geometry_transfer_20261006_r15/STEP2.md'
            files[str(p.relative_to(R))]=sha(p)
            assert 'NH/XYZ/HEX27/HRZ/C²' in p.read_text()
            kernel='C²';note+='；材料核按实际STEP2记录核对'
        if typ=='Abaqus':elem='S3R';n=None;qp=5;kernel='NH壳';w=None
        if rid=='04-Fill':T=.004;t=.5;elem='S3R+C3D8R';n=32;kernel='近零填料';qp=None
        if rid=='28-Dreplay':n=32;elem='HEX27';qp=27;kernel='NH'
        binary=rid in ['04-B32','04-B64'];occupancy='二值' if binary else ('中面壳' if typ=='Abaqus' else '平滑')
        if binary:w=0.
        if role=='主路径' and typ=='JAX':
            f=R/x['file'];fail=f.parent/'failure.json'
            if fail.exists():
                m=source(fail);x['metadata'].update({k:v for k,v in m.items() if k!='accepted_path'})
        m=x['metadata'];checks=m.get('checks',{});failed=[k for k,v in checks.items() if v is False]
        a=np.array([[y['time'],y['compression'],-y['Fz_N']] for y in rows],float)
        assert np.isfinite(a).all(),rid
        # Standard step time resets across static steps; preserve its original
        # row order and use only monotone compression, never as physical time.
        if not static:assert np.all(np.diff(a[:,0])>=-1e-12),rid
        # Only byte-equivalent numerical time/compression/force triples deduplicate.
        sig=hashlib.sha256(a.tobytes()).hexdigest()
        duplicate=next((z['id'] for z in records if z.get('data_signature')==sig),None)
        if duplicate:
            omissions.append({'source':x['file'],'id':rid,'reason':'相同时间/压缩/反力记录，已画 '+duplicate});continue
        if T and a[-1,0]>=T-1e-10:
            before=a[a[:,0]<T];end=np.array([T,np.interp(T,a[:,0],a[:,1]),np.interp(T,a[:,0],a[:,2])]);loading=np.vstack([before,end])
        else:loading=a.copy()
        # Rightmost value for exactly repeated compression; no hold values enter.
        mask=np.r_[np.diff(loading[:,1])>1e-12,True];xy=loading[mask][:,1:]
        assert np.all(np.diff(xy[:,0])>0),rid
        peak_i=int(np.argmax(xy[:,1]));peak=xy[peak_i]
        interior=0<peak_i<len(xy)-1 and np.min(xy[peak_i+1:,1])<.999*peak[1]
        drop=(1-np.min(xy[peak_i+1:,1])/peak[1]) if interior and peak[1]>0 else None
        observed=interior and role not in ['资源探针','局部重放','稀疏日志']
        complete=xy[-1,0]>=.2-1e-6 and role not in ['局部重放','资源探针']
        hold=None;holdmean=None
        if T and a[-1,0]>T*(1+1e-6):
            h=a[a[:,0]>T];hold=np.vstack([loading[-1],h]);holdmean=float(np.trapezoid(hold[:,2],hold[:,0])/(hold[-1,0]-T))
        if role=='稀疏日志':status='稀疏接受日志，人工中止'
        elif role=='局部重放':status='仅局部窗口'
        elif role=='资源探针':status='仅准备/资源探针'
        elif complete:status='完整20%'+('+保载' if hold is not None else '')
        elif m.get('type')=='TimeoutError':status='预算停止'
        elif rid=='04-J64':status='步长底线停止'
        elif rid=='04-J17':status='后续数值失败'
        elif rid=='04-Fill':status='填料畸变失败'
        else:status='完成已设定窗口'
        md={'id':rid,'case':'diverse_'+case,'solver':typ+('/Standard' if static else '/Explicit'),'role':role,'T_ms':T*1000 if T else None,'t_mm':t,'N':n,'element':elem,'quadrature_points':qp,'occupancy':occupancy,'interface_width_mm':w,'kernel':kernel,'a_start':float(xy[0,0]),'a_end':float(xy[-1,0]),'recorded_max_at':float(peak[0]),'recorded_max_N':float(peak[1]),'interior_peak_observed':bool(observed),'post_max_drop':float(drop) if drop is not None else None,'end_loading_force_N':float(xy[-1,1]),'hold_mean_N':holdmean,'status':status,'original_failed_checks':'; '.join(failed),'body_minutes':m.get('body_seconds',0)/60 if 'body_seconds' in m else None,'note':note,'source':x['file'],'source_sha256':files[x['file']],'data_signature':sig,'force_kind':'宏观总反力，压缩取正','PBC':'XYZ，宏观横向应变0','E_MPa':10.,'nu':.3,'eta':None if typ=='Abaqus' else 1e-4,'mass':'HRZ' if elem=='HEX27' else ('行和集中质量' if typ=='JAX' else '原Abaqus质量'),'reference_id':None}
        curves[rid]={'meta':md,'raw':a,'loading':loading,'xy':xy,'hold':hold,'rows':rows,'static':static};records.append(md)
    # Canonical shell contexts. Rate mismatch is explicit in every comparison.
    for rid,c in curves.items():
        m=c['meta']
        if m['solver'].startswith('JAX') and m['role'] not in ['资源探针','局部重放','稀疏日志','固定时间表']:
            m['reference_id']=({'04-B64':'04-S1','04-J64':'04-S2'}.get(rid,'04-S4') if m['case']=='diverse_04' else {'28-t45':'28-S45','28-t55':'28-S55'}.get(rid,'28-S40'))
    represented={x[0] for x in SPECS}
    for k,x in enumerate(inv):
        if k not in represented:
            reason='结果内嵌/原失败文件的相同曲线副本'
            if k in [2,3]:reason='r44原预算停止前缀，精确包含于04-B32完整接续，不另算新路径'
            if k in [6,7]:reason='仅零态准备，无非零接受路径'
            if k in [14,20,24,28,29]:reason='均匀或自由波数值控制，无TPMS压缩加载，不作该构型响应'
            omissions.append({'source':x['file'],'id':'','reason':reason})
    # All successful TPMS static points and thickness perturbations, separately.
    points=[]
    def point(rid,case,path,label,N,elem,rule,width=.05,occ='平滑',a=None,bc='XYZ',refpath=None,scope='初始平衡'):
        j=source(R/'validation'/path);k=j.get('stiffness_N_per_mm');aa=j.get('compression',a if a is not None else .0001)
        force=j.get('reaction_amplitude_N',-j.get('Fz_N',j.get('Fz_top_N',0.)))
        if k is None:k=force/(aa*10.)
        ref=source(R/'validation'/refpath) if refpath else None
        kr=ref.get('stiffness_N_per_mm') if ref else None
        fr=(-ref.get('Fz_N',ref.get('Fz_top_N',0))) if ref else None
        points.append({'id':rid,'case':'diverse_'+case,'label':label,'N':N,'element':elem,'quadrature':rule,'t_mm':j.get('thickness_mm',.5),'width_mm':width,'occupancy':occ,'compression':aa,'force_N':force,'K_N_per_mm':k,'shell_K_N_per_mm':kr,'shell_force_N':fr,'relative_to_shell':k/kr-1 if kr else (force/fr-1 if fr else None),'PBC':bc,'status':j.get('status'),'scope':scope,'source':'validation/'+path,'reference_source':'validation/'+refpath if refpath else None})
    b='thin_target_20261004_r5/'
    point('28-Lflat','28',b+'step2/jax.json','旧平端 HEX8',64,'HEX8','8',bc='XY周期+平端',refpath=b+'step2/shell.json',scope='早期小变形平衡')
    point('28-Lxyz','28',b+'step2/diagnostic_xyz/jax.json','早期 XYZ HEX8',64,'HEX8','8',refpath=b+'step2/diagnostic_xyz/shell.json',scope='早期小变形平衡')
    for s in ['a0p010000','a0p050000','a0p100000']:
        point('28-E'+s[3:5],'28',b+'step3_xyz/'+s+'/jax.json','早期 NH 静力 '+s,64,'HEX8','8',refpath=b+'step3_xyz/'+s+'/shell.json',scope='离散有限应变平衡；10%原比较未接受' if '100000' in s else '离散有限应变平衡')
    for sub in ['step4_thickness_a01','step4_thickness_a05']:
        for f in sorted((R/'validation'/b/sub).glob('t*/jax.json')):
            t=read(f).get('thickness_mm');a=read(f).get('compression')
            rp=str((f.parent/'shell.json').relative_to(R/'validation')) if (f.parent/'shell.json').exists() else None
            point('28-FD'+str(t)+'a'+str(a),'28',str(f.relative_to(R/'validation')),'早期厚度扰动',64,'HEX8','8',refpath=rp,scope='厚度FD前向点；不是连续压缩路径')
    ref04='initial_tangent_20261008_r30/abaqus/shell_result.json';ref28='initial_diverse28_20261008_r41/abaqus/shell_result.json'
    point('04-Kbase','04','initial_tangent_20261008_r30/jax_result.json','平滑 w0.05 原27',32,'HEX27','27',refpath=ref04)
    point('04-Kmixed','04','local_reequilibrium_20261008_r34/result.json','平滑 w0.05 局部12³',32,'HEX27','1941区12³/余27',refpath=ref04)
    widths=source(R/'validation/interface_width_20261008_r37/result.json')['cases']
    for z in widths:
        if z['interface_10_90_mm']==.05:continue
        points.append({'id':'04-Kw'+str(z['interface_10_90_mm']),'case':'diverse_04','label':'平滑 w'+str(z['interface_10_90_mm'])+' 原27','N':32,'element':'HEX27','quadrature':'27','t_mm':.5,'width_mm':z['interface_10_90_mm'],'occupancy':'平滑','compression':.0001,'force_N':-z['Fz_N'],'K_N_per_mm':z['K0_N_per_mm'],'shell_K_N_per_mm':read(R/'validation'/ref04)['stiffness_N_per_mm'],'shell_force_N':None,'relative_to_shell':z['relative_to_shell'],'PBC':'XYZ','status':z['status'],'scope':'初始平衡','source':'validation/interface_width_20261008_r37/result.json','reference_source':'validation/'+ref04})
    point('04-Kwmixed','04','interface_width_mixed_20261008_r38/result.json','平滑 w0.025 局部12³',32,'HEX27','1941区12³/余27',width=.025,refpath=ref04)
    point('04-Kbin','04','binary_occupancy_20261008_r40/original27/jax_result.json','二值 原27',32,'HEX27','27',width=0.,occ='二值',refpath=ref04)
    point('04-Kbinmixed','04','binary_occupancy_20261008_r40/mixed12/result.json','二值 局部12³',32,'HEX27','1941区12³/余27',width=0.,occ='二值',refpath=ref04)
    point('28-Kbase','28','initial_diverse28_20261008_r41/jax_result.json','平滑 原27',32,'HEX27','27',refpath=ref28)
    point('28-Kmixed','28','binary_diverse28_20261008_r43/smooth_mixed12/result.json','平滑 局部12³',32,'HEX27','3896区12³/余27',refpath=ref28)
    point('28-Kbin','28','binary_diverse28_20261008_r43/original27/jax_result.json','二值 原27',32,'HEX27','27',width=0.,occ='二值',refpath=ref28)
    point('28-Kbinmixed','28','binary_diverse28_20261008_r43/binary_mixed12/result.json','二值 局部12³',32,'HEX27','3896区12³/余27',width=0.,occ='二值',refpath=ref28)
    # Comparison metrics are retrospective, with actual source coverage.
    errors=[];fixed=[]
    def metric(rid,refid,hi,window):
        c=curves[rid];s=curves[refid];lo=max(c['xy'][0,0],s['xy'][0,0]);end=min(hi,c['xy'][-1,0],s['xy'][-1,0])
        if end<=lo or hi>min(c['xy'][-1,0],s['xy'][-1,0])+1e-6:return
        g=np.linspace(lo,end,2001);yc=np.interp(g,*c['xy'].T);ys=np.interp(g,*s['xy'].T)
        rms=float(np.sqrt(np.mean((yc-ys)**2)));den=s['meta']['recorded_max_N']
        def work(xy):
            xx=xy[(xy[:,0]>lo)&(xy[:,0]<end),0];xx=np.r_[lo,xx,end];return float(10*np.trapezoid(np.interp(xx,*xy.T),xx))
        wc=work(c['xy']);ws=work(s['xy']);m=c['meta'];sm=s['meta']
        errors.append({'id':rid,'case':m['case'],'reference_id':refid,'window':window,'a_start':float(lo),'a_end':float(end),'same_T':m['T_ms']==sm['T_ms'],'same_thickness':m['t_mm']==sm['t_mm'],'T_ms':m['T_ms'],'reference_T_ms':sm['T_ms'],'RMS_N':rms,'reference_full_loading_max_N':den,'RMS_over_reference_max':rms/den,'RMS_over_legacy_2p258914N':rms/2.2589142322540283,'work_N_mm':wc,'reference_work_N_mm':ws,'work_relative':wc/ws-1 if ws else None,'end_force_N':float(yc[-1]),'reference_end_force_N':float(ys[-1]),'end_point_relative':float(yc[-1]/ys[-1]-1) if abs(ys[-1])>1e-8 else None,'scope':'已有路径共同窗口，非新增验收；参考质量缺陷保持'})
    path_errors=[]
    for m in records:
        if not m['reference_id']:continue
        rid=m['id'];ref=m['reference_id']
        for hi in [.05,.1,.12,.15,.2]:metric(rid,ref,hi,f'到{hi*100:g}%')
        if min(m['a_end'],curves[ref]['meta']['a_end'])<.2-1e-6:
            metric(rid,ref,min(m['a_end'],curves[ref]['meta']['a_end']),'全部已覆盖')
        sm=curves[ref]['meta'];pk=m['interior_peak_observed'] and sm['interior_peak_observed']
        hm=m['hold_mean_N'];hs=sm['hold_mean_N'];samet=m['T_ms']==sm['T_ms']
        path_errors.append({'id':rid,'case':m['case'],'reference_id':ref,'same_T':samet,'same_thickness':m['t_mm']==sm['t_mm'],'peak_observed':pk,'peak_position_difference_pp':100*(m['recorded_max_at']-sm['recorded_max_at']) if pk else None,'peak_force_relative':m['recorded_max_N']/sm['recorded_max_N']-1 if pk else None,'hold_mean_N':hm,'reference_hold_mean_N':hs,'hold_mean_relative':hm/hs-1 if hm is not None and hs is not None else None,'hold_scope':'各自实际保载时窗；T不同，不是同物理时窗' if not samet else '匹配T各自保载时窗；仅原记录覆盖','coverage':m['status']})
        for aa in [.01,.05,.1,.12,.14,.15,.16,.18,.2]:
            c=curves[rid]['xy'];s=curves[ref]['xy']
            if aa<max(c[0,0],s[0,0])-1e-9 or aa>min(c[-1,0],s[-1,0])+1e-6:continue
            f=float(np.interp(aa,*c.T));fr=float(np.interp(aa,*s.T));fixed.append({'id':rid,'case':m['case'],'reference_id':ref,'compression':aa,'force_N':f,'reference_force_N':fr,'difference_N':f-fr,'relative_point_difference':f/fr-1 if abs(fr)>1e-8 else None,'difference_over_reference_max':(f-fr)/curves[ref]['meta']['recorded_max_N'],'same_T':m['T_ms']==curves[ref]['meta']['T_ms'],'point_kind':'固定压缩量插值'})
    for rid,ref,scope in [('04-S4','04-S40','壳T对照'),('04-S2','04-S40','壳T对照'),('04-S1','04-S40','壳T对照'),('04-S025','04-S40','壳T对照'),('28-S20','28-S40','壳T对照'),('28-NH4','28-NH8','JAX同核T对照'),('28-C4','28-C8','JAX同核T对照'),('04-B32','04-J18','占据诊断两路径差'),('04-Support','04-J18','虚域支撑诊断差'),('28-Q64','28-NH4','积分及对应质量诊断差')]:
        start=len(errors)
        for hi in [.05,.1,.12,.15,.2]:metric(rid,ref,hi,f'到{hi*100:g}%')
        end=min(curves[rid]['meta']['a_end'],curves[ref]['meta']['a_end'])
        if end<.2-1e-6:metric(rid,ref,end,'全部共同覆盖')
        for e in errors[start:]:e['scope']=scope+'；已有记录回顾，不是新增精度认证'
    for rid,refs in [('04-J15',['04-S40','04-S4']),('04-Rfail',['04-S4']),('04-Rdt',['04-S4'])]:
        m=curves[rid]['meta']
        for aa,f in curves[rid]['xy']:
            for ref in refs:
                s=curves[ref]['xy'];fr=float(np.interp(aa,*s.T))
                fixed.append({'id':rid,'case':m['case'],'reference_id':ref,'compression':float(aa),'force_N':float(f),'reference_force_N':fr,'difference_N':float(f-fr),'relative_point_difference':float(f/fr-1) if abs(fr)>1e-8 else None,'difference_over_reference_max':float((f-fr)/curves[ref]['meta']['recorded_max_N']),'same_T':m['T_ms']==curves[ref]['meta']['T_ms'],'point_kind':'原稀疏接受记录' if rid=='04-J15' else '局部重放接受记录'})
    # Local failure/resource records are kept, even when no valid curve exists.
    for p in sorted((R/'validation').rglob('*.json')):
        if p.name not in ['failure.json','algorithm_stop.json','stop_context.json','resource_failure.json','operator_stop.json']:continue
        rel=str(p.relative_to(R))
        if not any(z in rel for z in ['thin_target_','large_compression_','geometry_transfer_','mechanism_','step_control_','n64_resolution_','binary_n64_']):continue
        j=source(p);omissions.append({'source':rel,'id':'','reason':str(j.get('message',j.get('reason',j.get('error',j.get('status','原停止记录；见来源')))))[:250]})
    for directory,reason in [
        ('simulation_error_20261006_r7','恒厚保形体网格未通过；无独立新增TPMS实体响应路径'),
        ('virtual_kernel_20261006_r9','经典虚域核转动缺陷；未采用，未算独立压缩路径'),
        ('objective_virtual_20261006_r10','纯核/既有场验证；没有新压缩路径'),
        ('critical_mode_20261007_r21','受限谱/细空间残差诊断；不是力-压缩路径'),
        ('fine_mode_20261007_r22','四方向200迭代未收敛；不当作临界点证据'),
        ('background_probe_20261007_r24','参考背景源仅准备，未编译/运行；没有ABAQUS27节点对照曲线'),
        ('initial_bias_mechanism_20261008_r32','已有初始场能量/膜迹后处理，无新压缩路径'),
        ('thickness_kinematics_20261008_r35','穿厚度已有场后处理，无新平衡路径'),
        ('compatible_mode_20261008_r36','固定节点局部模式释放，不是新全局刚度'),
        ('initial_bias_reassessment_20261008_r42','既有状态能量账目后处理，不另算重复仿真'),
        ('binary_mesh_reassessment_20261009_r46','共同窗口复盘，原N32/N64重复使用，不另画成新仿真')]:
        omissions.append({'source':'validation/'+directory+'/','id':'','reason':reason})
    # Every consumed original file is frozen and rechecked at end.
    (O/'sources_before.json').write_text(json.dumps(files,indent=2))
    def export(name,rows):
        if not rows:return
        keys=list(dict.fromkeys(k for z in rows for k in z))
        with (O/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
            wr=csv.DictWriter(f,keys);wr.writeheader();wr.writerows(rows)
        (O/(name+'.json')).write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    export('curve_catalog',records);export('window_errors',errors);export('path_errors',path_errors);export('point_errors',fixed);export('static_points',points);export('coverage_notes',omissions)
    # Plot index and proof of inclusion. Every canonical physical run gets a figure.
    plotted=set();figures=[];figure_objects={}
    palette=['#1764a0','#d05b24','#279180','#8b63a4','#ad2545','#898025','#465b68','#ca7eae']
    def frame(title,sub,panels=2,ids=None):
        fig,axs=plt.subplots(1,panels,figsize=(13.4,8.4),squeeze=False)
        fig.subplots_adjust(left=.065,right=.98,top=.845,bottom=.43,wspace=.22)
        fig.suptitle(title,x=.055,y=.965,ha='left',fontsize=17,fontweight='bold')
        fig.text(.055,.905,sub,fontsize=10,color='#374151')
        return fig,list(axs[0])
    def trace(ax,rid,color,ls=None,window=None,label=None):
        c=curves[rid];m=c['meta'];a=c['xy'];mask=np.ones(len(a),bool)
        if window:mask=(a[:,0]>=window[0])&(a[:,0]<=window[1])
        x=a[mask];lab=label or rid+' '+('壳' if m['solver'].startswith('Abaqus') else 'JAX')+(f" T{m['T_ms']:g}ms" if m['T_ms'] else ' 静力')
        ax.plot(x[:,0]*100,x[:,1],color=color,lw=1.65,ls=ls or ('--' if m['solver'].startswith('Abaqus') else '-'),label=lab)
        if not m['status'].startswith('完整') and not c['static'] and m['role']!='资源探针':ax.plot(a[-1,0]*100,a[-1,1],'x',color=color,ms=7,mew=1.7)
        plotted.add(rid)
        ax.set(xlabel='宏观压缩量 (%)',ylabel='压缩宏观反力 (N)')
    def finish(fig,name,ids,caption,static_rows=None):
        ids=list(dict.fromkeys(ids));tax=fig.add_axes([.04,.048,.94,.235]);tax.axis('off')
        data=[]
        for rid in ids:
            m=curves[rid]['meta'];method=m['element']+(f" N{m['N']}" if m['N'] else '')+(f"/{m['quadrature_points']}点" if m['element'].startswith('HEX') else '')
            fld=(('二值' if m['occupancy']=='二值' else f"w{m['interface_width_mm']:g}")+f" {m['kernel']}") if m['solver'].startswith('JAX') else m['kernel']
            flag='*' if m['original_failed_checks'] else ''
            if rid in ['28-Q64','28-OV']:flag+='†'
            data.append([rid,method,f"{m['T_ms']:g}" if m['T_ms'] else '静力',f"{m['t_mm']:g}",fld,f"{max(0,100*m['a_start']):.3f}–{100*m['a_end']:.3f}",m['status']+flag])
        if not data and name.endswith('_12_initial_static'):
            case=name[:2]
            for p in points:
                if p['case']=='diverse_'+case and p['scope']=='初始平衡':
                    data.append([p['id'],f"HEX27 N32/{p['quadrature']}",'静力',f"{p['t_mm']:g}",('二值' if p['occupancy']=='二值' else f"w{p['width_mm']:g}"),f"{p['compression']*100:g}",'初始平衡'])
        if static_rows:
            for p in static_rows:
                data.append([p['id'],f"{p['element']} N{p['N']}/{p['quadrature']}点",'静力',f"{p['t_mm']:g}",f"w{p['width_mm']:g}",f"{p['compression']*100:g}",p['PBC']])
        if data:
            tab=tax.table(cellText=data,colLabels=['曲线ID','单元/网格/积分','T (ms)','t (mm)','占据/核','记录范围 (%)','记录状态'],colWidths=[.12,.21,.075,.075,.2,.14,.18],loc='center',cellLoc='center')
            tab.auto_set_font_size(False);tab.set_fontsize(8.5)
            # Fit all original condition rows inside the reserved table region.
            for cell in tab.get_celld().values():cell.set_height(min(.17,.94/(len(data)+1)))
            for (i,j),cell in tab.get_celld().items():
                cell.set_edgecolor('#dce1e7');cell.set_linewidth(.35)
                if i==0:cell.set_facecolor('#e9edf3');cell.set_text_props(weight='bold')
                elif i%2==0:cell.set_facecolor('#f7f8fa')
        fig.text(.055,.315,caption,fontsize=9,color='#374151')
        fig.text(.055,.018,'L=10mm，E=10MPa，ν=0.3，背景η=10⁻⁴。×最后接受点；*原筛查未过；†密采样材料域未过。完整=覆盖，非精度认证。',fontsize=9,color='#374151')
        fig.savefig(O/(name+'.png'),dpi=180);fig.savefig(O/(name+'.svg'));figure_objects[name]=fig
        figures.append({'file':name,'title':fig._suptitle.get_text(),'ids':ids,'caption':caption});plt.close(fig)
    def group(name,title,sub,groups,caption):
        ids=list(dict.fromkeys(x for gg in groups for x in gg));fig,axs=frame(title,sub,len(groups))
        for ax,gg in zip(axs,groups):
            for i,rid in enumerate(gg):trace(ax,rid,palette[i%len(palette)])
            ax.legend(fontsize=8.5,loc='best');ax.set_xlim(left=0 if all(curves[z]['xy'][0,0]<.001 for z in gg) else None)
        if name=='04_01_shell_T':
            axs[1].set_xlim(9.5,14.5);axs[1].set_ylim(0,6.3)
            for i,rid in enumerate(groups[1]):
                m=curves[rid]['meta'];x=100*m['recorded_max_at'];y=m['recorded_max_N']
                axs[1].plot(x,y,'o',color=palette[i],ms=4)
                axs[1].annotate(f'{x:.2f}%',(x,y),xytext=(0,9),textcoords='offset points',ha='center',fontsize=8,color=palette[i])
        finish(fig,name,ids,caption)
    group('04_01_shell_T','diverse_04：壳模型的峰位和峰后响应随T改变','同一中面、厚度、NH材料、XYZ周期和20%幅值；仅加载时长变化。右侧单独放大峰附近。',[['04-S40','04-S4','04-S2','04-S1','04-S025'],['04-S40','04-S4','04-S2','04-S1']], '记录最大力位置是动力指标；0.25ms极快路径保留原质量失败。保载时长也随T变化。')
    group('04_02_smooth_history','diverse_04：原控制与状态步长控制的已有路径','JAX平滑N32均为T4ms；r15只保存了稀疏日志，显示记录间折线，不证明中间过程。',[['04-J15','04-J17','04-J18','04-S4','04-S40']], 'r17后续数值失败，r18预算停止；壳40ms只作旧参照。反力误差早于最终中断。')
    group('04_03_binary_pairs','diverse_04：二值N32与N64分别对各自同T壳','左：N32 T4ms完整20%；右：N64 T1ms至13.59%。两幅采用各自匹配加载时长。',[['04-B32','04-J18','04-S4'],['04-B64','04-S1']], 'N64末端仍增载，尚未捕获峰；不能将左右差异解释为单独的网格效应。')
    group('04_04_N64_existing','diverse_04：已有两次N64采用不同占据和T','左：平滑T2ms；右：二值T1ms。均为HEX27/27点，厚度0.5mm。',[['04-J64','04-S2'],['04-B64','04-S1']], '平滑N64触原步长底线；二值N64预算停。两者均未完成20%，也不是单一因素对照。')
    group('04_05_support_fill','diverse_04：低占据支撑和壳填料诊断都未覆盖完整峰区','深虚域能量×0.1，HRZ质量保持；壳填料为另一个近零嵌入诊断，不能视为JAX等价。',[['04-J18','04-Support','04-Fill','04-S4']], '填料曲线最后历史记录8.3365%，最后场8.3500%；不能据峰前重合排除填料的峰区影响。')
    group('04_06_local_replays','diverse_04：首错定位和减步重放仅是局部证据','这些窗口都从既有状态出发，独立显示，不拼接成完整0–20%路径。',[['04-Rfail','04-Rdt']], '首错前有效逐步记录至16.0229%；减步短段至16.2139%。非有限/拒绝状态不绘作有效响应。')
    ps=[p for p in points if p['scope']=='早期小变形平衡']
    fig,axs=frame('diverse_28：早期小变形对照按各自边界分别比较','两组均为HEX8 N64/8点、平滑宽0.05mm；不同边界的壳刚度分别匹配，不共享参考线。',2)
    for ax,p in zip(axs,ps):
        ax.bar([0,1],[p['K_N_per_mm'],p['shell_K_N_per_mm']],color=palette[:2])
        ax.set_xticks([0,1],['JAX背景','匹配Standard壳']);ax.set_ylabel('初始刚度 K (N/mm)')
        ax.set_title(p['PBC']+f"：相对壳 {p['relative_to_shell']*100:+.2f}%",fontsize=11)
        ax.set_ylim(0,1.22*max(p['K_N_per_mm'],p['shell_K_N_per_mm']))
        for k,v in enumerate([p['K_N_per_mm'],p['shell_K_N_per_mm']]):ax.annotate(f'{v:.5f}',(k,v),xytext=(0,5),textcoords='offset points',ha='center',fontsize=10)
    finish(fig,'28_00_early_linear',[],'早期平端与XYZ诊断只解释当时的边界敏感性；本轮不重开压缩边界比较，也不与当前HEX27初始诊断混排。',static_rows=ps)
    group('28_01_shell_T','diverse_28：20/40ms壳与Standard原参考','XYZ周期，t0.5mm；Standard横轴为压缩量，其步时间不是物理加载时长T。',[['28-S40','28-S20','28-STD20']], 'Standard完成20%但原能量/人工能筛查未过，保留作诊断参考；壳不是物理真值。')
    group('28_02_elements','diverse_28：从HEX8到HEX27的历史响应','单元与空间分辨率同时改变：HEX8 N64 vs HEX27 N32。右图只比较同一NH候选的4/8ms。',[['28-H8','28-S40'],['28-NH4','28-NH8','28-S40','28-S20']], '不是单独改变单元类型的严格试验；旧NH材料域和步长/回退记录保持。')
    group('28_03_quadrature_kernel','diverse_28：积分加密未解决材料域问题，C²核改善可推进性','左：同HEX27、27/64点NH；右：旧客观虚域与C²快慢路径。',[['28-NH4','28-Q64','28-S40'],['28-OV','28-C4','28-C8','28-S40']], '64点路径完整但密采样有非正J材料域失败；旧客观虚域混合尾域也失败。曲线接近不消除这些缺陷。')
    group('28_04_thickness','diverse_28：三种厚度的20%路径与各自壳参照','每组厚度相同，JAX T4ms、壳T40ms；这里保留历史不同T的比较事实。',[['28-t45','28-S45'],['28-C4','28-S40'],['28-t55','28-S55']], '0.45/0.50/0.55mm有限范围验证；不能外推任意构型、接触或完整20%设计梯度。')
    group('28_05_gradient_forwards','diverse_28：20%厚度扰动和固定时间表的前向记录','NH，HEX27 N32/27点，T4ms；0.4975/0.5000/0.5025mm。固定时间表重放单独标出。',[['28-Dminus','28-NH4','28-Dplus','28-Dreplay','28-S40']], '这里只展示已有前向曲线。原20%总导数JVP与FD符号失败仍保留，不因曲线相近改判。')
    early=[z for z in ['28-ST1','28-ST5','28-ST10','28-ST1old','28-ST5old','28-ST5mid'] if z in curves]
    fig,axs=frame('diverse_28：早期静力只有三个JAX平衡点','左：壳Standard已有路径/中间版本；右：1%、5%、10%的JAX独立平衡点，未补造JAX连续路径。',2)
    for i,rid in enumerate(early):trace(axs[0],rid,palette[i])
    for p in points:
        if p['id'].startswith('28-E'):
            axs[1].plot(p['compression']*100,p['force_N'],'o',color=palette[0]);axs[1].plot(p['compression']*100,p['shell_force_N'],'s',color=palette[1])
            axs[1].annotate(f"{p['relative_to_shell']*100:+.2f}%",(p['compression']*100,p['force_N']),xytext=(3,7),textcoords='offset points',fontsize=9)
    axs[0].legend(fontsize=8);axs[1].plot([],[],'o',color=palette[0],label='JAX HEX8 N64 平衡点');axs[1].plot([],[],'s',color=palette[1],label='同压缩壳终点');axs[1].legend(fontsize=9);axs[1].set(xlabel='宏观压缩量 (%)',ylabel='压缩反力 (N)',ylim=(0,2.85))
    finish(fig,'28_06_early_static',early,'1%与5%原比较通过初筛，10%原比较和壳质量未接受。中间壳版本不替代最终参考。')
    ids=[z for z in ['28-ST498','28-ST498old','28-ST502','28-ST1'] if z in curves]
    fig,axs=frame('diverse_28：早期厚度梯度验证的前向点','左1%附近、右5%附近；点表示实际已算平衡状态。早期为HEX8 N64、平滑宽0.05mm。',2)
    for ax,aa in zip(axs,[.01,.05]):
        ps=[p for p in points if (p['id'].startswith('28-FD') or p['id'].startswith('28-E')) and abs(p['compression']-aa)<1e-8]
        for p in ps:
            ax.plot(p['t_mm'],p['force_N'],'o',color=palette[0]);
            if p['shell_force_N'] is not None:ax.plot(p['t_mm'],p['shell_force_N'],'s',color=palette[1])
        ax.plot([],[],'o',color=palette[0],label=f'JAX {aa*100:g}%');ax.plot([],[],'s',color=palette[1],label='原独立壳点（有记录时）');ax.legend(fontsize=9);ax.set(xlabel='厚度 (mm)',ylabel='压缩反力 (N)')
    # Tiny Standard perturbation curves are also shown in a separate path panel.
    finish(fig,'28_07_early_thickness',[],'5%没有独立壳厚度导数证据；空缺不记为0。1%旧内存保护/资源中止记录在覆盖索引保留。')
    group('28_08_early_shell_perturbations','diverse_28：1%厚度扰动的全部独立壳路径','Standard有限应变，无物理加载时长T；同中面XYZ周期。',[ids], '这些壳路径为早期厚度扰动参考，不是新的20%厚度范围试验。')
    group('28_09_resource_probes','diverse_28：三个准备探针只覆盖极小压缩','显示实际32步短路径；N2传输探针使用同几何输入，但空间分辨率仅用于软件准备检查。',[['28-P64','28-P32','28-P2']], '极小覆盖曲线不支持真实薄壁精度或20%可行性结论；其他均匀/自由波检查列在未成路径索引。')
    # Hold and inertia are separate axes, rather than force-vs-strain vertical stacks.
    for case,groups in [('04',[['04-S40','04-S4','04-S2','04-S1','04-S025'],['04-B32','04-S4']]),('28',[['28-H8','28-NH4','28-NH8','28-Q64','28-S40'],['28-OV','28-C4','28-C8','28-t45','28-t55','28-S40']])]:
        fig,axs=frame(f'diverse_{case}：20%后的保载振荡单独查看','保载横轴s=(time−T)/T；s=0.1代表各自0.1T，实际时间不同，不能按同一物理保载时长排名。',2)
        for ax,gg in zip(axs,groups):
            for i,rid in enumerate(gg):
                c=curves[rid];h=c['hold'];T=c['meta']['T_ms']/1000
                if h is not None:ax.plot((h[:,0]-T)/T,h[:,2],lw=1.45,color=palette[i],label=rid+f' T{T*1000:g}ms')
            ax.set(xlabel='相对保载时间 (time−T)/T',ylabel='压缩宏观反力 (N)');ax.legend(fontsize=8)
        finish(fig,case+'_10_hold',list(dict.fromkeys(z for gg in groups for z in gg)),'平均力按实际保载时间积分计算，不用一个偶然接近的末端反力代表整个保载响应。')
    ids=['04-B32','04-B64','04-J18','04-J64'];fig,axs=frame('diverse_04：T变化也改变反力的惯性贡献','内部宏观力是在当前动力变形状态上计算，不能把它叫作静态平衡反力。',2)
    for i,rid in enumerate(ids):
        c=curves[rid];rr=[z for z in c['rows'] if z['compression']>=.005 and z['time']<=c['meta']['T_ms']/1000];a=np.array([z['compression'] for z in rr]);f=np.array([-z['Fz_N']+z['internal_macro_Fz_N'] for z in rr]);ke=np.array([z['KE_over_U'] for z in rr]);lab=rid+f" T{c['meta']['T_ms']:g}ms"
        axs[0].plot(a*100,f,color=palette[i],label=lab);axs[1].plot(a*100,ke*100,color=palette[i],label=lab)
    axs[0].set(xlabel='压缩量 (%)',ylabel='总反力−内部宏观力 (N)');axs[0].axhline(0,color='#555',lw=.7);axs[1].set(xlabel='压缩量 (%)',ylabel='动能/内部能量 (%)',yscale='log');axs[0].legend(fontsize=8);axs[1].legend(fontsize=8)
    finish(fig,'04_11_inertia',ids,'瞬时惯性项不是全部速率影响：速率还能改变之前累积的变形路径。低末端KE不代表全过程准静态。')
    for case in ['04','28']:
        ps=[p for p in points if p['case']=='diverse_'+case and p['scope']=='初始平衡'];fig,axs=frame(f'diverse_{case}：初始静力偏差与占据/积分的关系','宏观压缩0.01%，无惯性；K=反力/位移，L10mm对应位移0.001mm。',2)
        assert all(abs(p['shell_K_N_per_mm']-ps[0]['shell_K_N_per_mm'])<1e-10 for p in ps)
        x=np.arange(len(ps));labels=[p['label'] for p in ps];K=[p['K_N_per_mm'] for p in ps];err=[p['relative_to_shell']*100 for p in ps]
        axs[0].bar(x,K,color=palette[0]);axs[0].axhline(ps[0]['shell_K_N_per_mm'],color=palette[1],ls='--',label='匹配Standard壳K');axs[0].legend(fontsize=8)
        axs[1].bar(x,err,color=[palette[0] if p['occupancy']=='平滑' else palette[2] for p in ps]);axs[1].axhline(0,color='#555',lw=.7)
        axs[1].margins(y=.2)
        for ax in axs:ax.set_xticks(x,labels,rotation=35,ha='right',fontsize=8)
        axs[0].set_ylabel('初始刚度 K (N/mm)');axs[1].set_ylabel('(K背景/K壳−1) (%)')
        for i,e in enumerate(err):axs[1].annotate(f'{e:+.2f}%',(i,e),xytext=(0,4 if e>=0 else -12),textcoords='offset points',ha='center',fontsize=8)
        finish(fig,case+'_12_initial_static',[],'二值和界面宽度影响初始K，原27点二值也有积分敏感性。局部12³仅选定区域，未认证全域积分收敛。')
    for case in ['04','28']:
        rids=[m['id'] for m in records if m['reference_id'] and m['case']=='diverse_'+case and m['role'] not in ['梯度扰动']]
        arr=np.full((len(rids),5),np.nan);wins=['到5%','到10%','到12%','到15%','到20%']
        for i,rid in enumerate(rids):
            for j,w in enumerate(wins):
                z=next((z for z in errors if z['id']==rid and z['window']==w),None)
                if z:arr[i,j]=100*z['RMS_over_reference_max']
        fig,axs=frame(f'diverse_{case}：误差必须结合已覆盖窗口读','每格为共同窗口曲线RMS/该参考完整加载最大力。灰格表示未覆盖，不能记作通过或0误差。',1)
        fig.subplots_adjust(left=.25,right=.95)
        ax=axs[0];im=ax.imshow(np.ma.masked_invalid(arr),cmap='YlOrRd',vmin=0,vmax=50,aspect='auto');ax.set_xticks(np.arange(5),wins);ax.set_yticks(np.arange(len(rids)),[rid+' / '+curves[rid]['meta']['reference_id']+(' (T不同)' if curves[rid]['meta']['T_ms']!=curves[curves[rid]['meta']['reference_id']]['meta']['T_ms'] else '') for rid in rids],fontsize=9);ax.grid(False);ax.set_facecolor('#e5e7eb');fig.colorbar(im,ax=ax,label='曲线RMS/参考加载最大力 (%)')
        for i in range(len(rids)):
            for j in range(5):ax.text(j,i,f'{arr[i,j]:.1f}' if np.isfinite(arr[i,j]) else '未覆盖',ha='center',va='center',fontsize=8,color='white' if arr[i,j]>30 else '#202733')
        finish(fig,case+'_13_window_errors',[],'表中另保留RMS原始N、旧固定2.258914N分母、功差和定点力差。新统一复盘口径不覆盖原验收与失败。')
    # PDF chapters are strictly separate by geometry, independent of generation order.
    with PdfPages(O/'TPMS_历史模拟图集.pdf') as pdf:
        for name in sorted(figure_objects):pdf.savefig(figure_objects[name])
    missing=set(curves)-plotted
    assert not missing,missing
    export('figure_index',figures)
    # Retained original gates/results are explicitly separate from new metrics.
    summary={'canonical_curves':len(curves),'static_points':len(points),'figures':len(figures),'window_error_rows':len(errors),'point_error_rows':len(fixed),'all_canonical_curves_plotted':not missing,'new_FEM_Abaqus_AD':0,'source_files':len(files),'source_hashes_unchanged':all(sha(R/p)==h for p,h in files.items()),'shell_T':[{k:m[k] for k in ['id','case','T_ms','recorded_max_at','recorded_max_N','original_failed_checks']} for m in records if m['role']=='壳速率']}
    assert summary['source_hashes_unchanged']
    (O/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False));print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    if '--build' in sys.argv:build()
    else:inventory()

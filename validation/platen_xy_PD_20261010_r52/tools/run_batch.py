"""Windows coordinator: P then D, each one Abaqus/JAX pair, <=100min including analysis."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, shutil, subprocess, sys, time
import numpy as np

B=Path(__file__).resolve().parents[1]
R=B.parents[1]
G=R/'validation/platen_xy_G_20261010_r51'
W=Path(r'C:\Users\xuehu\Documents\Codex\2026-10-01\referenced-chatgpt-conversation-this-is-an')
O=W/'output/r52_platen_xy_PD_20261010'
LINUX='/home/xuehu/projects/tpms_jax'
LB=LINUX+'/validation/platen_xy_PD_20261010_r52'
PY=LINUX+'/.pixi/envs/default/bin/python'
ABQ=r'E:\ABAQUS\2026\Commands\abaqus.bat'
NROOT=Path(r'E:\ABAQUS\2026temp\Abaqus_Work\tpms_jax_abaqus')

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,p)

def wsl(script,kind,seconds):
    return ['wsl.exe','-d','Ubuntu-24.04','--exec','/usr/bin/timeout','--signal=TERM','--kill-after=10s',f'{max(1,int(seconds))}s',
        '/usr/bin/env','PYTHONUNBUFFERED=1','JAX_PLATFORMS=cuda,cpu','XLA_PYTHON_CLIENT_PREALLOCATE=false','OMP_NUM_THREADS=1',
        PY,LB+'/tools/'+script,kind]

def run(cmd,log,limit,receipt):
    if limit<1:raise TimeoutError('No remaining budget for '+log.name)
    start=time.time();timed=False
    with log.open('w',encoding='utf-8') as stream:
        p=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        print(json.dumps({'launch':log.name,'pid':p.pid,'limit_seconds':limit}),flush=True)
        try:code=p.wait(timeout=limit+12)
        except subprocess.TimeoutExpired:
            timed=True;subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],stdout=stream,stderr=stream,creationflags=subprocess.CREATE_NO_WINDOW);code=p.wait()
    result={'command':cmd,'returncode':code,'budget_stop':timed or code in (124,137),'wall_seconds':time.time()-start,
            'started_UTC':datetime.fromtimestamp(start,timezone.utc).isoformat(),'log':str(log)}
    write(receipt,result);return result

def status(kind,stage,active):
    s=read(B/'status.json');s.update(stage=stage,execution_authorized=True,active_jobs=active,latest_kind=kind)
    if kind:s['cases'][kind]['stage']=stage
    write(B/'status.json',s);write(O/'status.json',s)
    if kind:write(B/kind/'status.json',{'kind':kind,'stage':stage,'active_jobs':active,'execution_authorized':True,'new_design_AD':False})

def publish_case(kind):
    E=B/kind;out=O/kind;out.mkdir(exist_ok=True)
    m=read(E/'analysis/metrics.json') if (E/'analysis/metrics.json').exists() else None
    c=read(E/'analysis/costs.json')
    for p in (E/'analysis').glob('*'):
        if p.is_file():shutil.copy2(p,out/p.name)
    for name in ['protocol.json']:
        if (E/'input'/name).exists():shutil.copy2(E/'input'/name,out/name)
    cfg=read(E/'input/protocol.json') if (E/'input/protocol.json').exists() else {}
    if m:
        full=m['paired_full20_and_hold'];passed=m['three_scalar_targets_within_10pct']
        hold=m.get('hold_last0p1ms',{}).get('force_N',{}).get('relative_bias')
        summary=f"{'完整20%及保载完成' if full else '仅完成部分路径'}；全曲线RMS {100*m['curve_RMS_over_new_shell_peak']:.3f}%，加载功差 {100*m['common_loading_work_N_mm']['relative_bias']:+.3f}%"
        if hold is not None:summary+=f"，保载均力差 {100*hold:+.3f}%"
        summary+=f"。三个预定约10%目标{'均达到' if passed else '未全部达到'}。"
        rows='\n'.join(f"| {x['compression']*100:.0f}% | {x['JAX_N']:.6f} | {x['Abaqus_N']:.6f} | {x['relative_bias']*100:+.3f}% |" for x in m['sample_force'] if x['relative_bias'] is not None)
        text=f'''# {kind}共同中面端部压板20%：实际结果

**{summary}** RMS分母为本次{kind}壳加载峰{m['primary_shell_full20_peak_N']:.6f} N，比较到{m['compared_compression_max']*100:.4f}%；RMS不是逐点误差上限。G固定峰仅供附加追溯。

L10 mm、t0.5 mm、NH E10 MPa/ν0.3/密度1e−9 tonne/mm³；JAX N32 HEX27/27点、连续占据、η1e−4、界面0.05 mm、HRZ/objective_void/C²；壳S3R/截面5点。两端从零状态以4 ms平滑压缩至2 mm，保载0.2 ms，XY周期与横向宏观伸缩自由，无Z周期、无壁节点固定点。

沿用G的共同中面端部加载理想化。壳中面Z=0/L节点接触压板，壳截面厚度仍0.5 mm；JAX自身占据端面面积聚合到相同初始节点位置，kn320 MPa/mm，面积守恒。端部节点数{cfg.get('rim_node_count')}，接触参考面积{cfg.get('contact_reference_area_mm2')} mm²。D包含端面相切节点。壳/实体运动学、罚律、接触插值、连续占据和惯性离散仍不同，没有全壁面／自接触；此对照不认证完整有限厚度接触、准静态全过程或设计总梯度。

![力、初段、保载及横向伸缩](force_and_lateral.png)

| 压缩量 | JAX上板力 N | Abaqus上板力 N | 相对偏差 |
| --- | ---: | ---: | ---: |
{rows}

![内能、动能、功平衡及实际J](energy_and_J.png)

实际F/J不夹断，低占据负J能量保留。动能与跳跃单独解释；数值有限、材料域正J及有限保存帧不能证明无自接触。详细步长、回退、能量账目和分段误差见metrics.json；保存态位移采用同一时间插值比较，无刚体对齐、曲线缩放或位移放大。

![实际尺度变形](deformation_10_20pct.png)

本构型实际总墙钟{c['overall_minutes']:.2f} min（100 min目标：{c['within_100min']}）。分项与预算停点见costs.json；若路径不完整，不据局部曲线宣布全20%通过。JAX末有效态与原路径、Abaqus原生INP/ODB保留。正式输入位于`{LINUX}/validation/platen_xy_PD_20261010_r52/{kind}`，原生目录`{NROOT}/r52_{kind}_midsurface_end_20pct_20261010`。
'''
        if not (out/'deformation_10_20pct.png').exists():text=text.replace('![实际尺度变形](deformation_10_20pct.png)','本次没有可比较的10/20%保存态变形图。')
    else:
        text=f"# {kind}实际停点\n\n本次未形成可分析的成对路径。实际成本{c['overall_minutes']:.2f}min，阶段和原因见status.json及checks中的收据；不宣布20%通过。\n"
    (out/'README.md').write_text(text,encoding='utf-8')
    formal=text
    for png in ['force_and_lateral.png','energy_and_J.png','deformation_10_20pct.png']:formal=formal.replace(']('+png+')','](analysis/'+png+')')
    (E/'README.md').write_text(formal,encoding='utf-8')
    return m

def main():
    resume=(B/'checks/preparation_bugfix.json').exists()
    if (B/'checks/batch_started.json').exists():
        assert resume and read(B/'status.json')['jobs_submitted']==0 and read(B/'status.json')['new_solver_datachecks']==0
        assert not (B/'P/input/protocol.json').exists() and not (B/'P/jax').exists(),'Only a pre-solver preparation repair can resume'
    protected=read(B/'checks/planning_receipt.json')['protected_source_sha256']
    assert all(sha(R/p)==v for p,v in protected.items())
    write(B/'checks/source_guard_before.json',protected)
    if not (B/'checks/batch_started.json').exists():write(B/'checks/batch_started.json',{'UTC':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'authorized':True})
    actual=read(B/'protocol_planned.json');actual.update(stage='authorized_execution',execution_authorized=True,authorization='2026-10-10 user: 好的，安排准备好就开始吧')
    write(B/'protocol.json',actual);write(O/'protocol.json',actual)
    summary=[]
    for kind in ['P','D']:
        E=B/kind;native=NROOT/f'r52_{kind}_midsurface_end_20pct_20261010'
        if resume and kind=='P':
            start=read(E/'checks/execution_clock.json')['started_unix']
            assert not native.exists() and not (E/'input/protocol.json').exists()
        else:
            start=time.time();assert not E.exists() and not native.exists(),'Never overwrite previous job'
            for sub in ['checks','input','abaqus','analysis']:(E/sub).mkdir(parents=True)
            write(E/'checks/execution_clock.json',{'started_unix':start,'started_UTC':datetime.fromtimestamp(start,timezone.utc).isoformat(),'overall_budget_seconds':6000})
        status(kind,'preparing',[]);common_error=False;cost={}
        def remaining(reserve=300):return 6000-(time.time()-start)-reserve
        try:
            prep=run(wsl('prepare_case.py',kind,min(600,remaining())),E/'checks/preparation.log',min(600,remaining()),E/'checks/preparation_receipt.json')
            if prep['returncode']:common_error=True;raise RuntimeError('Input preparation failed; see checks/preparation.log')
            native.mkdir();cost['preparation_seconds']=time.time()-start
            abqstart=time.time()
            for action in ['datacheck','solve']:
                status(kind,'abaqus_'+action,['Abaqus '+kind+' '+action])
                s=read(B/'status.json');s['cases'][kind]['Abaqus']=action
                if action=='solve':s['jobs_submitted']+=1
                else:s['new_solver_datachecks']+=1
                write(B/'status.json',s);write(O/'status.json',s)
                cmd=[sys.executable,str(G/'tools/run_abaqus.py'),action,'--native-dir',str(native),'--output-dir',str(E/'abaqus'),
                    '--input',str(E/'abaqus/platen_20pct.inp'),'--config',str(E/'input/protocol.json')]
                limit=min(1500-(time.time()-abqstart),remaining())
                receipt=run(cmd,E/'abaqus'/('driver_'+action+'.log'),limit,E/'checks'/('abaqus_'+action+'_driver.json'))
                if receipt['returncode']:
                    common_error=(action=='datacheck');raise RuntimeError('Abaqus '+action+' failed')
            cost['abaqus_check_solve_seconds']=time.time()-abqstart
            status(kind,'abaqus_extract',[])
            cmd=[sys.executable,str(G/'tools/run_abaqus.py'),'extract','--native-dir',str(native),'--output-dir',str(E/'abaqus'),
                '--config',str(E/'input/protocol.json')]
            ext=run(cmd,E/'abaqus/driver_extract.log',min(150,remaining()),E/'checks/abaqus_extract_driver.json')
            if ext['returncode']:raise RuntimeError('Abaqus force extraction failed')
            frames_script=G/'experiments/midsurface_rim_20pct_20261010/tools/extract_rim20_frames.py'
            shutil.copy2(frames_script,native/'extract_frames.py')
            # This Abaqus Python invocation reads the completed ODB; no solver call.
            cmd=['cmd','/d','/c',ABQ,'python',str(native/'extract_frames.py'),str(native)]
            fr=run(cmd,E/'abaqus/frame_extract.log',min(150,remaining()),E/'checks/frame_extract_driver.json')
            if fr['returncode'] or not (native/'frames.npz').exists():raise RuntimeError('Abaqus frame extraction failed')
            shutil.copy2(native/'frames.npz',E/'abaqus/frames.npz');cost['extraction_seconds']=time.time()-abqstart-cost['abaqus_check_solve_seconds']
            s=read(B/'status.json');s['cases'][kind]['Abaqus']='complete';s['cases'][kind]['JAX']='running';s['jobs_submitted']+=1
            write(B/'status.json',s);status(kind,'jax_running',['JAX '+kind])
            limit=min(3900,remaining());jr=run(wsl('run_jax.py',kind,limit),E/'checks/JAX.log',limit,E/'checks/JAX_driver.json')
            cost['JAX_driver_seconds']=jr['wall_seconds']
            status(kind,'analyzing',[])
            if (E/'jax/result.json').exists() or (E/'jax/path_latest.json').exists():
                ar=run(wsl('analyze_case.py',kind,min(240,remaining(10))),E/'analysis/analysis.log',min(240,remaining(10)),E/'checks/analysis_driver.json')
                if ar['returncode']:raise RuntimeError('Analysis failed; original paths remain saved')
            else:raise RuntimeError('JAX stopped before a valid path was saved')
        except Exception as exc:
            cost['failure']={'type':type(exc).__name__,'message':str(exc),'common_input_error':common_error}
            write(E/'checks/stopped.json',cost['failure'])
            print(json.dumps({'kind':kind,'stopped':str(exc),'common_input_error':common_error}),flush=True)
        cost.update(overall_minutes=(time.time()-start)/60,within_100min=(time.time()-start)<=6000,
            budget_minutes={'overall':100,'JAX':65,'Abaqus_datacheck_solve':25,'preparation_extraction_analysis':10})
        write(E/'analysis/costs.json',cost)
        m=publish_case(kind)
        stage='complete_pair_analyzed' if m and m['paired_full20_and_hold'] else 'partial_or_failed_pair'
        status(kind,stage,[])
        s=read(B/'status.json');s['cases'][kind].update(result=m is not None,
            full20_completed=bool(m and m['paired_full20_and_hold']),three_scalar_targets_within_10pct=bool(m and m['three_scalar_targets_within_10pct']),
            overall_minutes=cost['overall_minutes'],JAX='complete' if m and m['JAX_complete'] else 'partial_or_failed')
        write(B/'status.json',s);write(O/'status.json',s);write(E/'status.json',s['cases'][kind]);write(O/kind/'status.json',s['cases'][kind])
        summary.append({'kind':kind,'metrics':None if m is None else {k:m.get(k) for k in ['paired_full20_and_hold','curve_RMS_over_new_shell_peak','common_loading_work_N_mm','hold_last0p1ms','three_scalar_targets_within_10pct']},'costs':cost})
        print(json.dumps({'kind':kind,'finished':stage,'minutes':cost['overall_minutes'],'metrics':summary[-1]['metrics']}),flush=True)
        if common_error:break
    changed=[p for p,v in protected.items() if sha(R/p)!=v];assert not changed,changed
    write(B/'checks/source_guard_after.json',{'changed':changed,'count':len(protected)})
    write(B/'summary.json',summary);write(O/'summary.json',summary)
    status(None,'batch_finished',[])
    s=read(B/'status.json');s['new_FEM_responses']=sum(1 for case in s['cases'].values() for solver in ['Abaqus','JAX'] if case[solver]=='complete');write(B/'status.json',s);write(O/'status.json',s)
    print('Batch finished; prior sources unchanged; no further tasks launched.',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        write(B/'checks/batch_driver_failure.json',{'type':type(exc).__name__,'message':str(exc),'UTC':datetime.now(timezone.utc).isoformat()})
        status(None,'batch_driver_stopped',[]);raise

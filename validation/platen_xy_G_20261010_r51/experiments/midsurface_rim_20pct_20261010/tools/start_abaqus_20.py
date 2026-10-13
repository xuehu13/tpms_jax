"""Prepare and run only the authorized new Abaqus job, bounded by the pair budget."""
from pathlib import Path
import difflib,hashlib,json,os,shutil,subprocess,sys,time
from datetime import datetime,timezone
W=Path(r'C:\Users\xuehu\Documents\Codex\2026-10-01\referenced-chatgpt-conversation-this-is-an')
R=Path(r'\\wsl.localhost\Ubuntu-24.04\home\xuehu\projects\tpms_jax')
D=R/'validation/platen_xy_G_20261010_r51'
E=D/'experiments/midsurface_rim_20pct_20261010'
N=Path(r'E:\ABAQUS\2026temp\Abaqus_Work\tpms_jax_abaqus\r51_G_midsurface_rim_20pct_20261010')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8');os.replace(q,p)
def main():
    start=time.time();assert not N.exists() and not (E/'checks/execution_clock.json').exists()
    for sub in ['checks','abaqus','tools']:(E/sub).mkdir(exist_ok=True)
    protected=json.loads((D/'checks/organization_20261010/protected_before.json').read_text())
    assert all(sha(R/p)==h for p,h in protected.items())
    write(E/'checks/protected_before.json',protected)
    cfg=json.loads((E/'input/protocol_planned.json').read_text())
    for rel,h in cfg['execution_plan']['source_hashes'].items():assert sha(R/rel)==h
    cfg['stage']='G simplified midsurface-rim platen dynamic comparison to20pct; not finite-width contact or AD certification'
    cfg['source_inp_sha256']=cfg.pop('inp_sha256')
    source=(D/'abaqus/platen_2pct.inp').read_text()
    pairs=[('0.,0.,0.001,1.,0.0012000000000000001,1.','0.,0.,0.004,1.,0.0042,1.'),
           (', 0.0012000000000000001',', 0.0042'),('RP_TOP,3,3,-0.2','RP_TOP,3,3,-2.'),
           ('*Output, field, number interval=12','*Output, field, number interval=84')]
    text=source
    for a,b in pairs:assert text.count(a)==1;text=text.replace(a,b)
    inp=E/'abaqus/platen_20pct.inp';inp.write_text(text,encoding='utf-8')
    cfg['inp_sha256']=sha(inp);write(E/'input/protocol.json',cfg)
    (E/'checks/INP_change.diff').write_text(''.join(difflib.unified_diff(source.splitlines(True),text.splitlines(True),fromfile='original_2pct',tofile='new_rim20')),encoding='utf-8')
    clock={'started_unix':start,'started_UTC':datetime.fromtimestamp(start,timezone.utc).isoformat(),'overall_budget_seconds':5400,'preparation_seconds':time.time()-start}
    write(E/'checks/execution_clock.json',clock)
    status={'stage':'abaqus_datacheck','active_jobs':['Abaqus new rim20'],'new_design_AD':False,'full20_completed':False,'started_unix':start}
    write(E/'status.json',status)
    shutil.copy2(Path(__file__),E/'tools/start_abaqus_20.py')
    for action in ['datacheck','solve','extract']:
        status['stage']='abaqus_'+action;write(E/'status.json',status)
        cmd=[sys.executable,str(D/'tools/run_abaqus.py'),action,'--native-dir',str(N),'--output-dir',str(E/'abaqus'),
             '--input',str(inp),'--config',str(E/'input/protocol.json')]
        remaining=min(1500-(time.time()-start),5400-(time.time()-start)-300)
        if remaining<=0:raise TimeoutError('Abaqus combined25min budget reached')
        with (E/'abaqus'/('driver_'+action+'.log')).open('w',encoding='utf-8') as log:
            p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            print(json.dumps({'action':action,'pid':p.pid,'remaining_s':remaining}),flush=True)
            try:code=p.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],creationflags=subprocess.CREATE_NO_WINDOW,stdout=log,stderr=log)
                p.wait();raise TimeoutError('Combined Abaqus budget reached; new-job tree stopped')
        if code:raise RuntimeError('Abaqus '+action+' failed; see isolated log')
    result=json.loads((E/'abaqus/result.json').read_text());assert result['complete']
    write(E/'checks/abaqus_complete.json',{'wall_seconds_including_preparation':time.time()-start,'complete':True,'new_INP_sha256':sha(inp)})
    status.update(stage='abaqus_complete_waiting_JAX',active_jobs=[],abaqus_complete=True);write(E/'status.json',status)
    print(json.dumps({'Abaqus_complete':True,'elapsed_minutes':(time.time()-start)/60,'remaining_overall_minutes':(5400-time.time()+start)/60}),flush=True)
if __name__=='__main__':
    try:main()
    except Exception as exc:
        write(E/'status.json',{'stage':'abaqus_driver_stopped','active_jobs':[],'full20_completed':False,'error':str(exc),'new_design_AD':False});raise

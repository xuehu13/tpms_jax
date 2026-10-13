"""One JAX-only sequential coordinator; no scheduler or repeated submissions."""
from pathlib import Path
import json,os,shutil,signal,subprocess,sys,time,hashlib
B=Path(__file__).resolve().parents[1];R=B.parents[1]
W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an/output/r56_platen_xy_hex8_20261012')
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def read(p):return json.loads(p.read_text())
def mirror():
    W.mkdir(parents=True,exist_ok=True)
    for p in B.rglob('*'):
        if p.is_file() and (p.suffix in ['.json','.csv','.png','.md','.py']) and '__pycache__' not in str(p):
            d=W/p.relative_to(B);d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,d)
def run(cmd,log,limit,case,stage):
    start=time.time();log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w') as f:
        p=subprocess.Popen(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        s=read(B/'status.json');s.update(stage=stage,current_case=case,active_jobs=[{'pid':p.pid,'command':cmd,'started_unix':start,'budget_s':limit}]);write(B/'status.json',s);mirror()
        while p.poll() is None:
            if time.time()-start>limit+10:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                break
            time.sleep(5)
    return {'returncode':p.returncode,'wall_seconds':time.time()-start}
def main():
    assert not (B/'checks/batch_started.json').exists();start=time.time();write(B/'checks/batch_started.json',{'pid':os.getpid(),'started_unix':start})
    proto=read(B/'protocol.json')
    for i,k in enumerate(proto['cases']):
        s=read(B/'status.json');s.update(queue=proto['cases'][i+1:]);write(B/'status.json',s)
        prep=run([sys.executable,str(B/'tools/setup.py'),k],B/'checks'/f'{k}_preparation.log',300,k,'preparing_HEX8')
        write(B/'checks'/f'{k}_preparation_receipt.json',prep)
        if prep['returncode']==0:
            write(B/k/'checks/execution_clock.json',{'started_unix':time.time()})
            rec=run([sys.executable,str(B/'tools/run_jax.py'),k],B/'checks'/f'{k}_jax.log',2400,k,'HEX8_running')
            write(B/k/'checks/execution_receipt.json',rec)
        else:rec={'returncode':None,'not_started':'preparation_failed'}
        s=read(B/'status.json');s['cases'][k]={'preparation':prep,'JAX':rec};s.update(active_jobs=[]);write(B/'status.json',s)
        if (B/'tools/analyze.py').exists():
            ar=run([sys.executable,str(B/'tools/analyze.py')],B/'checks'/f'{k}_analysis.log',120,k,'analyzing');write(B/'checks'/f'{k}_analysis_receipt.json',ar)
        mirror();print(json.dumps({'case_finished':k,'preparation':prep,'JAX':rec}),flush=True)
    unchanged={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in proto['protected_source_sha256'].items()};assert all(unchanged.values());write(B/'checks/protected_sources_unchanged.json',unchanged)
    s=read(B/'status.json');s.update(stage='all_four_attempts_finished',current_case=None,active_jobs=[],queue=[],batch_seconds=time.time()-start,complete_attempts=True);write(B/'status.json',s);mirror()
    print(json.dumps(s),flush=True)
if __name__=='__main__':main()

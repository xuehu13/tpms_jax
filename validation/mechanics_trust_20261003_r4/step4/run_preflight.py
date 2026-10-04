"""Resource-monitored preflight; no precision/reference job is auto-started."""
from pathlib import Path
import json,os,subprocess,sys,time
P=Path('/home/xuehu/projects/tpms_jax');O=P/'validation/mechanics_trust_20261003_r4/step4';HERE=Path(__file__).resolve().parent
PY=str(P/'.pixi/envs/default/bin/python');plan=json.loads((O/'plan.json').read_text());ledger=[]
def dump(f,v):f.write_text(json.dumps(v,indent=2)+'\n')
def monitor(n,inc=.005,tag=None):
    tag=tag or f'N{n}_d{round(inc*1000):03d}';out=O/(tag+'.json');log=O/(tag+'.log')
    cmd=[PY,str(P/'scripts/finite_strain_gyroid.py'),'--N',str(n),'--increment',str(inc),'--limit','.05','--out',str(out)]
    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','XLA_PYTHON_CLIENT_PREALLOCATE':'false','JAX_PLATFORMS':'cuda'}
    start=time.monotonic();rssmax=0.;gpumax=0.;stop=None
    with log.open('x') as stream:
        proc=subprocess.Popen(cmd,cwd=P,env=env,stdout=stream,stderr=subprocess.STDOUT)
        while proc.poll() is None:
            time.sleep(1)
            status=Path(f'/proc/{proc.pid}/status')
            if status.exists():
                for line in status.read_text().splitlines():
                    if line.startswith('VmRSS:'):rssmax=max(rssmax,int(line.split()[1])/1024**2)
            mem={line.split(':')[0]:int(line.split()[1])/1024**2 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:','MemTotal:'))}
            # Device-total usage is conservative and includes the Windows display.
            if round(time.monotonic()-start)%3==0:
                q=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True)
                if q.returncode==0:gpumax=max(gpumax,float(q.stdout.strip().splitlines()[0])/1024)
            state=out.with_suffix('.current.json')
            current=json.loads(state.read_text()) if state.exists() else None
            if rssmax>12.5:stop='host_RSS_limit'
            elif gpumax>7.2:stop='GPU_used_limit'
            elif mem['MemAvailable']<1.5:stop='host_available_floor'
            elif time.monotonic()-start>1800:stop='path_timeout'
            elif current and time.monotonic()-current['state_start_monotonic']>300:stop='state_timeout'
            if stop:
                proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                break
        code=proc.wait()
    record={'kind':'JAX_resource_preflight' if tag.startswith('N') and 'half' not in tag else 'JAX_half_increment_failure_diagnostic',
        'tag':tag,'N':n,'increment':inc,'exit_code':code,'resource_stop':stop,'wall_seconds':time.monotonic()-start,
        'peak_sampled_host_RSS_GiB':rssmax,'peak_device_total_used_GiB':gpumax,'command':cmd,'log':str(log)}
    if out.exists():
        r=json.loads(out.read_text());record.update(status=r['status'],last_completed_compression=r['rows'][-1]['compression'],failure=r['failure'])
    else:record['status']='resource_stopped' if stop else 'implementation_failure'
    ledger.append(record);dump(O/'execution.json',ledger);print(json.dumps(record),flush=True)
    return record
assert not (O/'execution.json').exists();(O/'run_preflight.py').write_bytes(Path(__file__).read_bytes())
audit=json.loads((P/'validation/mechanics_trust_20261003_r4/step1/audit.json').read_text())
print('AUDIT_KEYS '+str(list(audit)),flush=True)
first=monitor(16)
if first['resource_stop'] or first['status']=='implementation_failure':
    dump(O/'preflight_decision.json',{'status':'stopped','reason':'preflight resource/implementation issue','record':first});sys.exit(2)
second=monitor(32)
if second['status']=='stopped' and second['failure'] and not second['resource_stop']:
    diagnostic=monitor(32,.0025,'N32_half_failure_diagnostic')
    second=diagnostic
decision={'status':'ready_for_precision_lock' if second['status']=='ok' else 'stopped',
    'reason':'No fine-grid accuracy claim from pilots; precision launch requires measured resources' if second['status']=='ok' else 'Preflight + single half-increment diagnostic did not establish robust continuation',
    'records':ledger,'reference_jobs_started':0,'precision_jobs_started':0}
dump(O/'preflight_decision.json',decision);print(json.dumps(decision,indent=2),flush=True)

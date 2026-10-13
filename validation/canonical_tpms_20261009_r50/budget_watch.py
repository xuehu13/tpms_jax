"""Wall-budget guard for only this batch's exact JAX command; no FEM."""
from run_batch import D,FORMAL,LINUX,read,write
from pathlib import Path
import time,subprocess,json,datetime
started=time.time();events=[]
while not (D/'batch_receipt.json').exists():
    try:p=read(D/'progress.json')
    except (OSError,ValueError):time.sleep(2);continue
    task=p.get('task','')
    if task.endswith('_dynamic_jax') and p.get('wall_seconds',0)>=2370:
        kind=task.split('_')[0];receipt=D/kind/'budget_interrupt.json'
        if not receipt.exists():
            code="""import pathlib,os,signal,time,json
base=__BASE__
script=__SCRIPT__
target=base+'/__KIND__/jax'
def owned(p):
 try:
  args=(p/'cmdline').read_bytes().split(b'\\x00')
  return bool(args) and args[0].rsplit(b'/',1)[-1].startswith(b'python') and script.encode() in args and target.encode() in args
 except (OSError,ProcessLookupError):return False
selected=[p for p in pathlib.Path('/proc').iterdir() if p.name.isdigit() and owned(p)]
for p in selected:
 try:os.kill(int(p.name),signal.SIGINT)
 except ProcessLookupError:pass
end=time.monotonic()+10
while time.monotonic()<end and any(owned(p) for p in selected):time.sleep(.5)
killed=[]
for p in selected:
 if owned(p):
  try:os.kill(int(p.name),signal.SIGKILL);killed.append(int(p.name))
  except ProcessLookupError:pass
print(json.dumps({'verified_owned_pids':[int(p.name) for p in selected],'hard_killed_pids':killed}))
"""
            code=code.replace('__BASE__',repr(LINUX)).replace('__SCRIPT__',repr(FORMAL+'/scripts/thin_target_explicit.py')).replace('__KIND__',kind)
            # Match /proc null-delimited argument tokens exactly, not a text
            # search that could select the guard or another research process.
            code=code.replace("b'\\x00'", "bytes([0])")
            start=time.time()
            cp=subprocess.run(['wsl.exe','-d','Ubuntu-24.04','--exec','/usr/bin/python3','-c',code],
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8',errors='replace',
                timeout=25,creationflags=subprocess.CREATE_NO_WINDOW)
            event={'reason':'User40min wall cap; 30s reserve to save accepted state','task':task,
                'observed_wall_seconds':p['wall_seconds'],'started_utc':datetime.datetime.fromtimestamp(start,datetime.timezone.utc).isoformat(),
                'guard_seconds':time.time()-start,'returncode':cp.returncode,'stdout':cp.stdout,'stderr':cp.stderr}
            write(receipt,event);events.append(event);print(json.dumps(event),flush=True)
    time.sleep(2)
write(D/'budget_guard_receipt.json',{'wall_seconds':time.time()-started,'events':events,'new_FEM':0,'new_AD':0})
print('BUDGET_GUARD_DONE',flush=True)

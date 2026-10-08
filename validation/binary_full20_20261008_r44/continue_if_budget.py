"""One waiting resource supervisor; never retries a numerical failure."""
from pathlib import Path
import hashlib,json,os,subprocess,time
R=Path('/home/xuehu/projects/tpms_jax');D=Path(__file__).resolve().parent
first=D/'full_from_zero';pid=json.loads((D/'background_launch.json').read_text())['pid']
def write(name,v):(D/name).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
write('supervisor_state.json',{'status':'waiting_for_original_phase','pid':os.getpid(),'original_pid':pid,'no_manual_solver_interrupt':True})
while not (D/'receipt.json').exists():
 if not Path('/proc',str(pid)).exists():
  write('supervisor_state.json',{'status':'original_ended_without_receipt','no_continuation_started':True});raise SystemExit(1)
 time.sleep(5)
# Receipt is written while the original finally block still owns the lock.
while Path('/proc',str(pid)).exists():
 try:
  if Path('/proc',str(pid),'stat').read_text().split(') ',1)[1][0]=='Z':break
 except FileNotFoundError:break
 time.sleep(1)
if (first/'result.json').exists():
 write('supervisor_state.json',{'status':'original_full_process_complete','no_continuation_needed':True});raise SystemExit(0)
failure=json.loads((first/'failure.json').read_text())
if failure['type']!='TimeoutError' or 'budget' not in failure['message'].lower():
 write('supervisor_state.json',{'status':'numerical_or_other_stop_retained','failure_type':failure['type'],'failure_message':failure['message'],'no_continuation_started':True});raise SystemExit(0)
choice=json.loads((D/'selected_continuation_budget.json').read_text());total=choice['total_budget_seconds']
if total<=10800:
 write('supervisor_state.json',{'status':'selected_total_budget_reached','no_continuation_started':True});raise SystemExit(0)
write('supervisor_state.json',{'status':'launching_exact_budget_checkpoint_continuation','total_budget_seconds':total,'source_failure_retained':True,'no_physical_or_numerical_rule_change':True})
with (D/'continuation.log').open('wb') as log:
 command=[str(R/'.pixi/envs/default/bin/python'),str(D/'resume_budget.py'),'--total-budget-seconds',str(total)]
 process=subprocess.Popen(command,cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,XLA_PYTHON_CLIENT_PREALLOCATE='false',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1'))
 write('continuation_background_launch.json',{'pid':process.pid,'started_unix':time.time(),'command':command,'choice_sha256':hashlib.sha256((D/'selected_continuation_budget.json').read_bytes()).hexdigest()})
 rc=process.wait()
write('supervisor_state.json',{'status':'continuation_process_ended','returncode':rc,'original_budget_stop_retained':True})

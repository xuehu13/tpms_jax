"""One diverse_04 forward using the maintained explicit entry; no retuning."""
from pathlib import Path
import fcntl,hashlib,json,os,shutil,subprocess,sys,time
R=Path('/home/xuehu/projects/tpms_jax');O=R/'validation/geometry_transfer_20261006_r15'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def write(p,x):
    assert not p.exists(),p
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
lock=Path('/tmp/tpms_jax_large_compression.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
cfg=read(O/'input.json');decision=read(O/'step1_decision.json')
assert decision['status']=='matched_inputs_ready_for_single_forward'
assert subprocess.check_output(['git','status','--porcelain'],cwd=R,text=True)==''
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()=='86d55b2d6633dd95f6b99b918dee7d749c3eac13'
out=O/'T0p004';assert not out.exists()
for p,h in cfg['solver_source_sha256'].items():assert sha(R/p)==h,p
for p,h in decision['input_sha256'].items():assert sha(O/p)==h,p
before={p:sha(R/p) for p in subprocess.check_output(['git','ls-files'],cwd=R,text=True).splitlines()}
write(O/'step2_before_manifest.json',before)
local_before={str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file()}
command=[str(R/'.pixi/envs/default/bin/python'),'scripts/thin_target_explicit.py','target',
 '--case',str(O),'--case-input',str(O/'input.json'),'--gauss-field',str(O/'gauss_field.npz'),
 '--material-model','objective_void','--element-degree','2','--cells','32','--quadrature-order','4',
 '--force-batch-cells','2048','--adaptive','--load-time','0.004','--output',str(out)]
env=os.environ.copy();env['JAX_PLATFORMS']='cuda'
write(O/'T0p004_manifest.json',{'command':command,'base_commit':'86d55b2d6633dd95f6b99b918dee7d749c3eac13',
 'main_plan_step':2,'research_question':'Can the fixed forward candidate complete 20% compression on diverse_04?',
 'environment_override':{'JAX_PLATFORMS':'cuda'},'source_sha256':cfg['solver_source_sha256'],
 'preexisting_local_file_sha256':local_before,'launcher_sha256':sha(Path(__file__)),
 'new_full_AD_jobs':0,'no_parameter_scan':True,'accepted_dt_recomputed_by_existing_solver':True})
started=time.perf_counter()
with (O/'T0p004.log').open('x') as log:
    job=subprocess.run(command,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
unchanged=all((R/p).exists() and sha(R/p)==h for p,h in before.items())
local_unchanged=all((O/p).exists() and sha(O/p)==h for p,h in local_before.items())
write(O/'T0p004_receipt.json',{'returncode':job.returncode,'wall_seconds':time.perf_counter()-started,
 'preexisting_tracked_files_unchanged':unchanged,'step1_local_files_unchanged':local_unchanged,
 'result_exists':(out/'result.json').exists(),'failure_exists':(out/'failure.json').exists()})
snap=O/'step2_tools';snap.mkdir(exist_ok=True);shutil.copy2(__file__,snap/'run_jax.py')
assert unchanged and local_unchanged
print((O/'T0p004_receipt.json').read_text());sys.exit(job.returncode)

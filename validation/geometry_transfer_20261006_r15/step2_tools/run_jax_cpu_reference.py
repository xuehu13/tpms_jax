"""Use the existing CPU reference-map option for the frozen CPU Gauss cache."""
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
before=read(O/'step2_before_manifest.json')
assert all(sha(R/p)==h for p,h in before.items())
assert read(O/'T0p004_receipt.json')['returncode']==1
assert not (O/'T0p004/result.json').exists() and not (O/'T0p004/input.json').exists()
assert 'assert np.array_equal(np.asarray(p.physical_quad_points),qp)' in (O/'T0p004.log').read_text()
out=O/'T0p004_cpu_reference';assert not out.exists()
for p,h in cfg['solver_source_sha256'].items():assert sha(R/p)==h,p
for p,h in decision['input_sha256'].items():assert sha(O/p)==h,p
files=[O/p for p in decision['input_sha256']]+[O/'preparation.json',O/'step1_decision.json',O/'band_bracket_diagnosis.json',
 O/'T0p004.log',O/'T0p004_manifest.json',O/'T0p004_receipt.json']
local_before={str(p.relative_to(O)):sha(p) for p in files}
command=[str(R/'.pixi/envs/default/bin/python'),'scripts/thin_target_explicit.py','target',
 '--case',str(O),'--case-input',str(O/'input.json'),'--gauss-field',str(O/'gauss_field.npz'),
 '--geometry-on-cpu','--material-model','objective_void','--element-degree','2','--cells','32','--quadrature-order','4',
 '--force-batch-cells','2048','--adaptive','--load-time','0.004','--output',str(out)]
env=os.environ.copy();env['JAX_PLATFORMS']='cuda,cpu'
write(O/'T0p004_cpu_reference_manifest.json',{'command':command,'base_commit':'86d55b2d6633dd95f6b99b918dee7d749c3eac13',
 'main_plan_step':2,'research_question':'One diverse_04 forward with the unchanged candidate',
 'environment_override':{'JAX_PLATFORMS':'cuda,cpu'},'source_sha256':cfg['solver_source_sha256'],
 'frozen_local_file_sha256':local_before,'launcher_sha256':sha(Path(__file__)),
 'reference_geometry_device':'cpu, same as the step1 cache; compact internal-force/time stepping on default cuda',
 'preceding_attempt':'pre-time-integration exact coordinate-cache assertion failure; kept independently',
 'new_full_AD_jobs':0,'no_physical_or_numerical_parameter_change':True,'no_relaxed_cache_tolerance':True})
started=time.perf_counter()
with (O/'T0p004_cpu_reference.log').open('x') as log:
    job=subprocess.run(command,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
unchanged=all(sha(R/p)==h for p,h in before.items())
local_unchanged=all(sha(O/p)==h for p,h in local_before.items())
write(O/'T0p004_cpu_reference_receipt.json',{'returncode':job.returncode,'wall_seconds':time.perf_counter()-started,
 'preexisting_tracked_files_unchanged':unchanged,'frozen_local_files_unchanged':local_unchanged,
 'result_exists':(out/'result.json').exists(),'failure_exists':(out/'failure.json').exists()})
snap=O/'step2_tools';snap.mkdir(exist_ok=True);shutil.copy2(__file__,snap/'run_jax_cpu_reference.py')
assert unchanged and local_unchanged
print((O/'T0p004_cpu_reference_receipt.json').read_text());sys.exit(job.returncode)

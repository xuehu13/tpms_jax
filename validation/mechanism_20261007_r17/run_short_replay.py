"""One fixed-dt causal diagnostic through the unique formal entry."""
from pathlib import Path
import fcntl,hashlib,json,os,subprocess,time
R=Path('/home/xuehu/projects/tpms_jax');O=R/'validation/mechanism_20261007_r17'
old=R/'validation/geometry_transfer_20261006_r15';out=O/'short_dt_replay'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
lock=Path('/tmp/tpms_jax_large_compression.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
before=json.loads((O/'launch_manifest.json').read_text())
for n,h in before['same_material_source_sha256'].items():assert sha(R/n)==h
assert '20 passed' in (O/'short_replay_tests.log').read_text()
assert not out.exists()
command=[str(R/'.pixi/envs/default/bin/python'),'scripts/thin_target_explicit.py','target',
 '--case',str(old),'--case-input',str(old/'input.json'),'--gauss-field',str(old/'gauss_field.npz'),
 '--geometry-on-cpu','--material-model','objective_void','--element-degree','2','--cells','32','--quadrature-order','4',
 '--force-batch-cells','2048','--load-time','0.004',
 '--replay-state',str(O/'original_diagnostic/last_accepted.npz'),'--replay-dt-factor','0.125','--output',str(out)]
manifest={'command':command,'input_sha256':sha(old/'input.json'),
 'checkpoint_sha256':sha(O/'original_diagnostic/last_accepted.npz'),
 'material_source_sha256':before['same_material_source_sha256'],
 'entry_sha256':sha(R/'scripts/thin_target_explicit.py'),
 'dt_factor':.125,'chosen_from_frozen_frequency_bound':11.692633436245462,
 'same_physical_time_window':True,'new_abaqus_jobs':0,'new_design_AD_jobs':0,
 'tests':'20 passed in 27.72 seconds'}
(O/'short_replay_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
start=time.perf_counter();env=os.environ.copy();env['JAX_PLATFORMS']='cuda,cpu'
with (O/'short_dt_replay.log').open('x') as f:
    child=subprocess.Popen(command,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
    code=child.wait()
receipt={'returncode':code,'wall_seconds':time.perf_counter()-start,
 'result_exists':(out/'result.json').exists(),'failure_exists':(out/'failure.json').exists(),
 'scope':'Only one original rejected time window. Not full 20% simulation.'}
(O/'short_replay_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))

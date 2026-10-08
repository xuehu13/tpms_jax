"""One saved-state/cost diagnostic through the unique formal entry."""
from pathlib import Path
import ast,fcntl,hashlib,json,os,subprocess,time
R=Path('/home/xuehu/projects/tpms_jax');O=R/'validation/step_control_20261007_r18'
old=R/'validation/geometry_transfer_20261006_r15';m=R/'validation/mechanism_20261007_r17'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
lock=Path('/tmp/tpms_jax_large_compression.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert '6 passed' in (O/'bound_tests.log').read_text()
origin=json.loads((m/'launch_manifest.json').read_text())
for n,h in origin['same_material_source_sha256'].items():assert sha(R/n)==h
states=[m/'original_diagnostic'/x for x in ('accepted_a0.1000.npz','accepted_a0.1400.npz','accepted_a0.1590.npz','first_rejection_replay/before_first_invalid.npz')]
states.append(m/'short_dt_replay/field.npz')
out=O/'saved_state_probe';assert not out.exists()
command=[str(R/'.pixi/envs/default/bin/python'),'scripts/thin_target_explicit.py','target',
 '--case',str(old),'--case-input',str(old/'input.json'),'--gauss-field',str(old/'gauss_field.npz'),
 '--geometry-on-cpu','--material-model','objective_void','--element-degree','2','--cells','32','--quadrature-order','4',
 '--force-batch-cells','2048','--load-time','0.004','--stability-states',*[str(s) for s in states],
 '--stability-batch-cells','128','--output',str(out)]
(O/'probe_manifest.json').write_text(json.dumps({'command':command,'state_sha256':{str(s):sha(s) for s in states},
 'case_input_sha256':sha(old/'input.json'),'same_core_source_sha256':origin['same_material_source_sha256'],
 'entry_sha256':sha(R/'scripts/thin_target_explicit.py'),'new_time_advancing_jobs':0,'new_design_AD_jobs':0,'new_abaqus_jobs':0},indent=2)+'\n')
start=time.perf_counter();env=os.environ.copy();env['JAX_PLATFORMS']='cuda,cpu'
with (O/'saved_state_probe.log').open('x') as f:
    process=subprocess.Popen(command,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
    (O/'process.json').write_text(json.dumps({'pid':process.pid,'command':command},indent=2)+'\n')
    code=process.wait()
receipt={'returncode':code,'wall_seconds':time.perf_counter()-start,
 'result_exists':(out/'result.json').exists(),'scope':'Saved-state mechanical tangent bound; no trajectory or design AD'}
(O/'probe_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))

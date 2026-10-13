"""One frozen-input forward attempt; first rejection is localized then stops."""
from pathlib import Path
import fcntl, hashlib, json, os, shutil, subprocess, time
R=Path('/home/xuehu/projects/tpms_jax');O=R/'validation/mechanism_20261007_r17'
W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an/work/mechanism_20261007')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
lock=Path('/tmp/tpms_jax_large_compression.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
old=R/'validation/geometry_transfer_20261006_r15'
before=json.loads((old/'T0p004_cpu_reference/input.json').read_text())
for name,h in before['source_sha256'].items():assert sha(R/name)==h
assert sha(old/'gauss_field.npz')==before['Gauss_field_sha256']
out=O/'original_diagnostic';assert not out.exists()
shutil.copytree(W/'shell_frames',O/'shell_frames')
shutil.copy2(W/'extract_shell_modes.py',O/'extract_shell_modes.py')
command=[str(R/'.pixi/envs/default/bin/python'),'scripts/thin_target_explicit.py','target',
 '--case',str(old),'--case-input',str(old/'input.json'),'--gauss-field',str(old/'gauss_field.npz'),
 '--geometry-on-cpu','--material-model','objective_void','--element-degree','2','--cells','32','--quadrature-order','4',
 '--force-batch-cells','2048','--adaptive','--load-time','0.004','--diagnose-first-failure',
 '--checkpoint-compressions','0.10','0.12','0.14','0.159','--output',str(out)]
manifest={'command':command,'old_physical_input_sha256':sha(old/'input.json'),
 'same_material_source_sha256':before['source_sha256'],'base_commit':'07b2042011fa4fb86a0da17c104fdf127c3b4239',
 'entry_sha256':sha(R/'scripts/thin_target_explicit.py'),
 'new_design_AD_jobs':0,'new_abaqus_jobs':0,
 'only_instrumentation_changed':True,'environment':{'JAX_PLATFORMS':'cuda,cpu'},
 'related_tests':'19 passed: first-failure replay, objective Explicit, void continuation; CPU 24.52 seconds'}
(O/'launch_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
start=time.perf_counter();env=os.environ.copy();env['JAX_PLATFORMS']='cuda,cpu'
with (O/'original_diagnostic.log').open('x') as log:
    process=subprocess.Popen(command,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
    (O/'process.json').write_text(json.dumps({'pid':process.pid,'command':command},indent=2)+'\n')
    code=process.wait()
receipt={'returncode':code,'wall_seconds':time.perf_counter()-start,
 'failure_exists':(out/'failure.json').exists(),'complete_result_exists':(out/'result.json').exists(),
 'first_failure_located':(out/'first_rejection_replay/first_failure.json').exists(),
 'accepted_state_retained':(out/'last_accepted.npz').exists() or (out/'last_valid_field.npz').exists()}
(O/'launch_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))

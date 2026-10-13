"""Exactly one controlled forward, using frozen geometry/material inputs."""
from pathlib import Path
import ast,fcntl,hashlib,json,os,subprocess,time
R=Path('/home/xuehu/projects/tpms_jax');O=R/'validation/step_control_20261007_r18'
old=R/'validation/geometry_transfer_20261006_r15';m=R/'validation/mechanism_20261007_r17'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
lock=Path('/tmp/tpms_jax_large_compression.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert '26 passed' in (O/'control_tests.log').read_text()
origin=json.loads((m/'launch_manifest.json').read_text())
for n,h in origin['same_material_source_sha256'].items():assert sha(R/n)==h
tree=ast.parse((R/'scripts/thin_target_explicit.py').read_text())
before=ast.parse((O/'entry_before.py').read_text())
def methods(t):return {n.name:ast.dump(n,include_attributes=False) for c in t.body if isinstance(c,ast.ClassDef) and c.name=='ExplicitXYZ' for n in c.body if isinstance(n,ast.FunctionDef)}
x,y=methods(tree),methods(before)
assert all(x[k]==v for k,v in y.items() if k!='block')
out=O/'controlled_forward';assert not out.exists()
command=[str(R/'.pixi/envs/default/bin/python'),'scripts/thin_target_explicit.py','target',
 '--case',str(old),'--case-input',str(old/'input.json'),'--gauss-field',str(old/'gauss_field.npz'),
 '--geometry-on-cpu','--material-model','objective_void','--element-degree','2','--cells','32','--quadrature-order','4',
 '--force-batch-cells','2048','--load-time','0.004','--state-step-control','--control-budget-seconds','1500',
 '--stability-batch-cells','128','--checkpoint-compressions','.10','.12','.14','.16','.18','.20','--output',str(out)]
manifest={'command':command,'case_input_sha256':sha(old/'input.json'),'Gauss_field_sha256':sha(old/'gauss_field.npz'),
 'same_core_source_sha256':origin['same_material_source_sha256'],'entry_sha256':sha(R/'scripts/thin_target_explicit.py'),
 'force_mass_acceleration_observables_AST_unchanged':True,'block_update_equations_unchanged_runtime_dt_opt_in':True,
 'tests':'26 passed in 35.50 seconds','new_abaqus_jobs':0,'new_design_AD_jobs':0,
 'scope':'One zero-initial-state controlled forward under predeclared limits; completion does not equal validation'}
(O/'forward_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
start=time.perf_counter();env=os.environ.copy();env['JAX_PLATFORMS']='cuda,cpu'
with (O/'controlled_forward.log').open('x') as log:
    child=subprocess.Popen(command,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
    (O/'forward_process.json').write_text(json.dumps({'pid':child.pid,'command':command},indent=2)+'\n')
    code=child.wait()
receipt={'returncode':code,'wall_seconds':time.perf_counter()-start,
 'result_exists':(out/'result.json').exists(),'failure_exists':(out/'failure.json').exists()}
(O/'forward_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))

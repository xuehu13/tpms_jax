"""Retain and execute postprocessing after the one forward attempt ends."""
from pathlib import Path
import json, os, shutil, subprocess, time
R=Path('/home/xuehu/projects/tpms_jax');O=R/'validation/step_control_20261007_r18'
W=Path(__file__).parent
assert (O/'forward_receipt.json').exists()
env=os.environ.copy();env['JAX_PLATFORMS']='cpu'
for name in ('analyze_control.py','diagnose_control_endpoint.py'):
    target=O/name;shutil.copy2(W/name,target)
    start=time.perf_counter()
    with (O/(target.stem+'.log')).open('x') as log:
        code=subprocess.call([str(R/'.pixi/envs/default/bin/python'),str(target)],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
    receipt={'script':name,'returncode':code,'wall_seconds':time.perf_counter()-start,'time_advancing_jobs':0}
    (O/(target.stem+'_receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
    if code:raise SystemExit(code)

"""Two physical inputs, one maintained explicit entry; one GPU job at a time."""
from pathlib import Path
import argparse, fcntl, hashlib, json, os, subprocess, sys, time
R=Path('/home/xuehu/projects/tpms_jax');O=R/'validation/thickness_range_20261006_r13'
p=argparse.ArgumentParser();p.add_argument('tag',choices=['t0p45','t0p55']);a=p.parse_args()
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
lock=Path('/tmp/tpms_jax_large_compression.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
cfg=read(O/'input.json');prep=read(O/'preparation.json')[a.tag]
for s,h in cfg['solver_source_sha256'].items():assert sha(R/s)==h
assert read(R/'validation/void_continuation_20261006_r12/decision.json')['accepted_for_scoped_forward_validation']
case=O/a.tag;out=case/'T0p004';assert not out.exists()
files=[O/'input.json',O/'preparation.json',R/'validation/large_compression_20261005_r6/quadratic_candidate/gauss_field.npz']
before={str(f):sha(f) for f in files}
command=[str(R/'.pixi/envs/default/bin/python'),'scripts/thin_target_explicit.py','target',
         '--material-model','objective_void','--element-degree','2','--cells','32','--quadrature-order','4',
         '--gauss-field',str(files[-1]),'--thickness-mm',str(prep['thickness_mm']),
         '--adaptive','--load-time','.004','--force-batch-cells','2048','--output',str(out)]
env=os.environ.copy();env['JAX_PLATFORMS']='cuda'
manifest={'command':command,'input_sha256':before,'source_sha256':cfg['solver_source_sha256'],
          'environment_override':{'JAX_PLATFORMS':'cuda'},'new_full_AD_jobs':0,
          'role':'Matched physical thickness, frozen candidate; thickness changes occupancy and mass',
          'launcher_sha256':sha(Path(__file__))}
(case/'T0p004_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
started=time.perf_counter()
with (case/'T0p004.log').open('w') as log:
    job=subprocess.run(command,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
after={str(f):sha(f) for f in files};assert before==after
for s,h in cfg['solver_source_sha256'].items():assert sha(R/s)==h
receipt={'returncode':job.returncode,'wall_seconds':time.perf_counter()-started,'input_unchanged':True,
         'solver_unchanged':True,'result_exists':(out/'result.json').exists(),'failure_exists':(out/'failure.json').exists()}
(case/'T0p004_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2));sys.exit(job.returncode)

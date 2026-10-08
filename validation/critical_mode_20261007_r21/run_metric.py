from pathlib import Path
import json,subprocess,time,shutil,hashlib
R=Path('/home/xuehu/projects/tpms_jax');D=R/'validation/critical_mode_20261007_r21';O=D/'physical_metric'
W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
for name in ['metric_protocol.py','physical_metric.py','run_metric.py']:shutil.copy2(W/'work/critical_mode_20261007'/name,D/name)
protocol=json.loads((O/'protocol.json').read_text());command=[str(R/'.pixi/envs/default/bin/python'),str(D/'physical_metric.py')]
(O/'launch_manifest.json').write_text(json.dumps({'command':command,'source_sha256':hashlib.sha256((D/'physical_metric.py').read_bytes()).hexdigest(),'protocol_sha256':hashlib.sha256((O/'protocol.json').read_bytes()).hexdigest()},indent=2))
start=time.perf_counter();budget=False
with (O/'diagnostic.log').open('x') as f:
 p=subprocess.Popen(command,cwd=R,stdout=f,stderr=subprocess.STDOUT)
 (O/'process.json').write_text(json.dumps({'pid':p.pid,'command':command},indent=2))
 try:code=p.wait(timeout=protocol['budget_seconds'])
 except subprocess.TimeoutExpired:
  budget=True;p.terminate()
  try:code=p.wait(timeout=15)
  except subprocess.TimeoutExpired:p.kill();code=p.wait()
receipt={'returncode':code,'wall_seconds':time.perf_counter()-start,'budget_stop':budget,'result_exists':(O/'result.json').exists()}
(O/'launch_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))

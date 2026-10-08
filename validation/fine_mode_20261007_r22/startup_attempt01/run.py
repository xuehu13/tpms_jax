from pathlib import Path
import json,subprocess,time,shutil,hashlib
R=Path('/home/xuehu/projects/tpms_jax');D=R/'validation/fine_mode_20261007_r22'
W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
for name in ['prepare.py','fine_modes.py','run.py']:shutil.copy2(W/'work/fine_mode_20261007'/name,D/name)
protocol=json.loads((D/'protocol.json').read_text());command=[str(R/'.pixi/envs/default/bin/python'),str(D/'fine_modes.py')]
(D/'launch_manifest.json').write_text(json.dumps({'command':command,'source_sha256':hashlib.sha256((D/'fine_modes.py').read_bytes()).hexdigest(),'protocol_sha256':hashlib.sha256((D/'protocol.json').read_bytes()).hexdigest()},indent=2))
start=time.perf_counter();budget=False
with (D/'diagnostic.log').open('x') as f:
 p=subprocess.Popen(command,cwd=R,stdout=f,stderr=subprocess.STDOUT)
 (D/'process.json').write_text(json.dumps({'pid':p.pid,'command':command},indent=2))
 try:code=p.wait(timeout=protocol['budget_seconds'])
 except subprocess.TimeoutExpired:
  budget=True;p.terminate()
  try:code=p.wait(timeout=15)
  except subprocess.TimeoutExpired:p.kill();code=p.wait()
receipt={'returncode':code,'wall_seconds':time.perf_counter()-start,'budget_stop':budget,'result_exists':(D/'result.json').exists()}
(D/'launch_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))

"""One bounded saved-state diagnostic; no solver trajectory is launched."""
from pathlib import Path
import json,subprocess,time,os,fcntl,hashlib,shutil,platform
R=Path('/home/xuehu/projects/tpms_jax');D=R/'validation/critical_mode_20261007_r21'
W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
lock=Path('/tmp/tpms_jax_large_compression.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (D/'launch_manifest.json').exists()
protocol=json.loads((D/'protocol.json').read_text())
states=R/'validation/step_control_20261007_r18/controlled_forward'
command=[str(R/'.pixi/envs/default/bin/python'),str(D/'diagnose.py')]
manifest={'command':command,'protocol_sha256':sha(D/'protocol.json'),'analysis_source_sha256':sha(D/'diagnose.py'),
 'saved_state_sha256':{name:sha(states/name) for name in protocol['states']},
 'source_sha256':json.loads((D/'source_before.json').read_text()),'new_forward_jobs':0,'new_Abaqus_jobs':0,'design_AD':False}
(D/'launch_manifest.json').write_text(json.dumps(manifest,indent=2))
(D/'environment.json').write_text(json.dumps({'Python':platform.python_version(),'platform':platform.platform(),'backend':'CPU x64','git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()},indent=2))
shutil.copy2(W/'work/critical_mode_20261007/run.py',D/'run.py')
start=time.perf_counter();env=os.environ.copy();env['JAX_PLATFORMS']='cpu';timeout=False
with (D/'diagnostic.log').open('x') as log:
 process=subprocess.Popen(command,cwd=R,stdout=log,stderr=subprocess.STDOUT,env=env)
 (D/'process.json').write_text(json.dumps({'pid':process.pid,'command':command},indent=2))
 try:code=process.wait(timeout=1500)
 except subprocess.TimeoutExpired:
  timeout=True;process.terminate()
  try:code=process.wait(timeout=15)
  except subprocess.TimeoutExpired:process.kill();code=process.wait()
receipt={'returncode':code,'wall_seconds':time.perf_counter()-start,'budget_stop':timeout,'result_exists':(D/'results/result.json').exists(),'partial_result_exists':(D/'results/partial_result.json').exists()}
(D/'launch_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)

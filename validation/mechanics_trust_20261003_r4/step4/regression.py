from pathlib import Path
import json,os,subprocess,time
P=Path('/home/xuehu/projects/tpms_jax');O=P/'validation/mechanics_trust_20261003_r4/step4'
start=time.monotonic();cmd=[str(P/'.pixi/envs/default/bin/python'),'-m','pytest','-q']
env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','JAX_PLATFORMS':'cpu','XLA_PYTHON_CLIENT_PREALLOCATE':'false'}
with (O/'regression_final.log').open('x') as stream:r=subprocess.run(cmd,cwd=P,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=300)
(O/'regression_final.json').write_text(json.dumps({'command':cmd,'exit_code':r.returncode,'wall_seconds':time.monotonic()-start,'maintenance_not_research_paths':True},indent=2)+'\n')
(O/'regression.py').write_bytes(Path(__file__).read_bytes());print((O/'regression_final.log').read_text());raise SystemExit(r.returncode)

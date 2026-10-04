from pathlib import Path
import os,json,subprocess,time
P=Path('/home/xuehu/projects/tpms_jax');O=P/'validation/mechanics_trust_20261003_r4/step3'
env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','XLA_PYTHON_CLIENT_PREALLOCATE':'false','JAX_PLATFORMS':'cpu'}
cmd=[str(P/'.pixi/envs/default/bin/python'),'-m','pytest','tests','-q'];start=time.monotonic()
with (O/'regression_final.log').open('x') as log:
    r=subprocess.run(cmd,cwd=P,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=360)
(O/'regression_final.json').write_text(json.dumps({'exit_code':r.returncode,'command':cmd,'wall_seconds':time.monotonic()-start},indent=2)+'\n')
(O/'final_checks.py').write_bytes(Path(__file__).read_bytes())
print((O/'regression_final.log').read_text(errors='replace')[-2500:])
raise SystemExit(r.returncode)

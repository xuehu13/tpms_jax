from pathlib import Path
import os,subprocess
P=Path('/home/xuehu/projects/tpms_jax');O=P/'validation/mechanics_trust_20261003_r4/step3'
(O/'launch.py').write_bytes(Path(__file__).read_bytes())
with (O/'driver.log').open('x') as stream:
    r=subprocess.run([str(P/'.pixi/envs/default/bin/python'),str(Path(__file__).with_name('run.py'))],cwd=P,
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','XLA_PYTHON_CLIENT_PREALLOCATE':'false','JAX_PLATFORMS':'cpu'},stdout=stream,stderr=subprocess.STDOUT)
print((O/'driver.log').read_text(errors='replace')[-4500:])
raise SystemExit(r.returncode)

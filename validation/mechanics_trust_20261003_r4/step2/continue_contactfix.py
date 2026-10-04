from pathlib import Path
import os,subprocess
P=Path('/home/xuehu/projects/tpms_jax');O=P/'validation/mechanics_trust_20261003_r4/step2'
(O/'summary_before_contactfix.json').write_bytes((O/'summary.json').read_bytes())
(O/'continue_contactfix.py').write_bytes(Path(__file__).read_bytes())
with (O/'driver_contactfix.log').open('x') as stream:
    r=subprocess.run([str(P/'.pixi/envs/default/bin/python'),str(Path(__file__).with_name('run.py')),'--resume','--contactfix'],cwd=P,
                     env={**os.environ,'XLA_PYTHON_CLIENT_PREALLOCATE':'false','PYTHONDONTWRITEBYTECODE':'1'},stdout=stream,stderr=subprocess.STDOUT)
print((O/'driver_contactfix.log').read_text(errors='replace')[-5000:])
raise SystemExit(r.returncode)

"""Install four genuinely new files, preserving all existing modules."""
from pathlib import Path
import os,json,subprocess,hashlib
P=Path('/home/xuehu/projects/tpms_jax');O=P/'validation/mechanics_trust_20261003_r4/step3'
HERE=Path(__file__).resolve().parent
newfiles=[]
for f in (HERE/'newfiles').rglob('*.py'):
    name=f.relative_to(HERE/'newfiles');target=P/name
    assert not target.exists(),target
    target.parent.mkdir(exist_ok=True,parents=True);target.write_bytes(f.read_bytes());newfiles.append(str(name))
(O/'new_source_files.json').write_text(json.dumps(newfiles,indent=2)+'\n')
(O/'install_check.py').write_bytes(Path(__file__).read_bytes())
env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','XLA_PYTHON_CLIENT_PREALLOCATE':'false','JAX_PLATFORMS':'cpu'}
cmd=[str(P/'.pixi/envs/default/bin/python'),'-m','pytest','tests/test_hyperelastic_fem.py','-q']
with (O/'new_physics_tests.log').open('x') as log:
    r=subprocess.run(cmd,cwd=P,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
(O/'new_physics_tests.json').write_text(json.dumps({'exit_code':r.returncode,'command':cmd,'new_sources':{n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in newfiles}},indent=2)+'\n')
print((O/'new_physics_tests.log').read_text(errors='replace')[-4500:])
raise SystemExit(r.returncode)

from pathlib import Path
import subprocess,sys,os

W=Path(__file__).resolve().parents[2];P=Path('/home/xuehu/projects/tpms_jax');O=P/'validation/mechanics_trust_20261003_r4/step2'
for src,dest in [('RESEARCH_PLAN.md','RESEARCH_PLAN.md'),('PROJECT_OVERVIEW.md','RESEARCH_STATUS.md')]:
    target=P/'docs'/dest
    backup=O/'documents_before'/dest;backup.parent.mkdir(parents=True,exist_ok=True);backup.write_bytes(target.read_bytes())
    target.write_text((W/src).read_text().replace('(PROJECT_OVERVIEW.md)','(RESEARCH_STATUS.md)').replace('(NEAR_TERM_REPORT.md)','(../validation/near_term_20261003/README.md)'))
(O/'launch.py').write_bytes(Path(__file__).read_bytes())
with (O/'driver.log').open('x') as stream:
    result=subprocess.run([str(P/'.pixi/envs/default/bin/python'),str(Path(__file__).with_name('run.py'))],cwd=P,
                          stdout=stream,stderr=subprocess.STDOUT,
                          env={**os.environ,'XLA_PYTHON_CLIENT_PREALLOCATE':'false','PYTHONDONTWRITEBYTECODE':'1'})
print((O/'driver.log').read_text()[-5000:],flush=True)
raise SystemExit(result.returncode)

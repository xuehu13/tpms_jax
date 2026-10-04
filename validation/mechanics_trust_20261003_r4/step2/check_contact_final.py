from pathlib import Path
import subprocess,os,json
P=Path('/home/xuehu/projects/tpms_jax');O=P/'validation/mechanics_trust_20261003_r4/step2'
t=P/'tests/test_abaqus_binary.py';old=t.read_text()
needle='    with jax.experimental.enable_x64():\n        np.testing.assert_allclose(primitive(points),expected,atol=1e-14)'
assert old.count(needle)==1
t.write_text(old.replace(needle,'    jax.config.update("jax_enable_x64", True)\n    np.testing.assert_allclose(primitive(points),expected,atol=1e-14)'))
env={**os.environ,'XLA_PYTHON_CLIENT_PREALLOCATE':'false','JAX_PLATFORMS':'cpu','PYTHONDONTWRITEBYTECODE':'1'}
with (O/'contactfix_regression_final.log').open('x') as stream:
    r=subprocess.run([str(P/'.pixi/envs/default/bin/python'),'-m','pytest','tests/test_abaqus_binary.py','-q'],cwd=P,env=env,stdout=stream,stderr=subprocess.STDOUT)
(O/'contactfix_regression.json').write_text(json.dumps({'exit_code':r.returncode,'platform':'cpu','log':'contactfix_regression_final.log','prior_logs_preserved':['contactfix_regression.log','contactfix_regression_retry.log'],'precision':'Primitive test declares x64 through the installed JAX config API'},indent=2)+'\n')
(O/'check_contact_final.py').write_bytes(Path(__file__).read_bytes())
print((O/'contactfix_regression_final.log').read_text())
raise SystemExit(r.returncode)

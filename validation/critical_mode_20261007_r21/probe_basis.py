from pathlib import Path
D=Path('/home/xuehu/projects/tpms_jax/validation/critical_mode_20261007_r21')
t=(D/'diagnose.py').read_text()
t=t[:t.index("write(D/'basis_checks.json'")]
t=t.replace("D.joinpath('results').mkdir(exist_ok=False)","None")
t=t.replace("assert err<2e-12", "print({'N':nc,'error':err,'max_grad':float(np.abs(gf).max()),'relative':err/float(np.abs(gf).max())},flush=True)")
exec(compile(t,str(D/'diagnose.py'), 'exec'))

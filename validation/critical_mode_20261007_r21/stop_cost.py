from pathlib import Path
import json,os,signal
D=Path('/home/xuehu/projects/tpms_jax/validation/critical_mode_20261007_r21')
pid=json.loads((D/'process.json').read_text())['pid']
assert str(D/'diagnose.py').encode() in (Path('/proc')/str(pid)/'cmdline').read_bytes()
assert not (D/'results/result.json').exists()
(D/'cost_stop.json').write_text(json.dumps({'reason':'Full jacfwd and generic einsum implementation exceeded useful cost before first eigendiagnostic. Same-operator JVP columns and BLAS contractions will be verified against original expressions, with the same states/subspaces/eigen tolerances.','not_a_method_or_physics_failure':True,'protocol_scientific_choices_unchanged':True},indent=2))
os.kill(pid,signal.SIGTERM);print('Stopped exact diagnostic process for same-operator cost repair.')

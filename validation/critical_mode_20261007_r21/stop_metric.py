from pathlib import Path
import json,os,signal
D=Path('/home/xuehu/projects/tpms_jax/validation/critical_mode_20261007_r21');O=D/'physical_metric'
x=json.loads((O/'process.json').read_text());pid=x['pid']
assert str(D/'physical_metric.py').encode() in (Path('/proc')/str(pid)/'cmdline').read_bytes()
(O/'operator_stop.json').write_text(json.dumps({'reason':'Generalized smallest-algebraic call has produced no eigenmode within useful remaining cost. Preserve as incomplete; one zero-centered extraction using unchanged K/M and fixed sigma=0 targets critical directions, not a shift fit or global minimum certification.','no_spectrum_yet':not (O/'partial_result.json').exists()},indent=2))
os.kill(pid,signal.SIGTERM);print('Stopped exact mass-metric process; evidence retained.')

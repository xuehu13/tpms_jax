from pathlib import Path
import json,os,signal,time
D=Path('/home/xuehu/projects/tpms_jax/validation/critical_mode_20261007_r21')
x=json.loads((D/'process.json').read_text());pid=x['pid']
assert str(D/'diagnose.py').encode() in Path('/proc') .joinpath(str(pid),'cmdline').read_bytes()
(D/'interface_stop.json').write_text(json.dumps({'reason':'Detected wrong signature/units in future mode interpolation before first eigen result; stop to correct only the postprocessor interface.','no_eigen_result_yet':not bool(list((D/'results').glob('*modes.npz'))),'physical_or_eigen_protocol_changed':False},indent=2))
os.kill(pid,signal.SIGTERM)
print('Stopped exact diagnostic PID for postprocessor interface correction.')

from pathlib import Path
import json,os,signal,time
D=Path('/home/xuehu/projects/tpms_jax/validation/fine_mode_20261007_r22')
info=json.loads((D/'process.json').read_text());pid=info['pid'];cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes()
assert str(D/'fine_modes.py').encode() in cmd
(D/'startup_operator_stop.json').write_text(json.dumps({'reason':'Unexpected long CPU preparation before any material-column or eigensolver output; request KeyboardInterrupt traceback to locate implementation cost. Not a spectrum failure or new physics result.','signal':'SIGINT','pid':pid,'new_time_advance':False},indent=2))
os.kill(pid,signal.SIGINT)
print('Interrupted only own diagnostic process before spectrum; traceback retained in diagnostic.log.')

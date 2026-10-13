from pathlib import Path
import shutil,json
R=Path('/home/xuehu/projects/tpms_jax');D=R/'validation/critical_mode_20261007_r21'
W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
A=D/'mechanical_attempt01';A.mkdir(exist_ok=False)
assert not (D/'results/partial_result.json').exists()
for name in ['diagnose.py','launch_manifest.json','launch_receipt.json','diagnostic.log','process.json','environment.json','basis_checks.json','interface_stop.json']:
 shutil.move(str(D/name),str(A/name))
shutil.move(str(D/'results'),str(A/'results'))
(A/'note.json').write_text(json.dumps({'repair':'Correct existing pure interpolation function signature interpolate(q), supply its original globals, keep mm output units.','stop_before_first_eigen_result':True,'physical_or_eigen_protocol_changed':False},indent=2))
for name in ['diagnose.py','repair_interface.py','stop_for_interface.py']:shutil.copy2(W/'work/critical_mode_20261007'/name,D/name)
print('Partial mechanical attempt retained; original interpolator signature and mm units corrected.')

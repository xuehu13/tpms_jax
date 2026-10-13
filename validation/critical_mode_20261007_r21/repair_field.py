from pathlib import Path
import shutil,json
R=Path('/home/xuehu/projects/tpms_jax');D=R/'validation/critical_mode_20261007_r21'
W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
A=D/'startup_attempt02';A.mkdir(exist_ok=False)
assert not (D/'basis_checks.json').exists() and not (D/'results/partial_result.json').exists()
for name in ['diagnose.py','launch_manifest.json','launch_receipt.json','diagnostic.log','process.json','environment.json']:
 shutil.move(str(D/name),str(A/name))
if (D/'results').exists():shutil.move(str(D/'results'),str(A/'results'))
(A/'note.json').write_text(json.dumps({'problem':'Read wrong surface array key, before material evaluation.','repair':'Use original surface_vertices and surface_triangles fields.','physical_or_solver_protocol_changed':False},indent=2))
shutil.copy2(W/'work/critical_mode_20261007/diagnose.py',D/'diagnose.py')
shutil.copy2(W/'work/critical_mode_20261007/repair_field.py',D/'repair_field.py')
print('Array-key startup failure preserved, no mechanical result discarded.')

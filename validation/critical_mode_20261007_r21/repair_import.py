from pathlib import Path
import shutil,json
R=Path('/home/xuehu/projects/tpms_jax');D=R/'validation/critical_mode_20261007_r21'
W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
A=D/'startup_attempt01';A.mkdir(exist_ok=False)
assert not (D/'results').exists()
for name in ['diagnose.py','launch_manifest.json','launch_receipt.json','diagnostic.log','process.json','environment.json']:
 shutil.move(str(D/name),str(A/name))
(A/'note.json').write_text(json.dumps({'problem':'Missing optional threadpoolctl import; failed before basis or material evaluation.','repair':'Use existing BLAS environment thread limits, no package or shared core change.','physical_or_solver_protocol_changed':False},indent=2))
shutil.copy2(W/'work/critical_mode_20261007/diagnose.py',D/'diagnose.py')
shutil.copy2(W/'work/critical_mode_20261007/repair_import.py',D/'repair_import.py')
print('Startup failure preserved; optional import removed before first mechanical evaluation.')

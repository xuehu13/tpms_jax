from pathlib import Path
import shutil,json
R=Path('/home/xuehu/projects/tpms_jax');D=R/'validation/critical_mode_20261007_r21'
W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
A=D/'startup_attempt04';A.mkdir(exist_ok=False)
for name in ['diagnose.py','launch_manifest.json','launch_receipt.json','diagnostic.log','process.json','environment.json']:
 shutil.move(str(D/name),str(A/name))
(A/'note.json').write_text(json.dumps({'problem':'Resume directory existed because completed N4 evidence was reused. Failure before material evaluation.','repair':'Allow that directory, assert no new partial/final result before execution.','old_science_or_physical_protocol_changed':False},indent=2))
for name in ['diagnose.py','repair_resume_directory.py']:shutil.copy2(W/'work/critical_mode_20261007'/name,D/name)

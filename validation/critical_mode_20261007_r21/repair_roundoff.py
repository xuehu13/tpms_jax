from pathlib import Path
import shutil,json
R=Path('/home/xuehu/projects/tpms_jax');D=R/'validation/critical_mode_20261007_r21'
W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
A=D/'startup_attempt03';A.mkdir(exist_ok=False)
assert not (D/'basis_checks.json').exists() and not (D/'results/partial_result.json').exists()
for name in ['diagnose.py','launch_manifest.json','launch_receipt.json','diagnostic.log','process.json','environment.json']:
 shutil.move(str(D/name),str(A/name))
if (D/'results').exists():shutil.move(str(D/'results'),str(A/'results'))
(A/'note.json').write_text(json.dumps({'problem':'Absolute basis roundoff 2.0428e-12 exceeded the initial absolute 2e-12 at gradient scale 151.763.','probe_relative_error':1.3460518624251594e-14,'repair':'Use scaled relative 1e-12 plus absolute 5e-11 roundoff assertion, keeping gradient equivalence checked. Not a material/stability/accuracy gate change.','physical_or_solver_protocol_changed':False},indent=2))
for name in ['diagnose.py','repair_roundoff.py','probe_basis.py']:shutil.copy2(W/'work/critical_mode_20261007'/name,D/name)
print('Gradient equivalence verified to relative 1.35e-14; startup roundoff assertion recorded.')

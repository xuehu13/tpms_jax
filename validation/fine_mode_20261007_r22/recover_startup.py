from pathlib import Path
import json,shutil
R=Path('/home/xuehu/projects/tpms_jax');D=R/'validation/fine_mode_20261007_r22'
receipt=json.loads((D/'launch_receipt.json').read_text());old=D/'startup_attempt01';old.mkdir(exist_ok=False)
assert not (D/'operator_checks.json').exists() and not (D/'initial_block.npz').exists()
for name in ['fine_modes.py','run.py','diagnostic.log','launch_receipt.json','launch_manifest.json','process.json','startup_operator_stop.json']:
 src=(D/name).resolve();assert src.parent==D.resolve();shutil.move(str(src),str(old/name))
update={'scope':'One pre-spectrum implementation recovery, same original operator/backend/initial modes; no spectrum or backend parameter retry.',
 'previous_attempt_seconds':receipt['wall_seconds'],'remaining_seconds':900-receipt['wall_seconds'],
 'old_phase':'Two material columns printed immediately before SIGINT; no global operator or eigeniteration result.',
 'change':'Add phase timings and replace batch 3x3 generic determinant with the exact triple-product determinant already used in r21; verify against16 generic determinants. No clipping of actual F/J or change to physical guards.',
 'cost_attribution':'Old first-column time alone does not locate the preceding slow instruction. This recovery improves instrumentation; do not claim that traceback proves determinant was the only bottleneck.',
 'no_further_recovery':True,'same_total_900s_launch_budget':True}
(D/'startup_recovery.json').write_text(json.dumps(update,indent=2))
print(json.dumps(update))

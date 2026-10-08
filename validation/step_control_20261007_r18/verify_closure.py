"""Final provenance and reading-document audit; no computation advances time."""
from pathlib import Path
import hashlib,json,shutil,subprocess
W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
R=Path('/home/xuehu/projects/tpms_jax');O=R/'validation/step_control_20261007_r18'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
report=(W/'STEP_CONTROL_PROGRESS.md').read_text()
(R/'docs/STEP_CONTROL_PROGRESS.md').write_text(report.replace('work/step_control_20261007/controlled_response.png','../validation/step_control_20261007_r18/analysis/controlled_response.png'))
(O/'REVIEW.md').write_text(report.replace('work/step_control_20261007/controlled_response.png','analysis/controlled_response.png'))
(R/'docs/RESEARCH_BACKGROUND.md').write_text((W/'RESEARCH_BACKGROUND.md').read_text().replace('(PROJECT_OVERVIEW.md)','(RESEARCH_STATUS.md)'))
frozen=read(O/'frozen_before.json');assert all(sha(R/n)==h for n,h in frozen.items())
manifest=read(O/'forward_manifest.json');assert sha(R/'scripts/thin_target_explicit.py')==manifest['entry_sha256']
assert all(sha(R/n)==h for n,h in manifest['same_core_source_sha256'].items())
assert (R/'docs/RESEARCH_PLAN.md').read_text()==(W/'RESEARCH_PLAN.md').read_text()
assert read(O/'analysis/verification.json')['default_block_AST_exactly_unchanged_after_removing_optional_runtime_dt']
assert '26 passed' in (O/'control_tests.log').read_text()
assert not (O/'controlled_forward/result.json').exists()
pid=read(O/'forward_process.json')['pid'];assert not Path('/proc',str(pid)).exists()
checks=subprocess.run(['git','diff','--check'],cwd=R,text=True,capture_output=True)
assert checks.returncode==0,checks.stdout+checks.stderr
shutil.copy2(Path(__file__),O/'verify_closure.py')
receipt={'old_frozen_files_checked':len(frozen),'old_frozen_files_changed':0,
    'shared_core_source_and_environment_unchanged':True,'live_entry_equals_forward_source_hash':True,
    'default_block_AST_equivalence_verified':True,'related_checks':'26 passed',
    'reading_plan_mirrors_identical':True,'no_full_20pct_result':True,'forward_process_ended':True,
    'git_diff_check_passed':True,'current_jobs':0,'new_design_AD':0,'new_abaqus':0,
    'scope':'Closure/provenance only; does not certify nonlinear trajectory, accuracy or design gradients.'}
(O/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
evidence={str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file() and p.name!='evidence_manifest.json'}
(O/'evidence_manifest.json').write_text(json.dumps({'files':evidence,'scope':'All r18 retained evidence and original failed postprocess attempt; large states/logs remain local.'},indent=2)+'\n')
print(json.dumps(receipt,indent=2))

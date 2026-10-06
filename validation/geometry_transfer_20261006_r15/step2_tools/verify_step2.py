"""Verify provenance and compact active prose, without new physics calculations."""
from pathlib import Path
import ast,hashlib,json,re,shutil,subprocess
R=Path('/home/xuehu/projects/tpms_jax');O=R/'validation/geometry_transfer_20261006_r15'
W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
D=W/'work/geometry_forward_20261006'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def write(p,x):
    assert not p.exists(),p
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
# Keep the readable execution record in STEP2, avoiding duplicate long prose.
for src,dest in [('RESEARCH_PLAN.md','docs/RESEARCH_PLAN.md'),('PROJECT_OVERVIEW.md','docs/RESEARCH_STATUS.md'),('TPMS_RESEARCH_REVIEW.md','docs/TPMS_RESEARCH_REVIEW.md')]:
    original=(W/src).read_text().rstrip()+'\n'
    (W/src).write_text(original)
    text=original.replace('PROJECT_OVERVIEW.md','RESEARCH_STATUS.md').replace('output/figures/','figures/')
    if 'REVIEW' in src:text=text.replace('figures/VOID_CONTINUATION_response.png','../validation/void_continuation_20261006_r12/response.png')
    (R/dest).write_text(text)
before=read(O/'step2_before_manifest.json')
allowed={'AGENTS.md','README.md','docs/FILE_MAP.md','docs/RESEARCH_BACKGROUND.md','docs/RESEARCH_PLAN.md',
 'docs/RESEARCH_STATUS.md','docs/TPMS_RESEARCH_REVIEW.md','scripts/README.md','validation/README.md'}
changed={p for p,h in before.items() if not (R/p).exists() or sha(R/p)!=h}
assert changed==allowed,changed
old_local=read(O/'T0p004_manifest.json')['preexisting_local_file_sha256']
assert all(sha(O/p)==h for p,h in old_local.items())
assert read(O/'preparation.json')['step1_input_ready'] is False
assert read(O/'step1_decision.json')['status']=='matched_inputs_ready_for_single_forward'
cfg=read(O/'input.json')
for p,h in cfg['solver_source_sha256'].items():assert sha(R/p)==h,p
source=Path(cfg['reused_screen']['source_directory'])
for p,h in cfg['original_source_sha256'].items():assert sha(source/p)==h,p
summary=read(O/'step2_summary.json')
assert not summary['step2_successful_completion'] and not summary['new_geometry_accuracy_accepted']
assert summary['new_Abaqus_jobs']==1 and summary['new_JAX_time_paths_started']==1
assert summary['new_full_AD_jobs']==summary['new_training_jobs']==summary['extra_slow_jobs']==0
assert read(O/'T0p004_cpu_reference_receipt.json')['preexisting_tracked_files_unchanged']
native=O/'abaqus/explicit_T0p040/results'
records=read(native/'retention.json')['files']
for name,r in records.items():
    assert sha(Path(r['native_path']))==r['sha256'],name
    if (native/name).exists():assert sha(native/name)==r['sha256'],name
for p in list(D.glob('*.py'))+[R/'scripts/thin_target_explicit.py']:
    ast.parse(p.read_text(),filename=str(p))
links=0
wdocs=['RESEARCH_BACKGROUND.md','RESEARCH_PLAN.md','PROJECT_OVERVIEW.md','FILE_MAP.md','TPMS_RESEARCH_REVIEW.md','START_HERE.md','AGENTS.md','work/README.md','work/geometry_forward_20261006/README.md']
rdocs=list(allowed)+['validation/geometry_transfer_20261006_r15/STEP2.md']
for root,docs in [(W,wdocs),(R,rdocs)]:
    for name in docs:
        p=root/name
        if p.suffix!='.md':continue
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            target=target.strip('<>').split('#',1)[0]
            if not target or re.match(r'^[a-zA-Z]+:',target):continue
            assert (p.parent/target).resolve().exists(),(p,target)
            links+=1
subprocess.run(['git','diff','--check'],cwd=R,check=True)
write(O/'step2_verification.json',{'all_old_tracked_scientific_files_unchanged':True,
 'old_tracked_validation_file_count':sum(p.startswith('validation/') and p!='validation/README.md' for p in before),
 'all_preexisting_r15_local_files_unchanged':True,'original_user_source_unchanged':True,
 'solver_and_environment_lock_unchanged':True,'original_native_job_and_ODB_preserved':True,
 'changed_preexisting_paths':sorted(changed),'active_document_links_verified':links,'syntax_and_diff_check_pass':True,
 'no_completed_JAX20_or_accuracy_or_gradient_claim':True,'only_one_time_path_and_one_shell_job':True,
 'scope':'Evidence/provenance verification, no new tests of material or simulation and no new calculations'})
shutil.copy2(__file__,O/'step2_tools/verify_step2.py')
newfiles={str(p.relative_to(O)):sha(p) for p in sorted(O.rglob('*')) if p.is_file() and str(p.relative_to(O)) not in old_local and p.name!='step2_evidence_manifest.json'}
write(O/'step2_evidence_manifest.json',{'new_files_including_local_ignored_fields_and_logs':newfiles,
 'old_scientific_input_files_preserved_manifest':'T0p004_manifest.json',
 'native_records':'abaqus/explicit_T0p040/results/retention.json','source_unchanged':True,
 'complete_20pct_JAX_path_retained':False,'no_everywhere_validity_or_gradient_certification':True})
print(json.dumps({'verified':True,'old_science':sum(p.startswith('validation/') and p!='validation/README.md' for p in before),'links':links,'new_local_files':len(newfiles)}))

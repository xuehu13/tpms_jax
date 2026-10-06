"""Bounded input/provenance verification; no FEM rebuild or displacement solve."""
from pathlib import Path
import ast,hashlib,json,re,shutil,subprocess

R=Path('/home/xuehu/projects/tpms_jax')
W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
D=W/'work/geometry_transfer_20261006'
O=R/'validation/geometry_transfer_20261006_r15'
P=Path('/mnt/e/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus/geometry_transfer_20261006_r15_diverse04_explicit_T0p040')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def write(p,x):
    assert not p.exists(),p
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

before=read(O/'before_manifest.json')
allowed={'AGENTS.md','README.md','docs/FILE_MAP.md','docs/RESEARCH_BACKGROUND.md',
         'docs/RESEARCH_PLAN.md','docs/RESEARCH_STATUS.md','docs/TPMS_RESEARCH_REVIEW.md',
         'scripts/README.md','scripts/thin_target_explicit.py','validation/README.md'}
changed={p for p,h in before.items() if not (R/p).exists() or sha(R/p)!=h}
assert changed==allowed,changed
cfg=read(O/'input.json');prep=read(O/'preparation.json');decision=read(O/'step1_decision.json')
diag=read(O/'band_bracket_diagnosis.json')
for p,h in read(O/'report_recovery_receipt.json')['all_preexisting_inputs_and_native_package_sha256'].items():
    assert sha(Path(p))==h,p
for p,h in decision['input_sha256'].items():assert sha(O/p)==h,p
for p,h in decision['source_sha256'].items():assert sha(R/p)==h,p
assert sha(O/'preparation.json')==decision['preparation_sha256']
assert sha(O/'band_bracket_diagnosis.json')==decision['targeted_diagnosis_sha256']
assert not prep['step1_input_ready'] and not prep['checks']['local_normal_band_brackets']
assert all(v for k,v in prep['checks'].items() if k!='local_normal_band_brackets')
assert diag['all_failed_endpoints_have_first_exit_then_reentry']
assert decision['status']=='matched_inputs_ready_for_single_forward'
assert decision['main_plan_steps_completed']==[1]
assert all(decision[k]==0 for k in ('new_displacement_solves','new_Abaqus_jobs','new_full_AD_jobs'))
assert not decision['gradient20_certified']
original=Path(cfg['reused_screen']['source_directory'])
for p,h in cfg['original_source_sha256'].items():assert sha(original/p)==h,p
for p in ('input.json','thin_shell.inp','extract_thin_explicit.py'):
    assert sha(P/p)==sha(O/'abaqus/explicit_T0p040'/p),p
assert sorted(p.name for p in P.iterdir())==['extract_thin_explicit.py','input.json','thin_shell.inp']
assert not list(O.rglob('*.odb')) and not list(O.glob('T0*/result.json'))
shell=read(P/'input.json')
for k in ('cell_size_mm','thickness_mm','E_MPa','nu','solid_density_tonne_per_mm3','mechanical_periodic_axes'):
    assert cfg[k]==shell[k],k
assert shell['load_time_seconds']==cfg['shell_load_time_seconds']==.04
assert shell['hold_time_seconds']==.004 and cfg['JAX_load_time_seconds']==.004
assert shell['material']['C10_MPa']==cfg['E_MPa']/(4*(1+cfg['nu']))
assert abs(shell['material']['D1_per_MPa']-6*(1-2*cfg['nu'])/cfg['E_MPa'])<1e-14
for p in [R/'scripts/thin_target_explicit.py',*D.glob('*.py')]:ast.parse(p.read_text(),filename=str(p))

# Link checks cover changed active prose only; frozen historical files are untouched.
wdocs=['RESEARCH_BACKGROUND.md','RESEARCH_PLAN.md','PROJECT_OVERVIEW.md','FILE_MAP.md',
       'TPMS_RESEARCH_REVIEW.md','START_HERE.md','AGENTS.md','work/README.md',
       'work/geometry_transfer_20261006/README.md']
rdocs=[p for p in allowed if p.endswith('.md')]+['validation/geometry_transfer_20261006_r15/README.md']
links=[]
for root,names in [(W,wdocs),(R,rdocs)]:
    for name in names:
        p=root/name
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            target=target.strip('<>').split('#',1)[0]
            if not target or re.match(r'^[a-zA-Z]+:',target):continue
            dest=(p.parent/target).resolve()
            assert dest.exists(),(p,target)
            links.append([str(p),target])
subprocess.run(['git','diff','--check'],cwd=R,check=True)
old_science=sum(p.startswith('validation/') and p!='validation/README.md' for p in before)
write(O/'verification.json',{
 'scope':'input readiness only; no new physics runs',
 'base_commit':cfg['base_commit'],'changed_preexisting_paths':sorted(changed),
 'all_other_preexisting_tracked_files_unchanged':True,'tracked_frozen_validation_files_verified':old_science,
 'raw_inputs_first_logs_attempt_sources_and_native_package_unchanged':True,
 'original_user_surface_and_case_unchanged':True,'source_hashes_match_declared_input':True,
 'matched_physical_constants_and_distinct_quasistatic_load_times_verified':True,
 'original_false_preparation_and_separate_targeted_decision_preserved':True,
 'metadata_reader_class_and_physical_suffix_checks':'see step1_decision.json; no material/HRZ/advance changes',
 'syntax_pass':True,'changed_active_document_links_verified':len(links),'git_diff_check_pass':True,
 'new_displacement_solves':0,'new_Abaqus_jobs':0,'new_full_AD_jobs':0,
 'not_a_new_full_test_suite_or_global_geometry_certification':True})
shutil.copy2(__file__,O/'tools/verify_inputs.py')
# Intermediate print-fix copy is kept locally as provenance, not another maintained tool.
with (O/'.gitignore').open('a') as f:f.write('tools/prepare_inputs.py\n')
manifest={str(p.relative_to(O)):sha(p) for p in sorted(O.rglob('*')) if p.is_file()
          and p.name not in {'evidence_manifest.json'} and not p.name.endswith('.log')}
write(O/'evidence_manifest.json',{
 'scope':'new r15 files, including local ignored arrays/snapshots; excludes live logs and itself',
 'files':manifest,'original_logs':'frozen hashes in report_recovery_receipt.json',
 'native_package':{p.name:sha(p) for p in sorted(P.iterdir()) if p.is_file()},
 'new_solve_count':0})
print(json.dumps({'verified':True,'frozen_tracked_science':old_science,'document_links':len(links),
                 'manifest_files':len(manifest),'next_only':'step2'}))

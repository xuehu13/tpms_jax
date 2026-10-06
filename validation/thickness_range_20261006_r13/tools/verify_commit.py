"""Freeze the new evidence, verify old science, and commit only this round."""
from pathlib import Path
import hashlib, json, re, shutil, subprocess
R=Path('/home/xuehu/projects/tpms_jax');O=R/'validation/thickness_range_20261006_r13'
W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
D=W/'work/thickness_range_20261006';BASE='dd66775a58fe329ae9f0d11ee2b333d08362093f'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def git(*args):return subprocess.check_output(['git',*args],cwd=R,text=True).strip()
assert git('rev-parse','HEAD')==BASE and git('branch','--show-current')=='main'
allowed=['.gitignore','AGENTS.md','README.md','docs/RESEARCH_BACKGROUND.md','docs/RESEARCH_PLAN.md',
         'docs/RESEARCH_STATUS.md','docs/FILE_MAP.md','docs/TPMS_RESEARCH_REVIEW.md','scripts/README.md','validation/README.md']
old=read(O/'before_manifest.json')
unchanged={}
for s,h in old.items():
    if s not in allowed:
        assert sha(R/s)==h,s;unchanged[s]=h
core=read(O/'input.json')['solver_source_sha256']
for s,h in core.items():assert sha(R/s)==h,s
prior=read(R/'validation/void_continuation_20261006_r12/evidence_manifest.json')
for key in ['frozen_input_sha256','new_evidence_sha256','active_code_sha256']:
    for s,h in prior[key].items():assert sha(R/s)==h,s

metrics={tag:read(O/tag/'analysis/comparison.json') for tag in ['t0p45','t0p55']}
for tag,m in metrics.items():
    assert m['candidate_completed_20']
    assert read(O/tag/'T0p004_receipt.json')['returncode']==0
    assert read(O/tag/'abaqus/explicit_T0p040/shell.json')['checks']['complete']
    # Gates are reported, not loosened or inferred from completed execution.
    assert m['scoped_engineering_response_pass']==bool(m['response_target_only_pass'] and
        m['material_domain_sampled_pass'] and m['JAX_energy_inertia_gates_pass'])
    prep=read(O/'preparation.json')[tag];src=Path(prep['native_job_directory'])
    assert sha(src/'thin_shell.inp')==prep['deck_sha256']
    assert sha(src/'input.json')==prep['input_sha256']
    for name,rec in read(O/tag/'abaqus/explicit_T0p040/retention.json')['files'].items():
        assert sha(Path(rec['native_path']))==rec['sha256'],rec['native_path']
for name in ['setup.py','run_path.py','collect_shell.py','analyze.py','finalize.py','update_docs.py','verify_commit.py']:
    p=D/name;compile(p.read_text(),str(p),'exec');shutil.copy2(p,O/'tools'/name)

missing=[]
newdocs=['docs/THICKNESS_RANGE_PROGRESS.md','validation/thickness_range_20261006_r13/README.md']
for s in [p for p in allowed if p.endswith('.md')]+newdocs:
    p=R/s
    for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
        if target.startswith(('http://','https://','#','codex:','chatgpt-')):continue
        if not (p.parent/target.split('#')[0]).resolve().exists():missing.append((s,target))
assert not missing,missing
deliberate=allowed+['docs/THICKNESS_RANGE_PROGRESS.md','docs/figures/THICKNESS_RANGE_response.png',
                   'validation/thickness_range_20261006_r13']
for line in subprocess.check_output(['git','status','--porcelain'],cwd=R,text=True).splitlines():
    s=line[3:];assert s in deliberate or any(s.startswith(p+'/') for p in deliberate),line
result={'old_tracked_files_preserved':len(unchanged),'solver_source_unchanged':True,
        'prior_evidence_unchanged':True,'matched_thickness_inputs_verified':True,
        'canonical_shell_deck_change_only':True,'new_complete_JAX_paths':2,'new_complete_Abaqus_jobs':2,
        'new_full_AD_jobs':0,'link_check_passed':True,'helper_syntax_passed':True,
        'response_and_domain_gates_reported_separately':True,'core_tests_reused':'No solver changes; r12 45 targeted checks unchanged.',
        'per_case_pass':{tag:m['scoped_engineering_response_pass'] for tag,m in metrics.items()},
        'strict_shell_quality_pass':{tag:m['reference_quality_certified'] for tag,m in metrics.items()}}
write(O/'verification.json',result);write(D/'verification.json',result)
evidence={str(p.relative_to(R)):sha(p) for p in sorted(O.rglob('*')) if p.is_file()
          and p.name!='evidence_manifest.json' and '__pycache__' not in p.parts}
write(O/'evidence_manifest.json',{'new_evidence_sha256':evidence,'active_solver_sha256':core,
     'old_tracked_files_verified_unchanged':len(unchanged),'prior_manifest_sha256':sha(R/'validation/void_continuation_20261006_r12/evidence_manifest.json')})
subprocess.run(['git','diff','--check'],cwd=R,check=True)
subprocess.run(['git','add','--',*deliberate],cwd=R,check=True)
stat=git('diff','--cached','--stat');(D/'staged_stat.txt').write_text(stat+'\n')
for s in git('diff','--cached','--name-only').splitlines():
    assert not any(part.startswith('source_') for part in Path(s).parts),s
    assert Path(s).suffix not in ['.npz','.log','.odb','.sta','.dat','.msg','.pdf'],s
    assert Path(s).name not in ['progress.json','extract_thin_explicit.py'],s
subprocess.run(['git','commit','-m','Validate frozen thin TPMS method at two matched wall thicknesses'],cwd=R,check=True)
commit=git('rev-parse','HEAD');assert not git('status','--porcelain')
bundle=D/'tpms_jax_cleanup.bundle';assert not bundle.exists()
subprocess.run(['git','bundle','create',str(bundle),'main','^'+BASE],cwd=R,check=True)
write(D/'local_commit.json',{'base':BASE,'commit':commit,'bundle_sha256':sha(bundle),
      'local_status':'clean','new_JAX_paths':2,'new_Abaqus_jobs':2,'new_full_AD_jobs':0,
      'solver_changed':False,'all_new_cases_pass_scoped_targets':read(O/'decision.json')['all_new_cases_meet_scoped_engineering_targets']})
print(json.dumps(result,indent=2));print(commit)

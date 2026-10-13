from pathlib import Path
import json,hashlib,ast,re,shutil,subprocess
import numpy as np
R=Path('/home/xuehu/projects/tpms_jax');D=R/'validation/fine_mode_20261007_r22'
W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
read=lambda p:json.loads(p.read_text())
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
frozen=read(D/'frozen_before.json');assert len(frozen)==490
assert all(sha(R/x['path'])==x['sha256'] for x in frozen)
assert all(sha(R/path)==value for path,value in read(D/'source_before.json').items())
assert all(sha(R/path)==value for path,value in read(D/'input_manifest.json').items())
manifest=read(D/'launch_manifest.json')
assert sha(D/'fine_modes.py')==manifest['source_sha256'] and sha(D/'protocol.json')==manifest['protocol_sha256']
receipt=read(D/'launch_receipt.json');first=read(D/'startup_attempt01/launch_receipt.json')
assert receipt['returncode']==0 and receipt['result_exists'] and not receipt['budget_stop']
assert first['returncode']==-2 and not first['result_exists']
assert abs(receipt['previous_attempt_seconds']-first['wall_seconds'])<1e-8
assert receipt['wall_seconds']+first['wall_seconds']<900
checks=read(D/'operator_checks.json')
assert checks['free_dofs']==786429 and checks['same_original_mass_relative_error']<1e-11
assert checks['projected_action_vs_original_N8_K_relative_error']<1e-9 and checks['H_JVP_vs_jacfwd_relative_error']<1e-11
assert checks['material_H_asymmetry']<1e-11 and checks['original_positive_J_min']>0
result=read(D/'result.json');assert len(result['rows'])==4
for row in result['rows']:
 assert row['direct_curvature_relative_error']<1e-6
 assert np.isfinite(list(row['parts_N_mm'].values())).all()
 assert abs(sum(row['parts_N_mm'].values())-row['curvature_for_1mm_max_surface_mode_N_mm'])<1e-10
 assert row['mode_converged']==(row['mass_whitened_relative_residual']<=1e-3 and row['original_fine_relative_residual']<=1e-3)
assert not any(x['mode_converged'] for x in result['rows'])
with np.load(D/'refined_modes.npz') as f:
 assert f['original_periodic_modes'].shape==(262144,3,4) and np.isfinite(f['original_periodic_modes']).all()
 assert np.array_equal(f['original_periodic_modes'][0],np.zeros((3,4)))
 assert np.array_equal(f['eigenvalues'],[x['eigenvalue_s_minus2'] for x in result['rows']])
 assert np.max(np.abs(f['mass_whitened_vectors'].T@f['mass_whitened_vectors']-np.eye(4)))<1e-8
assert 'UR' in read(D/'shell_data_inventory.json')['actual_frame_field_outputs_recorded_in_r20']
tree=ast.parse((W/'work/fine_mode_20261007/close.py').read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='formal_text')
ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'formal_text','exec'),ns)
names=['RESEARCH_BACKGROUND.md','RESEARCH_PLAN.md','PROJECT_OVERVIEW.md','FILE_MAP.md','TPMS_RESEARCH_REVIEW.md','MECHANISM_ANALYSIS.md','CRITICAL_MODE_PROGRESS.md']
mirrors=[];broken=[]
for name in names:
 p=R/'docs'/('RESEARCH_STATUS.md' if name=='PROJECT_OVERVIEW.md' else name)
 # Refresh only literal link transformations, without rerunning status mutations.
 p.write_text(ns['formal_text']((W/name).read_text(encoding='utf-8')),encoding='utf-8')
 assert p.read_text(encoding='utf-8')==ns['formal_text']((W/name).read_text(encoding='utf-8'))
 mirrors.append({'Windows':name,'formal':str(p.relative_to(R)),'sha256':sha(p)})
 for file in [W/name,p]:
  for link in re.findall(r'\]\(([^)]+)\)',file.read_text(encoding='utf-8')):
   if '://' in link or link.startswith('#'):continue
   if not (file.parent/link).exists():broken.append({'file':str(file),'link':link})
assert not broken,broken
assert sha(D/'modes.png')==sha(W/'output/figures/FINE_MODE_modes.png')
for p in [D/'process.json',D/'startup_attempt01/process.json']:
 info=read(p);cmd=Path('/proc')/str(info['pid'])/'cmdline'
 assert not cmd.exists() or str(D/'fine_modes.py').encode() not in cmd.read_bytes()
subprocess.run(['git','diff','--check'],cwd=R,check=True)
for name in ['close.py','verify.py','shell_inventory.py','recover_startup.py','stop_startup.py']:
 shutil.copy2(W/'work/fine_mode_20261007'/name,D/name)
src=(W/'work/fine_mode_20261007').resolve();dst=(W/'history/completed_tools_20261007/fine_mode_20261007').resolve()
src.relative_to(W.resolve());dst.relative_to(W.resolve());assert src==W.resolve()/'work/fine_mode_20261007' and not dst.exists()
inventory=[{'relative':str(p.relative_to(src)),'sha256':sha(p)} for p in sorted(src.rglob('*')) if p.is_file()]
dst.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(src),str(dst))
assert all(sha(dst/x['relative'])==x['sha256'] for x in inventory)
(D/'organization_receipt.json').write_text(json.dumps({'source':str(src),'destination':str(dst),'files':inventory,'bytes_preserved':True},indent=2))
verification={'frozen_files_unchanged':len(frozen),'shared_core_unchanged':True,'mirror_documents':mirrors,'broken_links':broken,
 'operator_and_mass_equivalence_checked':True,'four_returned_modes_converged':0,'not_a_critical_or_global_stability_certification':True,
 'visual_QA':'Four-candidate scientific figure inspected; labels/colorbar legible, mode normalization and non-actual-deformation scope visible.',
 'startup_interrupt_not_rewritten_as_success':True,'new_forward':0,'new_Abaqus':0,'new_design_AD':False,'current_running_or_queued_jobs':0,
 'Git_commit_push_merge':False,'archived_tool_files':len(inventory)}
(D/'verification.json').write_text(json.dumps(verification,indent=2))
evidence=[{'path':str(p.relative_to(D)),'sha256':sha(p)} for p in sorted(D.rglob('*')) if p.is_file() and p.name!='evidence_manifest.json']
(D/'evidence_manifest.json').write_text(json.dumps(evidence,indent=2))
print(json.dumps({'frozen_files':len(frozen),'core_unchanged':True,'mirrors':len(mirrors),'broken_links':len(broken),'archived_tools':len(inventory),'mode_convergence_not_passed':True,'new_jobs':0}))

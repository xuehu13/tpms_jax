"""Freeze checks and mirror checks, then archive completed invocation tools."""
from pathlib import Path
import json,hashlib,re,shutil,subprocess,ast
import numpy as np
R=Path('/home/xuehu/projects/tpms_jax');D=R/'validation/critical_mode_20261007_r21'
W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
read=lambda p:json.loads(p.read_text())
frozen=read(D/'frozen_before.json');core=read(D/'source_before.json')
assert all(sha(R/x['path'])==x['sha256'] for x in frozen)
assert all(sha(R/path)==value for path,value in core.items())
manifest=read(D/'launch_manifest.json')
assert sha(D/'diagnose.py')==manifest['analysis_source_sha256'] and sha(D/'protocol.json')==manifest['protocol_sha256']
assert all(sha(R/'validation/step_control_20261007_r18/controlled_forward'/name)==value for name,value in manifest['saved_state_sha256'].items())
assert all(sha(D/x['source'])==x['sha256']==sha(D/x['destination']) for x in read(D/'execution_update.json')['copied_files'])
result=D/'results/result.json'
if not result.exists():result=D/'results/partial_result.json'
raw=read(result)
for x in raw['rows']:
 assert x['relative_asymmetry']<1e-11
 for mode in x['modes']:
  assert mode['Ritz_direct_relative_error']<1e-6
  assert abs(sum(mode['parts_N_mm'].values())-mode['curvature_for_1mm_max_surface_mode_N_mm'])<1e-8*max(abs(mode['curvature_for_1mm_max_surface_mode_N_mm']),1.)
for check in (D/'results').glob('*_column_check.json'):assert read(check)['JVP_columns_vs_jacfwd_relative_error']<1e-11
for check in (D/'results').glob('*_assembly_check.json'):assert read(check)['BLAS_vs_original_contraction_relative_error']<1e-12
assert read(D/'results/GPU_CPU_matrix_check.json')['relative_Frobenius_error']<1e-10
for x in read(D/'basis_checks.json'):assert x['prolongation_gradient_relative_error']<1e-12 and x['P_partition_unity_error']<1e-12
O=D/'physical_metric';final_manifest=read(O/'launch_manifest.json')
assert all(sha(D/x['path'])==x['sha256'] for x in read(O/'frozen_ritz_before.json'))
assert sha(D/'physical_metric.py')==final_manifest['source_sha256']
assert sha(O/'protocol.json')==final_manifest['protocol_sha256']
assert read(O/'launch_receipt.json')['returncode']==0 and not read(O/'launch_receipt.json')['budget_stop']
reuse=read(O/'mass_reuse.json')
assert sha(O/'restricted_HRZ_mass.npz')==reuse['sha256']==sha(D/reuse['source'])
assert reuse['original_mass_inverse_residual']<1e-10
phys=read(O/'result.json');assert len(phys['rows'])==2
for row in phys['rows']:
 assert row['converged'] and len(row['eigenvalues_s_minus2'])==6
 assert max(row['generalized_relative_residuals'])<1e-8 and row['mass_orthogonality_error']<1e-8
 assert np.isfinite(row['eigenvalues_s_minus2']).all()
 with np.load(O/(row['state'][:-4]+'_modes.npz')) as f:
  assert np.isfinite(f['vectors']).all() and np.array_equal(f['eigenvalues'],row['eigenvalues_s_minus2'])
classification=read(O/'translation_classification.json');candidate=read(O/'deformation_candidate.json')
assert classification['all_modes_reported'] and classification['not_new_eigenvalues_after_mean_removal']
for info,row in zip(classification['rows'],candidate['rows'],strict=True):
 assert info['state']==row['state'] and len(info['modes'])==6
 rank=next(x['rank_nearest_zero'] for x in info['modes'] if x['uniform_translation_kinetic_fraction']<.5)
 assert rank==3==row['rank_nearest_zero']
 assert row['projected_generalized_residual']<1e-8 and row['direct_curvature_vs_eigen_mass_relative_error']<1e-6
 # The fine residual is reported, not an accepted fine-mode convergence test.
 assert .9<row['generalized_fine_eigen_residual']<=1.1
 assert np.isfinite(list(row['parts_N_mm'].values())).all()
 with np.load(O/(row['state'][:-4]+'_deformation_candidate.npz')) as f:
  assert np.isfinite(f['mode']).all() and abs(np.linalg.norm(f['surface_mode_mm'],axis=1).max()-1)<1e-10
assert read(O/'interpretation_receipt.json')['returncode']==0
mapping=read(D/'analysis/summary.json')['surface_mapping_checks']
assert all(x['same_node_labels_in_order'] and x['same_triangle_vertex_multiset'] and x['ODB_equals_exact_float32_rounding'] for x in mapping)
assert read(D/'decision.json')['true_fine_critical_mode_not_identified']
tree=ast.parse((W/'work/critical_mode_20261007/close_final.py').read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='formal_text')
ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'formal_text','exec'),ns)
names=['RESEARCH_BACKGROUND.md','RESEARCH_PLAN.md','PROJECT_OVERVIEW.md','FILE_MAP.md','TPMS_RESEARCH_REVIEW.md','MECHANISM_ANALYSIS.md','CRITICAL_MODE_PROGRESS.md']
mirrors=[];broken=[]
for name in names:
 p=R/'docs'/('RESEARCH_STATUS.md' if name=='PROJECT_OVERVIEW.md' else name)
 assert ns['formal_text']((W/name).read_text(encoding='utf-8'))==p.read_text(encoding='utf-8'),name
 mirrors.append({'Windows':name,'formal':str(p.relative_to(R)),'sha256':sha(p)})
 for file in [W/name,p]:
  for link in re.findall(r'\]\(([^)]+)\)',file.read_text(encoding='utf-8')):
   if '://' in link or link.startswith('#'):continue
   if not (file.parent/link).exists():broken.append({'file':str(file),'link':link})
assert not broken,broken
assert sha(D/'analysis/modes.png')==sha(W/'output/figures/CRITICAL_MODE_modes.png')
process=read(D/'process.json');cmd=Path('/proc')/str(process['pid'])/'cmdline'
assert not cmd.exists() or str(D/'diagnose.py').encode() not in cmd.read_bytes()
for p in [O/'process.json',D/'physical_metric_attempt01/process.json']:
 info=read(p);cmd=Path('/proc')/str(info['pid'])/'cmdline'
 assert not cmd.exists() or str(D/'physical_metric.py').encode() not in cmd.read_bytes()
subprocess.run(['git','diff','--check'],cwd=R,check=True)
shutil.copy2(W/'work/critical_mode_20261007/verify.py',D/'verify.py')
src=(W/'work/critical_mode_20261007').resolve();dst=(W/'history/completed_tools_20261007/critical_mode_20261007').resolve()
src.relative_to(W.resolve());dst.relative_to(W.resolve());assert src==W.resolve()/'work/critical_mode_20261007' and not dst.exists()
inventory=[{'relative':str(p.relative_to(src)),'sha256':sha(p)} for p in sorted(src.rglob('*')) if p.is_file()]
dst.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(src),str(dst))
assert all(sha(dst/x['relative'])==x['sha256'] for x in inventory)
(D/'organization_receipt.json').write_text(json.dumps({'source':str(src),'destination':str(dst),'files':inventory,'bytes_preserved':True},indent=2))
verification={'frozen_files_unchanged':len(frozen),'core_unchanged':True,'mirror_documents':mirrors,'broken_links':broken,
 'new_forward':0,'new_Abaqus':0,'design_AD':False,'current_running_jobs':0,'Git_commit_push_merge':False,
 'mechanical_checks':'Q2 embedding, original Gauss/periodic map, symmetry, direct versus Ritz curvature; JVP columns and BLAS equivalence for new implementation.',
 'not_full_space_or_static_buckling_certification':True,'visual_QA':'Mode figure inspected before closure, no actual deformation implied.',
 'zero_target_states_converged':2,'all_six_modes_classified':True,'candidate_not_fine_critical_mode':True,
 'CUDA_preallocation_warning_recorded_with_successful_interpretation_exit':True,
 'archived_tool_files':len(inventory)}
(D/'verification.json').write_text(json.dumps(verification,indent=2))
evidence=[{'path':str(p.relative_to(D)),'sha256':sha(p)} for p in sorted(D.rglob('*')) if p.is_file() and p.name!='evidence_manifest.json']
(D/'evidence_manifest.json').write_text(json.dumps(evidence,indent=2))
print(json.dumps({'frozen_files':len(frozen),'core_unchanged':True,'mirror_documents':len(mirrors),'broken_links':len(broken),'archived_tool_files':len(inventory),'current_running_jobs':0}))

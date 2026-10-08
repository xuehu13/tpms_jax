from pathlib import Path
import json,hashlib,shutil,subprocess
R=Path('/home/xuehu/projects/tpms_jax');D=R/'validation/fine_mode_20261007_r22'
W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an');old=R/'validation/critical_mode_20261007_r21'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
D.mkdir(exist_ok=False)
protocol={'question':'At the saved 12.1904% dynamic state, does refining the r21 block in the original N32 periodic space reveal a softer local deformation direction?',
 'stage':3,'state':'validation/step_control_20261007_r18/controlled_forward/accepted_a0.1200.npz',
 'original_N32_free_periodic_dofs':786429,'macro_H_fixed':True,'pin_periodic_class_zero_all_components':True,
 'same_material_Gauss_mass_PBC_as_r21':True,'K_action':'Exact original material stress Hessian at the fixed F, contracted with original N32 gradients and weights; no assembled global K.',
 'metric':'Original HRZ free diagonal M; solve A=M^(-1/2) K M^(-1/2).',
 'initial_block':'r21 first four near-zero N8/HRZ vectors prolonged to original N32, including the three translation-dominated vectors and fourth deformation candidate; no random modes or amplitude scan.',
 'translation':'Retain all original admissible directions and original pin; classify mean-translation inertia, do not introduce new forward constraints.',
 'solver':'SciPy LOBPCG largest=False, block4, maxiter200; positive diagonal preconditioner from original Hessian row bound and shape gradients, no changes to K or M.',
 'absolute_solver_tol':'1e-3 times smallest absolute initial r21 eigenvalue, fixed before iteration.',
 'candidate_acceptance':'Both independently recomputed mass-whitened and original generalized fine relative residual <=1e-3; finite direct full-domain curvature. Negative valid direction can be reported without claiming mode convergence.',
 'checks':'Material Hessian JVP versus jacfwd at16 original points; original N8 projected action/energy comparison for four initial vectors; operator symmetry and original mass mapping.',
 'budget_seconds':900,'GPU_preallocation':False,'single_attempt_no_backend_or_spectrum_retries':True,
 'limits':'Dynamic unbalanced saved state, block-local refinement and approximate spectrum do not certify static critical strain, global stability, thin-solid accuracy or design AD.',
 'new_time_advance':False,'new_Abaqus':False,'new_design_AD':False,'parameter_or_grid_changes':False,
 'reference':'https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lobpcg.html'}
(D/'protocol.json').write_text(json.dumps(protocol,indent=2))
frozen=json.loads((old/'frozen_before.json').read_text())
frozen += [{'path':str(p.relative_to(R)),'sha256':sha(p)} for p in sorted(old.rglob('*')) if p.is_file()]
(D/'frozen_before.json').write_text(json.dumps(frozen,indent=2))
(D/'source_before.json').write_bytes((old/'source_before.json').read_bytes())
(D/'input_manifest.json').write_text(json.dumps({str(p.relative_to(R)):sha(p) for p in [old/'diagnose.py',old/'physical_metric/accepted_a0.1200_modes.npz',old/'physical_metric/accepted_a0.1200_deformation_candidate.npz',old/'results/accepted_a0.1200_N8_K.npz',old/'physical_metric/restricted_HRZ_mass.npz',R/protocol['state']]},indent=2))
names=['RESEARCH_BACKGROUND.md','RESEARCH_PLAN.md','PROJECT_OVERVIEW.md','FILE_MAP.md','TPMS_RESEARCH_REVIEW.md','MECHANISM_ANALYSIS.md','AGENTS.md','START_HERE.md','CRITICAL_MODE_PROGRESS.md']
backup=W/'history/before_fine_mode_20261007';backup.mkdir(exist_ok=False)
formal=R/'docs/history/before_fine_mode_20261007';formal.mkdir(exist_ok=False)
for name in names:shutil.copy2(W/name,backup/name);shutil.copy2(W/name,formal/name)
note='\n\nr22执行前协议：仅12.1904%原保存态，在原786429个周期自由度上做矩阵自由HRZ模式细化；原N8前四模作初块，保留并分类平移，不加新约束。共享材料切线/积分/质量不改。SciPy LOBPCG最小代数值、最多200迭代、一次900s成本上限；独立原细空间及质量白化残差均≤1e−3才称候选模式收敛。若未收敛原样收口、不另换后端/谱参数重试。没有新前向/Abaqus/设计AD。协议在validation/fine_mode_20261007_r22/protocol.json。\n\n'
p=W/'RESEARCH_PLAN.md';t=p.read_text();a=t.index('## 4.');t=t[:a]+note+t[a:];p.write_text(t)
(R/'docs/RESEARCH_PLAN.md').write_text(t.replace('(PROJECT_OVERVIEW.md)','(RESEARCH_STATUS.md)'))
(D/'git_before.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=R,text=True))
print(json.dumps({'protocol':str(D/'protocol.json'),'frozen_files':len(frozen),'single_saved_state':True,'not_yet_submitted':True}))

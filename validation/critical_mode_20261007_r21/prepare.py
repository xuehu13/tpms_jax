"""Record the bounded periodic Ritz diagnostic before mechanical evaluation."""
from pathlib import Path
import json, hashlib, shutil
R=Path('/home/xuehu/projects/tpms_jax')
W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
D=R/'validation/critical_mode_20261007_r21'
D.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocol={
 'plan_step':3,
 'question':'Is a periodic long-wave folding direction already locally nonconvex in the saved pre-jump JAX states, despite positive curvature in the previous finite jump direction?',
 'states':['accepted_a0.1000.npz','accepted_a0.1200.npz','accepted_a0.1600.npz'],
 'subspaces':{'coarse_Q2_cells_per_axis':[4,8],
  'definition':'Nested periodic triquadratic fields exactly embedded in the original N32 Q2 displacement space. All three components, same pinned translation class, fixed macro H. This restricts only perturbations, not the base geometry/material/Gauss integration.',
  'free_dofs':[1533,12285],
  'no_wall_only_cut_or_void_removal':True,
  'basis_norm':'Euclidean coarse coefficient norm for extraction; signs are invariant under positive norm changes. Eigenvalue magnitudes across resolutions are not compared. Report physical curvature after mass-mean translation removal and 1mm maximum surface displacement normalization.'},
 'operator':'P^T K(F,phi) P; K is the full total-Lagrangian energy Hessian from the existing material stress derivative at all original 27 Gauss points, including prestress through exact dP/dF.',
 'eigensolver':{'N4':'dense symmetric eigh, lowest six',
  'N8':'ARPACK smallest algebraic, k=6 ncv=60 tol=1e-8 maxiter=400; retain convergence failures, no repeated settings or shift fit'},
 'checks':['original Gauss coordinates and class mapping','coarse prolongation gradients agree with original fine Q2 gradients','symmetry','eigenpair residual','selected mode direct full-domain curvature agrees with Ritz matrix','material participation and domain curvature partitions'],
 'mode_analysis':'Evaluate lowest extracted modes on the original surface; compare selected minimum mode to existing shell/JAX folding increments with area-weighted translation removed. Neither increments nor selected Ritz modes are certified full-space critical modes.',
 'budget':'One three-state pass, predeclared two nested spaces, total mechanical diagnostic watchdog 25min. No time advance, new Abaqus job, design AD or parameter fitting.',
 'interpretation':{
  'negative_wall_mode':'Sufficient evidence of local energy nonconvexity in that full-space admissible direction; supports investigating delayed excitation, but does not by itself establish a dynamic instability or its causal contribution.',
  'negative_void_mode':'Record as virtual-domain issue, not a physical shell-folding proof.',
  'positive':'Only subspace positivity. Missing finer-scale directions precludes full-space stability certification.',
  'no_convergence':'Inconclusive; do not substitute the previous positive finite direction or claim a critical load.'},
 'limits':['Saved states are dynamic, not exact equilibria; no static critical compression estimate.',
  'No Abaqus shell tangent is extracted, so stiffness representation versus trigger is not uniquely separated.',
  'The new calculation is a restricted Hessian eigendiagnostic, not a replacement FEM or design gradient.'],
 'sources':['https://docs.software.vt.edu/abaqusv2025/English/SIMACAETHERefMap/simathe-c-eigenbuckling.htm',
  'https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html']}
(D/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
frozen=[]
for name in ['geometry_transfer_20261006_r15','geometry_transfer_review_20261006_r16','mechanism_20261007_r17','step_control_20261007_r18','dynamic_branch_review_20261007_r19','shell_rate_20261007_r20']:
 for p in sorted((R/'validation'/name).rglob('*')):
  if p.is_file():frozen.append({'path':str(p.relative_to(R)),'sha256':sha(p)})
(D/'frozen_before.json').write_text(json.dumps(frozen,indent=2))
core={str(p.relative_to(R)):sha(p) for p in [R/'hyperelastic_fem.py',R/'surface_distance.py',R/'pbc.py',R/'scripts/thin_target_explicit.py']}
(D/'source_before.json').write_text(json.dumps(core,indent=2))
names=['RESEARCH_BACKGROUND.md','RESEARCH_PLAN.md','PROJECT_OVERVIEW.md','FILE_MAP.md','TPMS_RESEARCH_REVIEW.md','MECHANISM_ANALYSIS.md','AGENTS.md','START_HERE.md']
for root in [W/'history/before_critical_mode_20261007',R/'docs/history/before_critical_mode_20261007']:
 root.mkdir(parents=True,exist_ok=False)
 for name in names:shutil.copy2(W/name,root/name)
extra='''本轮r21第3步执行协议已明确：在10.0771%、12.1904%、16.0071%三个真实保存态，使用嵌入原N32 HEX27位移空间的周期Q2 N4/N8两个预定嵌套子空间，求完整材料域切线的最低特征模式。保持宏观H和原平移固定；不改变前向网格/材料，不推进时间，不做设计AD。所有能量仍在原真实27点求值。N4/N8分别1533/12285自由度，仅限制诊断方向，不把子空间正性称为全空间稳定。每态保留最低六模、求解残差、薄壁/虚域参与及中面模式；25min总成本上限，失败原样保存，不反复挑子空间/扰动幅值。协议见`validation/critical_mode_20261007_r21/protocol.json`。若未得到有物理薄壁参与的负方向，仍不能仅凭本试验定位触发或弯曲表示唯一原因，第4步不自动启动。'''
for p in [W/'RESEARCH_PLAN.md',R/'docs/RESEARCH_PLAN.md']:
 text=p.read_text(encoding='utf-8');ix=text.index('## 4.')
 p.write_text(text[:ix]+extra+'\n\n'+text[ix:],encoding='utf-8')
for name in ['prepare.py','diagnose.py']:
 shutil.copy2(W/'work/critical_mode_20261007'/name,D/name)
print(json.dumps({'new_experiment':str(D),'frozen_files':len(frozen),'protocol_recorded_before_execution':True}))

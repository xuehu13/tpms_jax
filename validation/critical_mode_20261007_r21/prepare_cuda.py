from pathlib import Path
import shutil,json,hashlib
R=Path('/home/xuehu/projects/tpms_jax');D=R/'validation/critical_mode_20261007_r21'
W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
A=D/'mechanical_attempt03';A.mkdir(exist_ok=False)
for name in ['diagnose.py','run.py','launch_manifest.json','launch_receipt.json','diagnostic.log','process.json','environment.json','basis_checks.json','cost_stop.json']:
 shutil.move(str(D/name),str(A/name))
shutil.move(str(D/'results'),str(A/'results'));(D/'results').mkdir()
rows=json.loads((D/'reuse_rows.json').read_text())
if (A/'results/partial_result.json').exists():rows=json.loads((A/'results/partial_result.json').read_text())['rows']
copied=[]
for row in rows:
 prefix=f"{row['state'][:-4]}_N{row['Ncoarse']}"
 for p in (A/'results').glob(prefix+'*'):
  shutil.copy2(p,D/'results'/p.name);copied.append({'path':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(D/'reuse_rows.json').write_text(json.dumps(rows,indent=2))
note={'cost_only_recovery':True,'scientific_protocol_unchanged':True,'same_states_subspaces_and_eigensolver_tolerances':True,
 'CUDA_x64_available':'[CudaDevice(id=0)], checked before execution','CPU_attempts_held_below_original_25min_budget':True,
 'additional_one_CUDA_call_budget_seconds':900,'no_more_backend_or_parameter_trials':True,
 'required_GPU_CPU_matrix_relative_Frobenius_error_max':1e-10,
 'checks':'Original full-domain F/Gauss/J and material; compare full assembled 10% N4 matrix to existing CPU matrix, reuse original CPU eigenvectors without re-extracting.',
 'copied_completed_evidence':copied,'new_forward':0,'new_Abaqus':0,'design_AD':False}
(D/'execution_update02.json').write_text(json.dumps(note,indent=2))
for name in ['diagnose.py','run.py','prepare_cuda.py']:shutil.copy2(W/'work/critical_mode_20261007'/name,D/name)
extra='\n本轮CPU诊断成本补充：25min边界内停止CPU路径并保留10% N4完成结果；已有CUDA可用，记录一次额外15min上限的同算子恢复调用（execution_update02.json），先核对GPU与已完成CPU全矩阵相对差≤1e−10，再完成同三个状态/两个子空间。科学状态/材料/子空间/特征判据不变，不追加后端/参数重试；不称原CPU调用通过。\n'
for p in [W/'RESEARCH_PLAN.md',R/'docs/RESEARCH_PLAN.md']:
 t=p.read_text(encoding='utf-8');ix=t.index('## 4.');p.write_text(t[:ix]+extra+'\n'+t[ix:],encoding='utf-8')
print(json.dumps({'CUDA_recovery_protocol_recorded':True,'completed_rows_reused':len(rows),'additional_budget_seconds':900}))

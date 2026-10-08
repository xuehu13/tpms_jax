from pathlib import Path
import shutil,json,hashlib
R=Path('/home/xuehu/projects/tpms_jax');D=R/'validation/critical_mode_20261007_r21';O=D/'physical_metric'
W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
A=D/'physical_metric_attempt01';assert not A.exists();shutil.move(str(O),str(A));O.mkdir()
old_receipt=json.loads((A/'launch_receipt.json').read_text());cuda=json.loads((D/'launch_receipt.json').read_text())
remaining=max(1.,900.-old_receipt['wall_seconds']-cuda['wall_seconds'])
protocol=json.loads((A/'protocol.json').read_text());protocol['solver']='One fixed sigma=0 shift-invert generalized eigsh, k=6 ncv=60 tol=1e-8 maxiter=400; same Kr/Mr. Extract closest-to-zero directions, not smallest algebraic/global minimum.'
protocol['reason']='Smallest-algebraic mass-metric call is incomplete after cost stop. Critical tangent singularity is associated with zero; fixed zero-target extraction is not a shift chosen to fit force peaks. No more spectral retries in this scope.'
protocol['budget_seconds']=remaining;protocol['does_not_certify_no_other_negative_modes']=True
(O/'protocol.json').write_text(json.dumps(protocol,indent=2))
shutil.copy2(A/'restricted_HRZ_mass.npz',O/'restricted_HRZ_mass.npz')
massline=next(json.loads(line) for line in (A/'diagnostic.log').read_text().splitlines() if line.startswith('{') and json.loads(line).get('phase')=='mass')
(O/'mass_reuse.json').write_text(json.dumps({'source':str((A/'restricted_HRZ_mass.npz').relative_to(D)),'sha256':hashlib.sha256((A/'restricted_HRZ_mass.npz').read_bytes()).hexdigest(),'original_mass_inverse_residual':massline['inverse_residual'],'no_mass_model_change':True},indent=2))
shutil.copy2(A/'frozen_ritz_before.json',O/'frozen_ritz_before.json')
shutil.copy2(D/'physical_metric.py',A/'source_at_run.py')
for name in ['physical_metric.py','prepare_zero.py','stop_metric.py']:shutil.copy2(W/'work/critical_mode_20261007'/name,D/name)
extra='\n质量度量最小代数值调用按成本中断，未产出谱，原件在physical_metric_attempt01。本轮最后一次提取固定零目标σ=0（理论切线奇异方向），原Kr/MHRZ/两状态/子空间不变；不扫位移或谱移参数，仅用同一次CUDA恢复剩余'+f'{remaining:.2f}'+'s。它提取零附近模式，不认证全局最小或无其他负模态。协议在physical_metric/protocol.json；此范围不再进行谱求解重试。\n'
for p in [W/'RESEARCH_PLAN.md',R/'docs/RESEARCH_PLAN.md']:
 t=p.read_text(encoding='utf-8');ix=t.index('## 4.');p.write_text(t[:ix]+extra+'\n'+t[ix:],encoding='utf-8')
print(json.dumps({'zero_target_protocol_recorded_before_execution':True,'budget_seconds':remaining,'same_K_M_and_states':True}))

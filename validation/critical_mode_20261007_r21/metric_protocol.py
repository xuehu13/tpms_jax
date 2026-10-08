from pathlib import Path
import json,hashlib
R=Path('/home/xuehu/projects/tpms_jax');D=R/'validation/critical_mode_20261007_r21';O=D/'physical_metric';O.mkdir(exist_ok=False)
W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
assert (D/'results/result.json').exists()
protocol={'plan_step':3,'reason':'N4 smallest bare stiffness modes are void dominated; all three N8 bare ARPACK calls did not converge at original limit. Identify physical periodic mode using actual mass metric, without changing the physical model.',
 'states':['accepted_a0.1200.npz','accepted_a0.1600.npz'],'subspace':'Same fixed N8 periodic Q2 space, same pin and complete original Gauss/material domain.',
 'equation':'Kr v=lambda Mr v; Kr is reused original full-domain Ritz matrix, Mr=P^T diag(actual HRZ periodic mass) P, all three components and same pinned class.',
 'meaning':'lambda has inverse-second-squared units for the entry normalization. No mass relumping or forward-model change.',
 'solver':'eigsh smallest algebraic k=6 ncv=60 tol=1e-8 maxiter=400; exact scalar mass sparse factorization applied to three components; one pass at two predefined states, no solver setting scan.',
 'checks':['positive mass diagonal/full rank embedding, deterministic quadratic checks','M orthonormality and generalized eigen residual','reused stiffness hashes','mechanical full-domain selected mode curvature agrees with Kr','surface/void motion and shell finite event increment patterns'],
 'positive_mode_limit':'No static base equilibrium or full-space stability certification. A positive minimum in this restricted space cannot rule out fine-space instability.',
 'negative_mode_limit':'Record sign and material participation; does not uniquely identify shell/JAX cause or imply dynamic growth history.',
 'budget_seconds':max(1.,900.-json.loads((D/'launch_receipt.json').read_text())['wall_seconds']),
 'cost_rule':'Uses remaining time of the single CUDA recovery scope; no new budget reset, no new forward, Abaqus or design AD.',
 'not_a_parameter_or_perturbation_amplitude_fit':True}
(O/'protocol.json').write_text(json.dumps(protocol,indent=2))
frozen=[{'path':str(p.relative_to(D)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted((D/'results').rglob('*')) if p.is_file()]
(O/'frozen_ritz_before.json').write_text(json.dumps(frozen,indent=2))
extra='\nr21限定质量度量补充协议已在执行前记录（physical_metric/protocol.json）：原裸刚度N8三个状态均未收敛，N4最低模式主要虚域。仅复用已组装N8矩阵和原HRZ，针对12.1904%/16.0071%两态求Kr v=λ(PᵀMHRZP)v；原空间/固定/状态/材料均保持，原特征设置不扫，只用同一次CUDA恢复剩余成本。没有新前向/Abaqus/设计AD；原不收敛记录保持。\n'
for p in [W/'RESEARCH_PLAN.md',R/'docs/RESEARCH_PLAN.md']:
 t=p.read_text(encoding='utf-8');ix=t.index('## 4.');p.write_text(t[:ix]+extra+'\n'+t[ix:],encoding='utf-8')
print(json.dumps({'mass_metric_protocol_recorded_before_execution':True,'states':protocol['states'],'remaining_budget_seconds':protocol['budget_seconds']}))

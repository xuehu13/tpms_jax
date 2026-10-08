"""Record a targeted step-3 saved-state protocol before tangent evaluation."""
from pathlib import Path
import json
R=Path('/home/xuehu/projects/tpms_jax');W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
D=R/'validation/shell_rate_20261007_r20';P=D/'directional_diagnosis';P.mkdir(exist_ok=False)
protocol={'plan_step':3,'factor':'Mechanical response in one observed periodic jump direction',
    'question':'Is the much later JAX jump accompanied by an earlier negative curvature in its observed jump direction, or is this direction still locally stiff?',
    'direction':'r18 last_valid_field q minus accepted_a0.1600 q, fixed at all tested states; full periodic DOFs, pin unchanged',
    'states':['accepted_a0.1000.npz','accepted_a0.1200.npz','accepted_a0.1400.npz','accepted_a0.1600.npz','last_valid_field.npz'],
    'kernel':'same existing material stress derivative, same HEX27 basis/Gauss and complete JxW domain',
    'partitions':['phi<=0.001','0.001<phi<0.01','0.01<=phi<0.5','phi>=0.5'],
    'method':'deltaF : dP(F)[deltaF], summed at each frozen state, actual F/J and scale',
    'not_an_eigenmode_or_static_buckling_point':True,
    'negative_direction_is_sufficient_for_local_energy_nonconvexity':True,
    'positive_direction_does_not_certify_global_stability':True,
    'frozen_states_are_dynamic_not_exact_equilibria':True,
    'no_time_advance_no_design_AD_no_input_changes':True,
    'bounded_attempt':'one batched pass through five retained states; no parameter or direction scan',
    'response_if_inconclusive':'Retain limited conclusion; do not claim stiffness proven or auto-fit a perturbation.'}
(P/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
extra='''
本轮r20第2步已完成：快壳一次81.00s至20%及保载，峰11.8241%，原壳11.4161%，仅移0.4080个百分点，JAX尚迟4.1168个百分点。速率差不是充分解释，原质量失败作诊断保持。

第3步当前限定为一次保存态方向曲率诊断，执行协议在r20/directional_diagnosis/protocol.json：固定r18实际跳跃增量方向`q(17.5363%)−q(16.0071%)`，在10/12/14/16%保存态和停止态用同材料核/真实27点计算完整周期方向的二阶能量变化，并分开深虚域、混合尾部、主要界面和几何核心。零时间推进/设计AD，不改物理输入，不扫方向；固定动态态不能给静态临界点，正曲率不认证全部方向稳定。它先区分“该实际跳跃方向已软化但触发晚”与“该方向局部仍刚”，结论不足时如实保留，不能宣称已经定位壳/实体唯一原因。新前向/扰动仍须有针对性协议，不自动执行。
'''
for file in (W/'RESEARCH_PLAN.md',R/'docs/RESEARCH_PLAN.md'):
    text=file.read_text(encoding='utf-8').replace('## 2. 执行中r20：','## 2. 已完成r20：')
    text=text.replace('准备阶段尚未提交作业，运行状态另见r20收据。','r20壳作业已完成，当前无运行中/排队求解作业。')
    text=text.replace('## 3. 根据第2步结果，只选择一个剩余主要因素','## 3. 执行中：实际跳跃方向的保存态诊断')
    point=text.index('## 4.')
    file.write_text(text[:point]+extra+'\n'+text[point:],encoding='utf-8')
print('One saved-state mechanical directional protocol recorded, no new trajectory.')

"""Record one controlled forward protocol before changing the entry."""
from pathlib import Path
import ast,json,shutil
R=Path('/home/xuehu/projects/tpms_jax');W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
O=R/'validation/step_control_20261007_r18'
probe=json.loads((O/'saved_state_probe/result.json').read_text())
assert probe['status']=='saved_state_bound_diagnostic_complete'
assert all(r['all_material_tangents_finite'] and not r['invalid_material_points'] for r in probe['rows'])
assert max(r['bound_seconds'][-1] for r in probe['rows'])<1.5
shutil.copy2(R/'scripts/thin_target_explicit.py',O/'entry_before_control.py')
text='''

r18保存态评估完成：6项真实残差/诊断检查通过，上界每次约0.76s，临时内存约79MiB。初态安全dt约原74%，原首错前约7%，短dt重放末态约21%。成本支持一次前向尝试，不能由冻结状态宣称全过程稳定。

一次从零到20%+10%时长保载的受控前向协议：材料/几何/加载与原缓存完全不变，原初始dt按原总时长对齐。每个接受块先用上界令dt=min(上一dt,0.8×2/sqrt(R))，只减不增；改变dt时按原加速度保持中心速度并重定位半步速度。复用唯一block推进公式，允许dt作为运行时标量，避免每个dt重新编译；默认旧调用等价。通常128步检查；dt<原dt/2或未续接NH最小J<0.10时缩为16步，仅改变监测频率。块末按原材料/有限性检查，并要求dt×sqrt(R_end)≤2，否则保存拒绝块、从最后接受态有限回退减半；dt低于原初始值1/16则停止，不下调旧保护。冻结端点检查不能认证内部全过程。

预算为推进主体25min（预计比通常15min略长，单次试验有机制价值），超限保存停止，区分预算/步长保护/材料拒绝。沿用原加载≥1%区间的KE/U≤5%时长占比≥95%；若已累计高惯性观察时长超过整个0.004s加载时长的5%，则即使剩余全低惯性也不能通过，按质量停止；不因一次真实突跳短峰立即否定物理响应。所有初始过渡仍保留，后验分母与旧口径一致，不放宽95%/末端5%/功平衡1%门槛。保存每个接受块的实际dt与状态、接近10/12/14/16/18/20%的真实接受场、拒绝场及停止态。完整路径才计算完整曲线/末态指标；若未完成，仅说明该候选实际限制。

当前只排这一项JAX前向，不新增壳/速率/材料/几何/AD作业。完整梯度将来须包括或明确冻结控制产生的时间网格，当前不认证控制决策导数。
'''
for p in (W/'RESEARCH_PLAN.md',R/'docs/RESEARCH_PLAN.md'):
    s=p.read_text();s=s.replace('\n## 3. 解释并匹配转折：',text+'\n## 3. 解释并匹配转折：',1);p.write_text(s,encoding='utf-8')
(O/'control_protocol.json').write_text(json.dumps({'safety':.8,'only_decrease_dt':True,'ordinary_chunk':128,
 'sensitive_chunk':16,'sensitive_trigger_dt_fraction':.5,'sensitive_trigger_required_J':.1,
 'end_frequency_limit':2.,'minimum_dt_fraction':1/16,'body_budget_seconds':1500,
 'quality_stop_when_irrecoverable_high_KE_time':.05*.004,'target_time_seconds':.0044,
 'new_abaqus_jobs':0,'new_design_AD_jobs':0},indent=2)+'\n')
p=R/'scripts/thin_target_explicit.py';s=p.read_text()
s=s.replace('No tangent/global solve, contact, plasticity, mass scaling or damping is added.','No global solve, contact, plasticity, mass scaling or damping is added.\nOptional cell tangents supply only a conservative time-step bound.')
s=s.replace('def advance(state,count,geometry,nodem=None,mass=None):','def advance(state,count,geometry,nodem=None,mass=None,step_dt=None):',1)
anchor='            def active(carry):\n'
assert s.count(anchor)==1;s=s.replace(anchor,'            local_dt=dt if step_dt is None else step_dt\n'+anchor,1)
s=s.replace('vhalf+dt*self.acceleration(q,h,hdd,geometry,nodem,mass)','vhalf+local_dt*self.acceleration(q,h,hdd,geometry,nodem,mass)',1)
s=s.replace('q+dt*vhalf','q+local_dt*vhalf',1)
s=s.replace('return q,vhalf,t+dt','return q,vhalf,t+local_dt',1)
s=s.replace('def apply(state,count=steps,material=None):','def apply(state,count=steps,material=None,step_dt=None):',1)
s=s.replace('return compiled(state,count,self.kernel_geometry)','return compiled(state,count,self.kernel_geometry,step_dt=step_dt)',1)
s=s.replace('return compiled(state,count,*material)','return compiled(state,count,*material,step_dt=step_dt)',1)
p.write_text(s)
print(json.dumps({'protocol_recorded_before_run':True,'shared_block_equations_unchanged':'runtime dt optional; constant default equivalent'}))

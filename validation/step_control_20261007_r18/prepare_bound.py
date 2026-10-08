"""Prepare only the saved-state bound experiment, with old evidence frozen."""
from pathlib import Path
import hashlib,json,shutil
R=Path('/home/xuehu/projects/tpms_jax');W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
O=R/'validation/step_control_20261007_r18';assert not O.exists();O.mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
frozen={str(p.relative_to(R)):sha(p) for n in ('geometry_transfer_20261006_r15','geometry_transfer_review_20261006_r16','mechanism_20261007_r17') for p in (R/'validation'/n).rglob('*') if p.is_file()}
(O/'frozen_before.json').write_text(json.dumps(frozen,indent=2)+'\n')
shutil.copy2(R/'scripts/thin_target_explicit.py',O/'entry_before.py')
detail='''

2026-10-07继续授权“有进展的话就继续研究”。r18先执行保存态上界/成本，不自动提交完整作业。候选：B=M^(-1/2)KM^(-1/2)，K为唯一共享材料切线产生的周期波动切线；逐单元积分出K_e，先取各单元贡献的绝对行和，再周期归并，以三角不等式形成全局谱半径上界R。固定平移自由度的行/列排除，使用实际周期HRZ质量，所有材料域参与。dt候选=min(原初始dt,0.8×2/sqrt(R))；绝对值只用于界限，不替代内力或投影物理负切线。这是冻结状态正频率限制，不证明有限时段的非线性全过程稳定。

先用小周期模型的共享残差Jacobian检查R≥实际最大特征值绝对值、周期重合/固定自由度和力学未改；随后只评估r17约10/14/15.97%、首错前一步及短重放末态，记录编译/重复调用时间、内存和dt缩减。分批128单元，不存全局K或全材料H。若计算可接受才将监测间隔、减步/预算/质量停止协议写回本节，执行最多一次前向。若过于保守或耗时，则报告结果，选择有证据的下一项，不无限缩步或为20%放宽门槛。材料/几何/加载、原能量与质量门槛保持。记录机械切线计算成本，不称设计AD。
'''
for p in (W/'RESEARCH_PLAN.md',R/'docs/RESEARCH_PLAN.md'):
    backup=O/('plan_before_windows.md' if p.parent==W else 'plan_before_formal.md');shutil.copy2(p,backup)
    s=p.read_text();s=s.replace('\n## 3. 解释并匹配转折：',detail+'\n## 3. 解释并匹配转折：',1);p.write_text(s,encoding='utf-8')
(O/'authorization.json').write_text(json.dumps({'request':'好的，有进展的话就继续研究','current_scope':'Saved-state conservative bound and cost first; full forward only after recorded decision','new_abaqus_jobs':0,'new_design_AD_jobs':0},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'experiment':str(O),'frozen_file_count':len(frozen)}))

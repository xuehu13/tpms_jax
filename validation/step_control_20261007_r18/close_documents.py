"""Publish one current reading report and plan; preserve scientific originals."""
from pathlib import Path
import hashlib, json, shutil
W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
R=Path('/home/xuehu/projects/tpms_jax');O=R/'validation/step_control_20261007_r18'
read=lambda p:json.loads(p.read_text())
m=read(O/'analysis/controlled_analysis.json');q=read(O/'analysis/quality_bound.json')
assert m['stop']['type']=='TimeoutError' and not m['complete_20pct_and_hold']
assert not q['original_95pct_quality_still_possible']
assert read(O/'diagnose_control_endpoint_attempt02_receipt.json')['returncode']==0
shutil.copy2(W/'work/step_control_20261007/plan_after.md',W/'RESEARCH_PLAN.md')
shutil.copy2(O/'analysis/controlled_response.png',W/'work/step_control_20261007/controlled_response.png')
report=(W/'STEP_CONTROL_PROGRESS.md').read_text()
(R/'docs/STEP_CONTROL_PROGRESS.md').write_text(report.replace('work/step_control_20261007/controlled_response.png','../validation/step_control_20261007_r18/analysis/controlled_response.png'))
(O/'REVIEW.md').write_text(report.replace('work/step_control_20261007/controlled_response.png','analysis/controlled_response.png'))

status='''更新2026-10-07，r18：前置状态相关时间步候选的保存态评估、26项检查及一次从零前向已完成。推进到17.5363%，685个接受端点无NH越界/非有限，1次频率界限回退；按25min预算停止，调用25.50min。末端KE/U20.41%、已观察功平衡差0.00246%；即使剩余全部低惯性，原低动能时长目标最多94.02%<95%。这是部分材料有效路径，不是可信20%结果。1%～15%共同采样新旧JAX力最多差0.0101%，较早降载差异仍在。下一项依唯一规划解释降载模式与准静态/壳质量；没有运行中或排队作业、设计AD或训练。证据见[前置控制执行](STEP_CONTROL_PROGRESS.md)。

以下r17定位/短重放和r12～r16为当时事实，历史下一步不替代当前唯一主规划。

'''
p=W/'PROJECT_OVERVIEW.md';s=p.read_text();assert s.startswith('# 当前状态\n');p.write_text('# 当前状态\n\n'+status+s.split('\n\n',1)[1])
p=W/'FILE_MAP.md';s=p.read_text();s=s.replace('更新2026-10-07。正式程序','更新2026-10-07，r18。正式程序',1)
s=s.replace('| MECHANISM_PROGRESS.md |','| STEP_CONTROL_PROGRESS.md | r18保守时间步上界/单次部分前向/质量上界及域诊断；docs同名，validation/step_control_20261007_r18/REVIEW.md |\n| MECHANISM_PROGRESS.md |',1)
s=s.replace('定位/短dt因果重放完成，通用控制/分支/迁移待处理','定位/短dt与前置控制候选评估完成；20%目标未过，降载分支/准静态条件待处理',1)
s=s.replace('| validation/mechanism_20261007_r17/ |','| validation/step_control_20261007_r18/ | 保存态界限、一次受控前向至17.5363%预算停止、真实接受场/拒绝场、质量上界/域切线及来源；不是完整20%或AD |\n| validation/mechanism_20261007_r17/ |',1)
s+='\n2026-10-07 r18：Windows `work/step_control_20261007`仅调用/只读分析与文档工具；正式脚本仍唯一入口，控制默认关闭。旧r15/r16/r17的174个科学文件冻结。改写前阅读文档保存在`history/before_step_control_closure_20261007`及正式`docs/history/`同名目录；实际协议另保存在r18。当前无运行中/排队作业。\n'
p.write_text(s)
p=W/'RESEARCH_BACKGROUND.md';s=p.read_text().replace('当前代表为diverse_28中面','已验证代表为diverse_28中面；当前机制研究为diverse_04，二者均用',1)
s=s.replace('20%代表算例已通过','diverse_28的20%代表算例已通过',1)
anchor='上一轮四步已收口；当前唯一近期四步围绕';i=s.index(anchor);j=s.index('\n\n正式程序',i)
s=s[:i]+'r15/r16迁移失败及r17首因证据保持；r18前置时间步控制避开原数值爆发，接受路径至17.5363%后预算停止，惯性仍不合格，未通过20%。当前只按机制研究主规划解释较早降载/分支与准静态条件，不无限缩步。详细进展见[前置控制执行](STEP_CONTROL_PROGRESS.md)。完整20%新核梯度未认证；不自动全面扫描曲面/构型/本构或启动训练。'+s[j:]
p.write_text(s)
p=W/'START_HERE.md';s=p.read_text();s=s.replace('\n上一轮四步已完成并冻结：','\n当前2026-10-07 r18：前置控制候选完成一次评估，材料有效接受路径至17.5363%后预算停止；惯性/较早分支差异仍未解，未认证20%。见[最新执行](STEP_CONTROL_PROGRESS.md)。当前无运行中或排队作业，只执行唯一主规划。\n\n以下r12～r17历史阅读说明不生成待办。原四步已完成并冻结：',1)
s=s.replace('当前diverse_04四步的第1步','原r15/r16 diverse_04四步的第1步',1);p.write_text(s)
p=W/'TPMS_RESEARCH_REVIEW.md';s=p.read_text();pos=s.index('\n\n')
s=s[:pos]+'\n\n2026-10-07机制增量：r17定位到原步长超限，r18保守前置控制避开原首错，部分接受路径至17.5363%后预算停止；降载惯性和较早分支差异仍未解。未通过新构型20%或完整AD。最新事实见[前置控制](STEP_CONTROL_PROGRESS.md)，行动仅看主规划；下文2026-10-06方法底稿与旧限定范围保留。'+s[pos:];p.write_text(s)
bullet='- 当前2026-10-07机制规划：r17首因/短dt与r18前置状态相关上界候选评估已完成。r18一次从零前向至17.5363%，685接受端点无NH越界/非有限，1次频率界限回退；25min预算停止，不是方法失败。末端KE/U20.41%，已观察功平衡差0.00246%；剩余全低惯性最多94.02%<原95%，不延长原路径凑过。1%～15%六个采样新旧JAX力最多差0.0101%，较早分支/壳质量未解。下一项按唯一主规划复用真实场解释降载模式/准静态条件，先形成一次针对性协议再新前向；无排队作业/AD。控制默认关闭，共享内力/质量保持；block只新增可选运行时dt，默认公式AST等价，不宣称类全部原字节。短重放不拼路径，r18不冒充完整20%。证据STEP_CONTROL_PROGRESS.md/r18；174旧科学文件冻结。'
for p in (W/'AGENTS.md',R/'AGENTS.md'):
    lines=p.read_text().splitlines();found=[i for i,l in enumerate(lines) if l.startswith('- 当前2026-10-07机制规划：')];assert len(found)==1
    lines[found[0]]=bullet if p.parent==W else bullet.replace('证据STEP_CONTROL_PROGRESS.md','证据docs/STEP_CONTROL_PROGRESS.md')
    p.write_text('\n'.join(lines)+'\n')
for name in ('RESEARCH_BACKGROUND.md','RESEARCH_PLAN.md','PROJECT_OVERVIEW.md','FILE_MAP.md','TPMS_RESEARCH_REVIEW.md'):
    dest=R/'docs'/('RESEARCH_STATUS.md' if name=='PROJECT_OVERVIEW.md' else name)
    text=(W/name).read_text().replace('(PROJECT_OVERVIEW.md)','(RESEARCH_STATUS.md)');dest.write_text(text)
p=R/'README.md';s=p.read_text();s+='\n2026-10-07 r18当前执行见[前置时间步控制](docs/STEP_CONTROL_PROGRESS.md)：一次受控前向至17.5363%后预算停止，未通过新构型20%；惯性/分支问题依唯一主规划处理，无排队作业。旧报告的下一步不是当前待办。\n';p.write_text(s)
p=R/'validation/README.md';s=p.read_text();s+='\n- `step_control_20261007_r18/`：保守界限/单次受控前向至17.5363%预算停止、质量上界及域切线分析。未认证20%或设计AD；当前行动只看docs/RESEARCH_PLAN.md。\n';p.write_text(s)
f=read(O/'controlled_forward/failure.json')
decision={'current_plan_step':2,'candidate_evaluation_complete':True,'trusted_20pct_goal_passed':False,
 'actual_stop':'body_budget','forward_call_seconds':m['receipt']['wall_seconds'],'body_seconds':f['body_seconds'],
 'bound_seconds_total':f['bound_seconds_total'],'accepted_material_valid_partial_path':True,
 'original_loading_quality_unattainable_on_this_path':True,'next_plan_step':3,
 'next_item':'Existing-field branch/mechanism and quasistatic-reference diagnosis, then one evidence-based comparison protocol.',
 'new_queued_jobs':0,'new_abaqus_jobs':0,'new_design_AD_jobs':0,'github_committed_this_round':False}
(O/'decision.json').write_text(json.dumps(decision,indent=2,ensure_ascii=False)+'\n')
for src in (W/'work/step_control_20261007').glob('*.py'):
    if not (O/src.name).exists():shutil.copy2(src,O/src.name)
print(json.dumps(decision,indent=2))

from pathlib import Path
import json,hashlib,shutil,subprocess,re
R=Path('/home/xuehu/projects/tpms_jax');W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an');O=R/'validation/learning_bridge_20261003_r3'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
initial=json.loads((O/'initial_manifest.json').read_text())
p=R/'docs/PAPER_ROUTE.md';before=O/'source_before/docs/PAPER_ROUTE.md';shutil.copy2(p,before);initial['source_before']['docs/PAPER_ROUTE.md']=sha(before)
paper=(W/'PAPER_ROUTE.md').read_text().replace('N64 仍是待验证假设','第三轮N64专用导数已通过方向差分及资源核查，通用N64伴随未执行').replace('唯一四步和预算均在主规划','第三轮已按停止条件收口，原四步与预算保留于计划归档，结论见第三轮报告')
(W/'PAPER_ROUTE.md').write_text(paper);p.write_text(paper)
filemap=(W/'FILE_MAP.md').read_text().replace('第一、二轮数值改动尚未提交。','本项目阶段改动尚未提交。').replace('两轮 source_before/source_after','三轮 source_before/source_after').replace('NEAR_TERM_REPORT/ROUND2_REPORT 为历史完成报告。','NEAR_TERM_REPORT/ROUND2_REPORT/ROUND3_REPORT 为历史收口报告。')
(W/'FILE_MAP.md').write_text(filemap);(R/'docs/FILE_MAP.md').write_text(filemap.replace('(PROJECT_OVERVIEW.md)','(RESEARCH_STATUS.md)'))
p=R/'README.md';t=p.read_text().replace('- [本轮四步规划](docs/RESEARCH_PLAN.md)：后续执行的唯一入口。','- [近期主规划与收口状态](docs/RESEARCH_PLAN.md)：执行的唯一入口。').replace('第一轮实际完整回归 127/127，87.63s；当前数值基线','本轮原130项回归82.25s与新增3项6.95s分别通过；当前数值基线')
t=t.replace('第三轮已按停止条件收口，见','第二轮M64隐式数组三锚点全局刚度通过，见[第二轮报告](docs/ROUND2_REPORT.md)。\n\n第三轮已按停止条件收口，见');p.write_text(t)
p=R/'validation/README.md';t=p.read_text().replace('第一轮已封存；第二轮也已按判据收口，证据见','第一轮已封存；第二轮也已按判据收口，证据见')
t=t.replace('下一轮未制定。\n| 目录','下一轮未制定。\n\n| 目录')
t=t.replace('| [near_term_20261003](near_term_20261003/README.md)', '| [learning_bridge_20261003_r3](learning_bridge_20261003_r3/README.md) | N64专用梯度通过；新壁宽示例性能跨度不足／G48网格审查失败，第3/4步未启动 |\n| [geometry_interface_20261003_r2](geometry_interface_20261003_r2/README.md) | 隐式输入三锚点与小／中网格梯度，通过与资源停止证据冻结 |\n| [near_term_20261003](near_term_20261003/README.md)')
t=t.replace('正式成功 Abaqus 作业共 23 项（含 2 项矩阵诊断），第一轮实际完整回归 127/127，87.63s。','历史截至2026-10-02的Abaqus分析／矩阵诊断共23项；加第二轮4、第三轮1，当前总数28，datacheck另计。原23项索引及第一轮127项回归是历史快照。')
t=t.replace('研究完成还需要多构型接口、训练、逆向设计和独立设计复核，不能以这些验证记录替代。','这些验证记录不能替代训练与逆向设计证据；具体后续只能由新的近期主规划决定。');p.write_text(t)
(O/'initial_manifest.json').write_text(json.dumps(initial,indent=2)+'\n')
for name in initial['source_before']:
 p=O/'source_after'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(R/name,p)
for name in ['validation/README.md','docs/ROUND3_REPORT.md','docs/RESEARCH_PLAN_ROUND3_COMPLETED.md']:
 p=O/'source_after'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(R/name,p)
print('Final documentation state corrected; only document changes after tests.')

from pathlib import Path
import json
W=Path(__file__).resolve().parents[2];P=Path('/home/xuehu/projects/tpms_jax');O=P/'validation/mechanics_trust_20261003_r4/step4'
names={'RESEARCH_BACKGROUND.md':'RESEARCH_BACKGROUND.md','RESEARCH_PLAN.md':'RESEARCH_PLAN.md','PROJECT_OVERVIEW.md':'RESEARCH_STATUS.md','FILE_MAP.md':'FILE_MAP.md'}
(O/'documents_before_execution').mkdir(exist_ok=False)
for src,dst in names.items():
    f=W/src;(O/'documents_before_execution'/src).write_bytes(f.read_bytes());s=f.read_text()
    if src=='RESEARCH_BACKGROUND.md':s=s.replace('第4步已验证中等壁厚Gyroid的输入、参考和资源预算待锁定，尚未启动','第4步已验证中等壁厚Gyroid的输入、参考和资源预算已锁定，先执行资源与畸变预检查')
    if src=='PROJECT_OVERVIEW.md':s=s.replace('先锁定插值、参考路径与资源/停止预算；尚未启动','插值、参考路径与资源/停止预算已锁定，资源与畸变预检查执行中')
    if src=='FILE_MAP.md':
        s=s.replace('第1/2/3步已收口，无本轮运行/排队作业；第4步未启动','第1/2/3步已收口；第4步输入/参考/预算已锁定，预检查执行中')
        s=s.replace('Abaqus根 `E:', '第4步执行证据在`validation/mechanics_trust_20261003_r4/step4/`，Windows一次性工具在`work/mechanics_trust_20261004_r4_step4/`；当前先作资源/畸变预检查，不认证粗网格精度。\n\nAbaqus根 `E:')
    f.write_text(s);(P/'docs'/dst).write_text(s)
f=W/'AGENTS.md';s=f.read_text().replace('下一步为原第4步已验证中等壁厚Gyroid的有限应变探查，输入/参考/资源预算待锁定，尚未启动','当前执行原第4步已验证中等壁厚Gyroid的有限应变探查，输入/参考/资源预算已锁定，资源与畸变预检查执行中')
(O/'documents_before_execution/AGENTS.md').write_bytes(f.read_bytes());f.write_text(s)
for n in ('RESEARCH_BACKGROUND.md','RESEARCH_PLAN.md','PROJECT_OVERVIEW.md','FILE_MAP.md','TPMS_RESEARCH_REVIEW.md','ROUND4_STEP1_REPORT.md','ROUND4_STEP2_REPORT.md','ROUND4_STEP3_REPORT.md','ROUND3_REPORT.md'):
    s=s.replace(']('+n+')','](docs/'+names.get(n,n)+')')
s=s.replace('此工作区主要是阅读副本和历史结果','本目录为正式程序；Windows工作区主要是阅读副本和历史结果')
(P/'AGENTS.md').write_text(s);(O/'sync_in_progress.py').write_bytes(Path(__file__).read_bytes())
print('Active plan/status synchronized; frozen reports unchanged')

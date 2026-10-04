"""Close the sole plan and record the measured boundary without new tasks."""
from pathlib import Path
W=Path(__file__).resolve().parents[2];P=Path('/home/xuehu/projects/tpms_jax');O=P/'validation/mechanics_trust_20261003_r4/step4'
names=['RESEARCH_BACKGROUND.md','RESEARCH_PLAN.md','PROJECT_OVERVIEW.md','FILE_MAP.md','START_HERE.md','AGENTS.md']
(O/'documents_before_close').mkdir(exist_ok=False)
for n in names:(O/'documents_before_close'/n).write_bytes((W/n).read_bytes())
def replace(s,old,new):
    assert old in s,old
    return s.replace(old,new)
f=W/'RESEARCH_PLAN.md';s=f.read_text()
s=replace(s,'第1/2步收口，第3步完整均匀实体有限应变基准通过；第4步输入/参考/预算已锁定，资源与畸变预检查执行中',
    '第四轮四步已按判据/停止条件收口；第4步受当前线性求解成本限制，TPMS有限应变精度尚未认证，下一轮待锁定')
s=replace(s,'**2026-10-04已锁定并执行中：**','**2026-10-04锁定记录（本步已收口）：**')
key='## 第4步：对一个已验证Gyroid探查有限应变压缩的可信上限\n'
note='\n**2026-10-04按资源停止条件收口：** N16/N32预检查均至5%，内部物理检查通过；N48完成0.01%小应变点，刚度与冻结线性结果差0.00171%，随后在0.5%点第二次线性校正中触及每点300秒门槛。实际3条路径尝试、26个保存状态，N64/半步/独立Abaqus非线性参考/10%及20%均未启动；148项完整回归通过。报告见[第4步](ROUND4_STEP4_REPORT.md)，正式step4保留锁定/执行/停止/源码和指纹。当前是配置成本边界，没有TPMS有限应变精度认证；本轮不自动续跑条件项，也不临时添加稳定化、接触或训练。\n'
s=replace(s,key,key+note)
s=replace(s,'本轮结束应得到两项决策依据：**小变形TPMS计算的实际适用范围；向有限应变压缩延伸的可信区间与主要障碍。** 然后才制定下一轮。训练和逆设计继续后置，不自动接回第三轮未执行步骤。',
    '本轮已形成两项决策依据：**小变形TPMS计算的限定适用范围；有限应变基础/粗网格前向证据，以及目标精度网格的求解成本障碍。** TPMS有限应变精度区间仍未认证。下一轮应先对同一模型目标网格的切线方程锁定一次有针对性的成本/求解配置检查，再恢复独立实体对照；新输入/预算尚未锁定，不自动续跑本轮条件项。训练和逆设计继续后置，不自动接回第三轮未执行步骤。')
f.write_text(s)
f=W/'RESEARCH_BACKGROUND.md';s=f.read_text()
s=replace(s,'第4步已验证中等壁厚Gyroid的输入、参考和资源预算已锁定，先执行资源与畸变预检查',
    '第4步按资源停止条件收口：Gyroid粗网格至5%有内部证据，N48在0.5%点触及每点时间门槛，目标精度网格和独立有限应变参考未成立；下一轮输入/预算待锁定')
s=replace(s,'完整均匀实体的有限应变算法基准通过至20%。薄壁参考未成立；',
    '完整均匀实体的有限应变算法基准通过至20%；Gyroid有限应变粗网格前向有内部一致性，精度认证受当前求解成本限制。薄壁参考未成立；')
f.write_text(s)
f=W/'PROJECT_OVERVIEW.md';s=f.read_text()
s=replace(s,'薄壁Gyroid的独立细参考未成立，其精度尚未知；TPMS大压缩尚未验证。',
    '薄壁Gyroid的独立细参考未成立，其精度尚未知；Gyroid有限应变粗网格至5%有内部证据，N48在0.5%点触及每点时间上限，目标精度与独立有限应变对照尚未完成。')
key='数字含义和筛查量的非严格性质见综合报告第6节。'
row='| Gyroid有限应变探查 | N16/N32至5%内部检查通过；N48在0.01%接回线性刚度，差0.00171%；0.5%点按300秒上限停止 | 当前GMRES/GAMG求解成本边界；没有N48/N64有限应变网格/步长或独立实体对照，未认证TPMS有限应变区间 |\n\n'
s=replace(s,key,row+key)
s=replace(s,'下一步为原第4步已验证中等壁厚Gyroid的有限应变探查，插值、参考路径与资源/停止预算已锁定，资源与畸变预检查执行中。',
    '第4步已按资源停止条件收口，见[报告](ROUND4_STEP4_REPORT.md)；下一轮待锁定。')
s+='\n第4步实际3条科研路径尝试、26个保存状态（含3个零状态）；N48仅完成0.01%小应变点，在0.5%第二次线性校正期间触及300秒上限。N16/N32粗路径反力相差约5.268%，是离散变化指示量，不是真误差；不能据粗网格认证5%精度。N32非零完成点约42–107s，N48小应变点约298s；目标精度网格未建立可负担路径，没有新Abaqus分析/输入、半步/孔隙诊断、设计梯度或训练。\n\n本步扩展hyperelastic_fem.py，新增有限应变Gyroid捕获脚本和4项物理测试，最终148项完整回归通过。原FEM/PBC/线性密度/设计梯度模块和环境保持指纹不变，第四轮前三步冻结；新verification/closure在正式step4。没有Git提交、推送或合并。下一轮应针对同一目标网格切线方程检查求解配置与成本，再恢复二值实体对照；不自动放宽本轮时限、增加稳定化或接触/训练平台。\n'
f.write_text(s)
f=W/'FILE_MAP.md';s=f.read_text()
s=replace(s,'及[第3步报告](ROUND4_STEP3_REPORT.md)','、[第3步](ROUND4_STEP3_REPORT.md)及[第4步报告](ROUND4_STEP4_REPORT.md)')
s=replace(s,'第四轮唯一计划；第1/2/3步收口，下一步为已验证Gyroid的有限应变探查',
    '第四轮四步按判据/停止条件收口；第4步成本受限，下一轮待锁定')
key='正式目录 `/home/xuehu/projects/tpms_jax`'
s=replace(s,key,'| ROUND4_STEP4_REPORT.md | docs/同名文件及validation/mechanics_trust_20261003_r4/step4/README.md | Gyroid粗网格有限应变路径、N48时间预算停止与成本边界 |\n\n'+key)
s=replace(s,'Neo-Hookean有限应变材料/响应，复用周期运动学和JAX-FEM求解器；当前只有完整实体认证',
    'Neo-Hookean及参考Gauss占据能量缩放，复用周期/求解器；完整实体认证，Gyroid前向精度尚未认证')
s=replace(s,'| tests/ | 维护检查；与研究工况分别计数 |',
    '| scripts/finite_strain_gyroid.py | 已锁定Gyroid的有限应变前向捕获、内部物理/软域检查、保存成功状态；本轮无完整精度路径 |\n| tests/ | 维护检查；与研究工况分别计数；当前148项通过 |')
s=replace(s,'第4步执行证据在`validation/mechanics_trust_20261003_r4/step4/`，Windows一次性工具在`work/mechanics_trust_20261004_r4_step4/`；当前先作资源/畸变预检查，不认证粗网格精度。',
    '第4步证据在`validation/mechanics_trust_20261003_r4/step4/`：plan/precision_launch、两条完整预检查JSON、N48部分进度/原始日志/当前点、全部节点位移和最后状态、执行与停止决策、summary/details、区域能量后处理、源码/测试/verification/closure及报告。N48没有完整成功路径JSON；N64和Abaqus非线性参考未启动。Windows一次性工具、图和发布收据在`work/mechanics_trust_20261004_r4_step4/`，正式维护入口仍为WSL新捕获脚本。')
s=replace(s,'第1/2/3步已收口；第4步输入/参考/预算已锁定，预检查执行中',
    '第四轮四步已按判据/停止条件收口，无本轮运行/排队作业；第4步有限应变精度未认证，下一轮待锁定')
s=replace(s,'本步同步见`work/mechanics_trust_20261004_r4_step3/publish_receipt.json`',
    '第3步同步见`work/mechanics_trust_20261004_r4_step3/publish_receipt.json`，本次见`work/mechanics_trust_20261004_r4_step4/publish_receipt.json`')
s+='\n第4步仅扩展一个材料模块、增加一个捕获脚本及一个4项物理检查文件，148项回归通过。本步无新Abaqus包/ODB、无依赖安装、无复制FEM、无设计梯度或训练；旧证据/传输副本保持原位置和指纹。综合PDF仍为此前快照，本步报告为Markdown和科学图。\n'
f.write_text(s)
f=W/'START_HERE.md';s=f.read_text();s=replace(s,'下一步是已验证Gyroid的有限应变探查，尚未启动。',
    '[第4步](ROUND4_STEP4_REPORT.md)Gyroid粗网格至5%有内部证据，N48在0.5%点按时间预算停止；第四轮已收口，TPMS有限应变精度未认证，下一轮待锁定。');f.write_text(s)
f=W/'AGENTS.md';s=f.read_text();s=replace(s,'当前执行原第4步已验证中等壁厚Gyroid的有限应变探查，输入/参考/资源预算已锁定，资源与畸变预检查执行中',
    '第4步已按资源停止条件收口，见[报告](ROUND4_STEP4_REPORT.md)：N16/N32至5%内部检查通过，N48在0.01%接回线性极限，在0.5%点触及每点300秒上限；N64/半步/独立有限应变Abaqus/10%及20%未启动，TPMS有限应变精度未认证。下一轮输入/预算待锁定，先针对同一目标网格的切线求解成本；不自动续跑本轮条件项或临时增加稳定化/接触/训练');f.write_text(s)
(O/'update_documents.py').write_bytes(Path(__file__).read_bytes())
print('Sole plan and canonical facts closed at resource boundary; no new stage started')

"""Keep one current plan and update only reading entry points after step 3."""
from pathlib import Path
import json, shutil
R=Path('/home/xuehu/projects/tpms_jax');O=R/'validation/thickness_range_20261006_r13'
W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
read=lambda p:json.loads(p.read_text())
d=read(O/'decision.json');passed=d['all_new_cases_meet_scoped_engineering_targets']
outcome='两点均通过原前向门槛' if passed else '已完成，但范围门槛未全部通过，失效原样保留'
worst=f'{100*d["maximum_hold_force_difference"]:.2f}%/{100*d["maximum_curve_RMS_over_fixed_Standard_peak"]:.2f}%/{100*d["maximum_work_difference"]:.2f}%'
brief=f'第3步匹配0.45/0.55mm已完成，0.50mm复用；{outcome}，三个采样厚度最大反力/曲线/功差{worst}。'
def replace_once(text,old,new):
    assert text.count(old)==1,old
    return text.replace(old,new)

p=W/'RESEARCH_PLAN.md';s=p.read_text()
s=replace_once(s,'当前第1/2步完成，第3步待执行，本次整理零新科学作业。','当前第1/2/3步完成；第4步待收口，本轮新增2条JAX路径与2个匹配Abaqus壳作业，无完整AD/训练。')
s=replace_once(s,'## 3. 有限物理工况：当前唯一下一项','## 3. 有限物理工况：已执行，结果冻结')
s=replace_once(s,'科研问题：固定方法的约7%反力差能否在相邻真实厚度下保持，而不是只对一个点有效？',
               brief+' 证据见[厚度执行报告](THICKNESS_RANGE_PROGRESS.md)和`validation/thickness_range_20261006_r13`；以下输入/门槛为本步先固定的口径，不生成重复作业。\n\n科研问题：固定方法的约7%反力差能否在相邻真实厚度下保持，而不是只对一个点有效？')
s=replace_once(s,'## 4. 收口前向范围与未来梯度关口：待第3步','## 4. 收口前向范围与未来梯度关口：当前唯一下一项')
s=replace_once(s,'合并已验证范围与新增两点结果，','按第3步实测结论收口，不重算两点。曲面参数、一个不同构型或材料均可作为后续范围候选；先评估已有匹配中面与参照，以一个有区分力的几何变化为优先，不能把候选表变成自动广泛扫描。材料需匹配真实本构并另核虚域处理，E缩放不等于新机制验证。完整20%梯度是另一个关口，不由厚度范围通过代替。\n\n合并已验证范围与新增两点结果，')
s=replace_once(s,'r6至r12科学结果冻结。本次仅整理文档与完成工具，科学原件、用户论文、ODB不动。',
               'r6至r12科学结果及本轮r13原始结果冻结。原科学字节、用户论文及旧ODB不动；新ODB在独立厚度作业目录。')
p.write_text(s)

p=W/'PROJECT_OVERVIEW.md'
p.write_text(f'''# 当前状态

2026-10-06匹配厚度轮次完成；正式仓库基线dd66775。仿真优先、训练后置。具体数字及解释见[厚度报告](THICKNESS_RANGE_PROGRESS.md)，任务仅看[主规划](RESEARCH_PLAN.md)。

| 进度 | 当前事实 |
| --- | --- |
| 第1步定位 | 完成；原HEX8/HEX27、桥接、速率、保存场及失败诊断冻结 |
| 第2步JAX改善 | 完成；objective_void+C²的45项核/接口、原6点局部AD/FD、快慢20%及125点通过；默认NH未改 |
| 第3步有限厚度 | {outcome}；新增0.45/0.55mm匹配壳/JAX均计算至20%及保载，0.50mm复用 |
| 响应范围 | 三个采样厚度最大反力/曲线/输入功差{worst}；只限同中面、NH、XYZ弹性无接触，不认证所有构型 |
| 参照与材料域 | 新27/125点、壳能量质量、峰值/模式各项见执行报告；虚域实际翻转保留，有限采样不认证全过程 |
| 第4步收口 | 当前唯一下一项；依据实测范围选择有区分力的后续候选，无自动扫描/复杂模型 |
| 完整20%梯度 | 新核尚未认证；原NH JVP +76.093668对FD −10.797551N/mm符号失败保持；厚度粗变化不认证导数 |

本轮仅两条新JAX完整路径和两项Abaqus Explicit，无新完整AD/训练、接触/塑性、网格/积分/η/界面扫描；维护一套共享FEM和既有显式入口，未改运行核。每个物理厚度同步改变两侧，不能只改JAX去对原0.50mm壳。

新证据：WSL `validation/thickness_range_20261006_r13`。旧r12速率与局部导数证据按原范围保持；0.50mm慢路径反力/曲线/功差6.85%/5.05%/5.98%，快慢2%门槛通过。壳不是实体真值，约10%仍是项目工作目标；最新参考质量见新工况，旧壳1.33%能量漂移未过旧1%门槛保持。

曲面参数、构型和材料值得扩大验证，下一候选优先一个几何变化，需要匹配中面/真实厚度和参照。全路径梯度是未来独立关口，不把前向结果等同于可直接训练。文件位置见[地图](FILE_MAP.md)。
''')

p=W/'RESEARCH_BACKGROUND.md';s=p.read_text()
s=replace_once(s,'限定的弹性周期前向工具有较强继续依据；相邻厚度、跨形态、完整20%导数、高维反向、接触/塑性替代尚待验证。',
               brief+' 限定的弹性周期前向工具有继续依据；跨形态、完整20%导数、高维反向、接触/塑性替代尚待验证。')
s=replace_once(s,'当前下一项只执行主规划第3步的匹配厚度范围检验，复杂独立三维参照后置。',
               '当前只看主规划第4步的范围收口与后续关口选择；曲面/构型/本构是未来候选，不自动全面扫描，复杂独立三维参照后置。')
p.write_text(s)

p=W/'TPMS_RESEARCH_REVIEW.md';s=p.read_text()
s=replace_once(s,'现在最需要补的是“同一固定方法是否只在这一个厚度有效”。',
               brief+' 方法、峰值/模式、材料与参照质量的逐项说明见[厚度执行报告](THICKNESS_RANGE_PROGRESS.md)。\n\n本次已补“同一固定方法是否只在这一个厚度有效”。')
s=replace_once(s,'下一项维持原中面、网格、本构、边界和候选常数，做0.45/0.50/0.55mm匹配壳工况；0.50mm复用。',
               '执行时维持原中面、网格、本构、边界和候选常数，新增0.45/0.55mm匹配壳工况；0.50mm复用。')
s=replace_once(s,'近期仍是原四步架构：定位已完成、单因素修复已完成、有限物理工况待做、范围/未来梯度待收口。',
               '曲面参数、构型和本构都可继续研究：先用一个已有匹配中面的几何变化检验迁移，再按实际需要扩大材料范围。c与真实壁厚需分开；仅改变E可能主要缩放力级，ν或能量形式改变更能检验新机制，但也需要重新验证匹配本构及虚域处理。它们是下一阶段的候选，不是本轮扫描任务；换构型前向成功不等于形态梯度认证。\n\n近期仍是原四步架构：定位、单因素修复、有限物理工况已执行，范围/未来梯度待收口。')
p.write_text(s)

p=W/'FILE_MAP.md';s=p.read_text()
s=s.replace('唯一四步及当前第3步','唯一四步及当前第4步')
s=s.replace('| VOID_CONTINUATION_PROGRESS.md |','| THICKNESS_RANGE_PROGRESS.md | r13匹配0.45/0.55mm完整20%及范围；docs同名 |\n| VOID_CONTINUATION_PROGRESS.md |')
s=s.replace('| validation/void_continuation_20261006_r12/ |','| validation/thickness_range_20261006_r13/ | 匹配厚度输入、两条完整20%、两份壳参照、27/125点与模式、范围决定及哈希 |\n| validation/void_continuation_20261006_r12/ |')
s=s.replace('本轮零新Abaqus作业。','r13新增0.45/0.55mm各一个Explicit匹配壳，原模型/资料不改。')
s+='\n## 当前新增证据\n\nr13新原生作业：`E:/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus/thickness_range_20261006_r13_t0p45_explicit_T0p040`及同名`t0p55`目录。ODB留该处；正式实验各厚度`abaqus/explicit_T0p040/`有提取和retention.json。各厚度`T0p004/`含场/完整结果、`analysis/`含27/125点/模式与comparison.json；一份规范INP和厚度单行差异可复现两侧输入。Windows `work/thickness_range_20261006`仅本轮一次性准备/报告/发布收据，不是FEM入口。\n'
p.write_text(s)

p=W/'START_HERE.md';s=p.read_text();s=replace_once(s,'第3步待做少量匹配厚度工况。',brief+' 当前只看第4步收口。');s=s.replace('work仅放当前整理收据','work仅放当前实验/整理收据');s=s.replace('最新科学细节见[冻结执行报告](VOID_CONTINUATION_PROGRESS.md)','最新科学细节见[厚度执行报告](THICKNESS_RANGE_PROGRESS.md)，核修复见[冻结执行报告](VOID_CONTINUATION_PROGRESS.md)');p.write_text(s)

p=W/'AGENTS.md';s=p.read_text();old='前两步完成，下一项仅第3步：固定objective_void+C²候选，匹配同中面0.45/0.50/0.55mm壳/JAX工况，0.50mm复用。'
s=replace_once(s,old,'前三步已执行；r13匹配同中面0.45/0.55mm壳/JAX完整20%，0.50mm复用，结果见THICKNESS_RANGE_PROGRESS.md。下一项仅第4步收口范围与未来关口；曲面参数/构型/材料为后续候选，优先一个有区分力且有匹配中面的几何变化，不自动广泛扫描。')
p.write_text(s)

mapping={'RESEARCH_BACKGROUND.md':'docs/RESEARCH_BACKGROUND.md','RESEARCH_PLAN.md':'docs/RESEARCH_PLAN.md',
         'PROJECT_OVERVIEW.md':'docs/RESEARCH_STATUS.md','FILE_MAP.md':'docs/FILE_MAP.md',
         'TPMS_RESEARCH_REVIEW.md':'docs/TPMS_RESEARCH_REVIEW.md','START_HERE.md':'README.md','AGENTS.md':'AGENTS.md'}
for src,dest in mapping.items():
    text=(W/src).read_text().replace('PROJECT_OVERVIEW.md','RESEARCH_STATUS.md').replace('output/figures/','figures/')
    if dest=='docs/FILE_MAP.md':
        text=text.replace('[归档索引](history/completed_tools_20261006/README.md)',
                          '`Windows history/completed_tools_20261006/README.md`（Windows归档索引）')
    if dest=='docs/TPMS_RESEARCH_REVIEW.md':
        text=text.replace('figures/VOID_CONTINUATION_response.png',
                          '../validation/void_continuation_20261006_r12/response.png')
    if dest in ['AGENTS.md','README.md']:
        for name in ['RESEARCH_BACKGROUND.md','RESEARCH_PLAN.md','RESEARCH_STATUS.md','FILE_MAP.md',
                     'TPMS_RESEARCH_REVIEW.md','VOID_CONTINUATION_PROGRESS.md','THICKNESS_RANGE_PROGRESS.md','PAPER_ROUTE.md']:
            text=text.replace(']('+name+')','](docs/'+name+')')
    (R/dest).write_text(text)

p=R/'validation/README.md';s=p.read_text();s=s.replace('| [void_continuation_20261006_r12]',
    '| [thickness_range_20261006_r13](thickness_range_20261006_r13/README.md) | 固定C²候选，0.45/0.55mm匹配壳与完整20%，0.50mm复用；反力/曲线/功、峰值/模式与27/125点见decision |\n| [void_continuation_20261006_r12]')
s=s.replace('当前第1/2步完成，第3步固定C²候选做匹配厚度','当前第1/2/3步已执行，第4步收口范围/未来关口')
s=s.replace('本次仅阅读/规范化，没有新增科学作业。','本次新增两条JAX前向及两个匹配壳Explicit，无完整AD/训练；详见r13。')
p.write_text(s)
p=R/'scripts/README.md';s=p.read_text().replace('第3步匹配厚度','第4步范围收口')
s+='\n匹配厚度r13沿用同一入口，在上列代表命令加`--thickness-mm 0.45`或`0.55`并使用全新输出。占据和HRZ质量随真实厚度计算；不能改厚度后仍对原0.50mm壳。新核/时间算法未改，r13原证据不覆盖；读取详细口径见[厚度报告](../docs/THICKNESS_RANGE_PROGRESS.md)。\n'
p.write_text(s)
p=R/'.gitignore';s=p.read_text();s+='\n# Matched physical thickness: publish one canonical deck; complete pair stays local.\nvalidation/thickness_range_20261006_r13/**/*.inp\n!validation/thickness_range_20261006_r13/t0p45/abaqus/explicit_T0p040/thin_shell.inp\nvalidation/thickness_range_20261006_r13/**/progress.json\nvalidation/thickness_range_20261006_r13/**/extract_thin_explicit.py\n';p.write_text(s)
(O/'README.md').write_text('''# 固定方法的匹配厚度范围

本轮执行唯一主规划第3步：同diverse_28中面、XYZ周期、真实NH，0.45/0.55mm双方匹配；0.50mm复用r12。输入/门槛先固定，无数值参数拟合和新AD/训练。

解释/结论见[执行报告](../../docs/THICKNESS_RANGE_PROGRESS.md)，任务只看[主规划](../../docs/RESEARCH_PLAN.md)。input/preparation先冻结；各厚度T0p004为前向及源码/场，abaqus/explicit_T0p040为提取/留存，analysis为完整响应、模式和27/125点。decision、verification、evidence_manifest为范围及哈希。

tools仅一次性调度/分析过程，不是第二套FEM；分析复用r12保存场函数并显式对齐物理厚度和匹配壳路径。厚度范围不是梯度认证。大场、日志、重复INP及提取器副本留本机，规范0.45mm INP和preparation的厚度单行差异可复现0.55mm；ODB留E盘独立作业目录。冻结原r6-r12不改。
''')
(W/'work/README.md').write_text('''# 当前过程收据

正式程序仅WSL `/home/xuehu/projects/tpms_jax`。`thickness_range_20261006`为r13一次性匹配厚度准备、运行、分析、文档/发布过程，零新FEM实现；`research_synthesis_20261006`为上一轮已完成阅读/整理收据。均不自动重跑，科学入口只看[地图](../FILE_MAP.md)，任务只看[唯一规划](../RESEARCH_PLAN.md)。

更早完成工具在[历史](../history/completed_tools_20261006/README.md)，旧科学字节与用户论文/ODB不变。本轮结果、程序和参照路径见[厚度报告](../THICKNESS_RANGE_PROGRESS.md)。
''')
print(brief)

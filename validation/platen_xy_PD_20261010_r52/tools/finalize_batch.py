"""Summarize existing r52 results, plot G/P/D, and update navigation. No solver calls."""
from pathlib import Path
import csv,hashlib,json,re,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

B=Path(__file__).resolve().parents[1];R=B.parents[1]
W=Path(r'C:\Users\xuehu\Documents\Codex\2026-10-01\referenced-chatgpt-conversation-this-is-an')
O=W/'output/r52_platen_xy_PD_20261010'
GE=R/'validation/platen_xy_G_20261010_r51/experiments/midsurface_rim_20pct_20261010'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pc(x):return '未取得' if x is None else f'{x*100:.3f}%'

def navigation(conclusion,table):
    """Update current navigation only; dated experiments and Word/PPT stay frozen."""
    receipt={}
    def update(p,s):
        before=sha(p);p.write_text(s,encoding='utf-8');receipt[str(p)]={'before':before,'after':sha(p)}
    for root,formal in [(W,False),(R,True)]:
        docs=root/'docs' if formal else root
        link='validation/platen_xy_PD_20261010_r52/README.md' if formal else 'output/r52_platen_xy_PD_20261010/README.md'
        status=docs/('RESEARCH_STATUS.md' if formal else 'PROJECT_OVERVIEW.md')
        s=status.read_text(encoding='utf-8');s=s.replace('# 当前状态：G共同端缘加载20%对照达到本轮目标','# 当前状态：G/P共同端部20%对照通过；D壳参考失败',1)
        s=s.replace('## 当前结论','## r52跨构型结果\n\n'+conclusion+' 详细曲线、实际范围和成本见[本轮报告]('+('../' if formal else '')+link+')。\n\n| 构型 | 完整成对20%＋保载 | 曲线RMS | 加载功差 | 保载均力差 | 三项目标 |\n| --- | --- | ---: | ---: | ---: | --- |\n'+table+'\n\n## G已完成的原r51结论',1)
        update(status,s)
        plan=docs/'RESEARCH_PLAN.md';s=plan.read_text(encoding='utf-8');s=s.replace('当前没有P/D新压板结果。','实际P结果与D失败／JAX路径已保存；本轮已收口，没有自动追加求解。以下条件及流程保留启动前冻结口径，实际停点以报告为准。',1).replace('[冻结方案与检查]','[冻结方案、结果与失败]',1).replace('按对应构型启动时创建。','现已分别创建，原生失败ODB保留。',1).replace('目录拟为','目录为',1).replace('Windows只保留方案阅读副本','Windows保留方案与结果阅读副本',1)
        update(plan,s)
        agents=root/'AGENTS.md';s=agents.read_text(encoding='utf-8');s=re.sub(r'更新2026-10-10。当前用户已授权.*?当前作业以r52/status.json为准。','更新2026-10-10。用户授权的P、D共同中面端部压板／XY周期／20%四条动态路径已各尝试一次，两个datacheck完成，每种沿用原100min总预算。'+conclusion+' 当前无活动作业；无额外重算、参数扫描、设计AD或训练授权。最新用户指令优先，历史建议不自动成为待办。实际停点见r52/status.json。',s,count=1)
        s=s.replace('## 当前科研结论','## 当前科研结论\n\n'+conclusion+' [跨构型证据]('+link+')。D失败保留，局部诊断不升级为完整20%验收。',1);update(agents,s)
        entry=root/('README.md' if formal else 'START_HERE.md');s=entry.read_text(encoding='utf-8');s=re.sub(r'当前任务为P/D成对执行。P/D已确认执行并准备／推进中，实际状态见r52/status.json；无新设计AD。',conclusion+' 当前无活动作业；实际停点见[r52结果]('+link+')，没有新设计AD。',s,count=1)
        s=s.replace('说明已确认执行的P/D20%成对方案','保存P/D20%冻结方案及实际收口状态').replace('每种总预算100min；按P后D推进。','每种总预算100min，实际成本见报告；本轮已收口。');update(entry,s)
        fmap=docs/'FILE_MAP.md';s=fmap.read_text(encoding='utf-8').replace('当前任务为r52 P/D成对执行，实际状态见批次status.json。','r52 P/D本轮已分析收口，实际范围与失败见批次status.json。',1).replace('## P/D成对执行（r52）','## P/D成对结果与失败（r52）',1)
        s=s.replace('结果随实际执行写入，状态以status.json为准','P完整20%通过；D的壳在约18.4%畸变失败，JAX实际范围、原始失败历史及固定0–15%诊断独立记录，状态以status.json为准',1)
        s=s.replace('work/与tmp/仅保留说明，无活动脚本。','本轮Windows编辑配方完成后归档到history/execution_platen_xy_PD_20261010_r52，work/与tmp/仅留说明；正式程序仍仅在WSL。',1);update(fmap,s)
        bg=docs/'RESEARCH_BACKGROUND.md';s=bg.read_text(encoding='utf-8');s=s.replace('## 固定的方法边界','## 固定的方法边界')
        para='\n\nr52跨构型更新：'+conclusion+' 这增强了受控共同加载下的前向可行性证据；尚未证明通用构型、完整有限厚度接触或20%有效设计总梯度。实际证据见[当前状态]('+('RESEARCH_STATUS.md' if formal else 'PROJECT_OVERVIEW.md')+')。\n'
        lines=s.splitlines();idx=next((i for i,x in enumerate(lines) if i>1 and x.startswith('## ')),len(lines));lines.insert(idx,para);update(bg,'\n'.join(lines)+'\n')
        review=docs/'TPMS_RESEARCH_REVIEW.md';s=review.read_text(encoding='utf-8');s=s.replace('## 2.','r52补充：'+conclusion+' 固定窗口诊断与完整20%验收分开记录，详见[跨构型报告]('+('../' if formal else '')+link+')。\n\n## 2.',1);update(review,s)
    p=W/'output/README.md';s=p.read_text(encoding='utf-8');s=re.sub(r'\[r52 P/D共同端部压板20%对照\].*?每构型JAX＋Abaqus＋准备分析总预算100min。','[r52 P/D共同端部压板20%对照](r52_platen_xy_PD_20261010/README.md)：'+conclusion+' 每种实际总成本和失败记录见该报告，当前无活动作业。',s,count=1);update(p,s)
    p=R/'validation/README.md';s=p.read_text(encoding='utf-8');s=s.replace('无活动任务；P/D共同端部压板20%方案已保存但未提交，无新求解／AD。','当前r52已执行收口；无活动任务或新设计AD。',1);s=re.sub(r'待审阅：\[r52 P/D压板20%方案\].*?当前仅只读检查，无INP／新结果。','最新：[r52 P/D压板20%结果](platen_xy_PD_20261010_r52/README.md)。'+conclusion,s,count=1);update(p,s)
    write(B/'checks/navigation_closeout_receipt.json',receipt)

def main():
    state=read(B/'status.json');assert state['active_jobs']==[],state
    out=B/'analysis';out.mkdir(exist_ok=True)
    rows=[]
    for kind,e in [('G',GE),('P',B/'P'),('D',B/'D')]:
        if not (e/'analysis/metrics.json').exists():
            rows.append({'kind':kind,'full20':False,'RMS':None,'work_bias':None,'hold_bias':None,'targets_pass':False});continue
        m=read(e/'analysis/metrics.json')
        rows.append({'kind':kind,'full20':m['paired_full20_and_hold'],'RMS':m['curve_RMS_over_new_shell_peak'],
            'work_bias':m['common_loading_work_N_mm']['relative_bias'],
            'hold_bias':m.get('hold_last0p1ms',{}).get('force_N',{}).get('relative_bias'),
            'targets_pass':m['three_scalar_targets_within_10pct']})
    plt.rcParams.update({'font.size':10,'figure.dpi':160,'savefig.dpi':180,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
    for ax,(kind,e) in zip(axs,[('G',GE),('P',B/'P'),('D',B/'D')]):
        f=e/'analysis/force_comparison.csv'
        if not f.exists():ax.text(.5,.5,'No paired curve',ha='center',va='center',transform=ax.transAxes);ax.set_title(kind);continue
        data=np.genfromtxt(f,delimiter=',',names=True)
        ax.plot(data['engineering_compression']*100,data['Abaqus_upper_force_N'],color='#285c8e',label='Abaqus S3R / Explicit')
        ax.plot(data['engineering_compression']*100,data['JAX_upper_force_N'],color='#c45b25',ls='--',label='JAX HEX27 / Verlet')
        row=next(x for x in rows if x['kind']==kind)
        title=f"{kind} | full20 RMS {100*row['RMS']:.2f}%" if row['RMS'] is not None else f'{kind} | shell reference failed near 18.4%'
        if kind=='D':
            ax.axvspan(15,20,color='gray',alpha=.12)
            ax.text(.04,.95,'Paired diagnostics only through 15%',va='top',fontsize=8,transform=ax.transAxes)
        ax.set(xlabel='Engineering compression (%)',ylabel='Upper-platen compressive force (N)',xlim=(0,20),title=title)
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('G / P / D | common midsurface-end platen idealization | free XY periodicity\nL10 mm, t0.5 mm, NH E10 MPa, 4 ms loading + 0.2 ms hold')
    fig.savefig(out/'G_P_D_force_comparison.png');plt.close(fig)
    write(out/'G_P_D_metrics.json',rows)
    with (out/'G_P_D_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    table='\n'.join(f"| {x['kind']} | {'是' if x['full20'] else '否'} | {pc(x['RMS'])} | {pc(x['work_bias'])} | {pc(x['hold_bias'])} | {'达到' if x['targets_pass'] else '不可评价：参考失败'} |" for x in rows)
    costs=[]
    for k in ['P','D']:
        p=B/k/'analysis/costs.json'
        if p.exists():
            v=read(p)
            if k=='D':
                other=v['overall_minutes']*60-v['Abaqus_check_solve_seconds']-v['JAX_driver_seconds']
                v['other_elapsed_seconds_including_failure_review_extraction_analysis']=other
                v['10min_overhead_reserve_exceeded']=other>600
                v['budget_interpretation']='Original100min total clock and65/25min solver limits retained; failed-reference handling consumes remaining overall budget.'
                write(p,v);write(O/k/'costs.json',v)
            costs.append((k,v))
        report=B/k/'README.md'
        s=report.read_text(encoding='utf-8')
        if k=='P':s=s.replace('D包含端面相切节点。','')
        if k=='D':
            s=s.replace('总成本另见analysis/costs.json。',f"本构型总墙钟{read(p)['overall_minutes']:.2f} min，沿用原100min预算时钟；分项见analysis/costs.json。")
            diagnostic=B/'D/analysis/failed_element_geometry.json'
            if diagnostic.exists():
                geo=read(diagnostic);edges=geo['initial_edges_mm']
                s+='\n只读网格定位：报错单元初始质心约(4.119, 7.794, 0.466) mm，三边约'+', '.join(f'{v:.4f}' for v in edges)+' mm，面积位于原网格约3.69%分位。它靠近底端且较细长，这是后续定点检查的线索；初始形状及有限位移帧不足以独立认定失效原因。详见analysis/failed_element_geometry.json。\n'
        report.write_text(s,encoding='utf-8')
        (O/k/'README.md').write_text(s.replace('](analysis/','](').replace('analysis/costs.json','costs.json').replace('analysis/failed_element_geometry.json','failed_element_geometry.json'),encoding='utf-8')
    costrows='\n'.join(f"| {k} | {v['overall_minutes']:.2f} | {'是' if v['within_100min'] else '否'} |" for k,v in costs)
    new=rows[1:];done=sum(x['full20'] for x in new);passed=sum(x['targets_pass'] for x in new)
    dm=read(B/'D/analysis/metrics.json');dh=dm['JAX_sampled_health']
    dlast=dm['partial_diagnostic_windows'][-1] if dm['partial_diagnostic_windows'] else None
    dnote=(f"D的Abaqus壳在约18.4%压缩时因单元16329过度畸变停止，完整20%成对验收不可评价；D/JAX{'完成20%和保载' if dm['JAX_complete'] else '仅完成部分路径'}。")
    conclusion=(f"P完成20%与保载，曲线RMS2.78%、加载功差3.65%、保载均力差4.80%，三项达到约10%工作目标。"+dnote)
    text=f'''# P、D跨构型验证：共同中面端部压板／XY周期／20%

更新2026-10-10。**{conclusion}** 各构型的曲线、变形、能量与实际范围分别见[P报告](P/README.md)、[D报告](D/README.md)；不能把一个构型的接近外推到所有TPMS。本次没有新设计AD或训练。

| 构型 | 完整20%＋保载 | 全曲线RMS | 加载功偏差 | 保载均力偏差 | 三项约10%目标 |
| --- | --- | ---: | ---: | ---: | --- |
{table}

全曲线RMS以各自本次Abaqus加载峰为分母，是力差均方根而非逐点误差上限；功、保载均力分别以各自新壳结果为分母。D三个完整路径指标留空，不能以参考失败宣布通过或判定JAX不准确。G为此前已完成的独立实验，仅加入图表比较，未重算或改写。

{dnote} D在提交JAX前固定0–2%、0–10%、0–15%诊断窗口，逐项结果见D报告；窗口指标以自身窗口峰为分母，不能与G/P全20%直接排名。图中D仅画至15%的成对诊断，完整JAX及失效前ODB历史另见D报告。ODB未写入致命增量的历史能量，STA最终动能／总能量6.154e8 Nmm与失败特殊帧另行保留。没有自动重算、改网格或更换材料。

D初段2%、5%、10%点差约−2.03%、−2.24%、−1.29%，15%已增至约+9.51%；分歧早于最终数值失败，不能都归于致命增量。G/P均主要持续增载，因此它们的通过尚未验证复杂后屈曲的通用准确性。当前结论是共同加载下的前向能力已从G迁移到P，D中后段仍需可靠参考与模式分析，不能宣布所有TPMS已验证通过。

![G/P/D共同加载下的实际曲线](G_P_D_force_comparison.png)

## 实际条件与接触范围

L10 mm、t0.5 mm、NH E10 MPa/ν0.3/密度1e−9 tonne/mm³；JAX N32 HEX27/27点、HRZ、连续占据／η1e−4／界面0.05 mm／objective_void/C²。Abaqus S3R壳截面5点；P沿用质量修整后的三角连接，D沿用原网格，JAX均保留原距离表面与Gauss数组。相同厚度不等于相同体积分数。

两端采用4 ms五次平滑下压2 mm及0.2 ms保载，XY周期、横向宏观伸缩自由、宏观剪切零，无Z周期和额外壁节点固定点。沿用G中面端部节点接触理想化：P每端116节点、D每端254节点（含2个端面相切节点）；JAX自身占据加权面积聚合到相同初始位置，kn320 MPa/mm，不调材料、厚度、η或阻尼拟合。

壳截面厚度仍参与刚度与质量；接触不覆盖完整厚度端面，也没有全壁面接触、自接触或周期镜像接触。两端的运动学、罚律、接触插值及惯性离散仍不同。约10%是本范围的工作目标，壳不是绝对真值；曲线接近不自动认证真实接触或有效设计总梯度。

## 成本与失败记录

| 构型 | 从原输入准备计时的总分钟 | 是否在100min内 |
| --- | ---: | --- |
{costrows}

按P后D、各自Abaqus后JAX执行，无资源竞争、参数扫描或额外短程动力筛选。P首次准备中将Element Output误当Element卡，启动前检查捕捉后修复；没有提交错误INP或求解。失败准备文件逐字节归档，沿用原P时钟；修复耗时计入上表。D在Abaqus失败后仍执行原计划尚未提交的第四条JAX路径，沿用原D时钟，未追加Abaqus尝试。失败原路径与ODB保留，不冒充完整成对20%。

D失败记录提取、定位和处置占用了额外非求解时间，10min准备／分析预留有所超出；这部分已计入原100min总墙钟，详见costs.json中的other_elapsed。JAX65min、Abaqus25min和每构型总100min限额均未延长或重新计时。

## 文件入口

正式批次`/home/xuehu/projects/tpms_jax/validation/platen_xy_PD_20261010_r52`。每构型input保存实际参数、端面及端部接触数组，abaqus保存独立INP、提取路径与位移帧，jax保存原路径与有效状态，analysis保存图表／指标／成本，checks保存时钟、启动收据和输入审计。原生INP／ODB在E:/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus/r52_P_midsurface_end_20pct_20261010及D对应目录。

protocol_planned.json是启动前快照，protocol.json为授权执行记录，当前实际停点只见status.json。原G、r50及历史失败数据保持原字节。完整实施源码与后处理变换凭据均在正式仓库，Windows仅为阅读副本。下轮任务需根据本轮证据另行决定，不自动追加。
'''
    for p in out.iterdir():
        if p.is_file():shutil.copy2(p,O/p.name)
    (O/'README.md').write_text(text,encoding='utf-8')
    (B/'README.md').write_text(text.replace('](G_P_D_force_comparison.png)','](analysis/G_P_D_force_comparison.png)'),encoding='utf-8')
    protected=read(B/'checks/source_guard_before.json');changed=[p for p,h in protected.items() if sha(R/p)!=h];assert not changed,changed
    write(B/'checks/final_integrity.json',{'protected_sources':len(protected),'changed':changed,'new_design_AD':False,
        'tools_sha256':{p.name:sha(p) for p in (B/'tools').glob('*.py')}})
    state.update(stage='batch_analyzed',active_jobs=[],next_action='No automatic next task; discuss scientific results first',complete_pairs=done,passing_pairs=passed)
    write(B/'status.json',state);write(O/'status.json',state)
    note=conclusion+' 实际范围、成本与曲线见r52报告。当前无活动作业，无新设计AD或训练；G结果和原XYZ历史保持。'
    for p,link in [(W/'PROJECT_OVERVIEW.md','output/r52_platen_xy_PD_20261010/README.md'),(R/'docs/RESEARCH_STATUS.md','../validation/platen_xy_PD_20261010_r52/README.md')]:
        s=p.read_text(encoding='utf-8');s=re.sub(r'G本轮计算已完整结束。最新安排为P、D.*?长期问题与范围见',note+' [P/D结果]('+link+')。长期问题与范围见',s,count=1)
        p.write_text(s,encoding='utf-8')
    for p in [W/'RESEARCH_PLAN.md',R/'docs/RESEARCH_PLAN.md']:
        s=p.read_text(encoding='utf-8').replace('（已确认执行，准备启动）','（已执行并分析）',1)
        s=s.replace('**P后D，每种Abaqus后JAX；当前启动准备中，实际进展与结果见r52/status.json。**','**'+note+'**')
        p.write_text(s,encoding='utf-8')
    navigation(conclusion,table)
    # Windows keeps one reading copy and a recoverable editing archive.
    staging=W/'work/r52_tools';archive=W/'history/execution_platen_xy_PD_20261010_r52/tools_editing';archive.mkdir(exist_ok=True)
    assert staging.resolve().is_relative_to(W.resolve()) and archive.resolve().is_relative_to(W.resolve())
    manifest={}
    for p in list(staging.iterdir()):
        assert p.is_file() and p.suffix=='.py',p
        target=archive/p.name;assert not target.exists();shutil.copy2(p,target);assert sha(target)==sha(p)
        manifest[p.name]=sha(p);p.unlink()
    staging.rmdir();write(archive/'manifest.json',manifest)
    (W/'work/README.md').write_text('# 临时工作区\n\n当前无活动作业或待执行脚本。本轮r52编辑配方已逐字节校验归档到history/execution_platen_xy_PD_20261010_r52/tools_editing；正式运行版本仅在WSL validation/platen_xy_PD_20261010_r52/tools。当前结果从START_HERE.md进入。\n',encoding='utf-8')
    write(B/'checks/Windows_editing_archive.json',{'archive':str(archive),'sha256':manifest,'single_file_removal_after_copy_verification':True,'original_data_removed':False})
    print(json.dumps({'conclusion':conclusion,'rows':rows,'costs':costs,'protected_sources':len(protected)},ensure_ascii=False,indent=2))

if __name__=='__main__':main()

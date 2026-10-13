"""Publish local scientific records after this one pair; no solver or Git calls."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,re,shutil,time
W=Path(r'C:\Users\xuehu\Documents\Codex\2026-10-01\referenced-chatgpt-conversation-this-is-an')
R=Path(r'\\wsl.localhost\Ubuntu-24.04\home\xuehu\projects\tpms_jax')
D=R/'validation/platen_xy_G_20261010_r51';E=D/'experiments/midsurface_rim_20pct_20261010'
O=W/'output/r51_platen_xy_G_20261010/midsurface_rim_20pct_20261010'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,s):p.write_text(s.rstrip()+'\n',encoding='utf-8')
def write(p,x):save(p,json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def pct(v):return '—' if v is None else f'{v*100:.3f}%'
def main():
    m=read(E/'analysis/metrics.json');j=read(E/'jax/result.json');clock=read(E/'checks/execution_clock.json')
    elapsed=(time.time()-clock['started_unix'])/60;abq=read(E/'checks/abaqus_complete.json')['wall_seconds_including_preparation']/60
    jm=j['wall_seconds']/60;complete=m['paired_full20_and_hold'];passed=m['three_scalar_targets_within_10pct']
    assessment=('完整20%及保载已完成，全曲线、功与保载三个预定标量目标均在约10%内。' if passed else
                ('完整20%及保载已完成，但三个预定标量目标未全部达到约10%。' if complete else 'JAX仅完成部分路径，完整20%及保载尚未通过；下面只报告实际达到范围。'))
    hold=m.get('hold_last0p1ms',{}).get('force_N',{});h=hold.get('relative_bias')
    protected=read(E/'checks/protected_before.json');changed=[p for p,v in protected.items() if sha(R/p)!=v];assert not changed,changed
    costs={'started_UTC':clock['started_UTC'],'finished_UTC':datetime.now(timezone.utc).isoformat(),'elapsed_overall_minutes':elapsed,
           'Abaqus_preparation_check_solve_extract_minutes':abq,'JAX_build_compile_checks_advance_minutes':jm,
           'extra_field_extraction_minutes':read(E/'checks/frame_extract_cost.json')['wall_seconds']/60,
           'overall_90min_met':elapsed<=90,'JAX_55min_met':jm<=55.25,'Abaqus_25min_met':abq<=25.25}
    write(E/'analysis/costs.json',costs)
    write(E/'checks/final_integrity.json',{'protected_scientific_files':len(protected),'changed_scientific_files':changed,
          'original_source_preserved':True,'current_tools_sha256':{p.name:sha(p) for p in (E/'tools').glob('*.py')},'no_new_design_AD':True})
    for p in (E/'analysis').iterdir():
        if p.is_file():shutil.copy2(p,O/p.name);assert sha(p)==sha(O/p.name)
    shutil.copy2(E/'input/protocol.json',O/'protocol.json')
    rows='\n'.join(f"| {x['compression']*100:.0f}% | {x['JAX_N']:.6f} | {x['Abaqus_N']:.6f} | {pct(x['relative_bias'])} |" for x in m['sample_force'])
    segments='\n'.join(f"| {x['compression'][0]*100:.0f}–{x['compression'][1]*100:.0f}% | {pct(x['RMS_over_same_full20_shell_peak'])} |" for x in m['segments'])
    fields='\n'.join(f"| {x['target_pct']}% | {x['displacement_RMS_mm']:.6f} | {pct(x['relative_displacement_RMS'])} | {x['JAX_finite_width_cut_face_outside_platen_max_mm']['0.5']:.6f} |" for x in m['field_comparison'])
    p=m['peak_in_compared_window'];health=m['JAX_sampled_health']
    text=f'''# G中面端缘等效压板20%对照：实际结果

更新2026-10-10。**{assessment}** 本轮为用户明确选择的端缘简化边界前向比较，不代表完整有限厚度端面或自接触试验。原有限宽G 2%未通过的事实与全部历史结果保持。

## 做了什么

仅G一组，L=10 mm、t=0.5 mm、NH E=10 MPa/ν=0.3/密度1e−9 tonne/mm³。Abaqus保留原S3R壳，JAX保留HEX27 N32、27点、连续占据、η=1e−4、界面0.05 mm、HRZ及原objective_void/C²材料核。双方从零用4 ms五次平滑下压2 mm，到20%，再保载0.2 ms；XY周期、横向ax/ay自由、无Z周期。

JAX仅把原占据端面权重聚合到已有壳中面端缘，复用原接触数组和kn=320 MPa/mm，没有改体内材料/质量或拟合参数。Abaqus保留原节点端缘的自动罚接触。接触允许分离、切向无摩擦，端缘并未绑定到压板。两端接触支持趋同，但壳/实体运动学、连续占据、接触插值/罚律及惯性仍不同，残差不能全部归为体内离散误差。

## 结果与指标解释

实际可比较加载范围0–{m['compared_compression_max']*100:.4f}%。全曲线RMS为力差均方根除以本次壳加载峰{m['primary_shell_full20_peak_N']:.6f} N；它不是每点误差上限。旧2%壳峰0.724118 N分母仅保留追溯，本次不据其验收。

| 指标 | 实际结果 |
| --- | --- |
| 全曲线RMS / 本次壳峰 | {pct(m['curve_RMS_over_new_shell_peak'])} |
| 同窗口加载功偏差 JAX/壳−1 | {pct(m['common_loading_work_N_mm']['relative_bias'])} |
| 末0.1 ms保载上板均力偏差 | {pct(h)} |
| 加载最大力偏差 | {pct(m['peak_force_relative_bias'])} |
| 加载最大力位置差 | {m['peak_compression_difference_percentage_points']:.4f} 个压缩量百分点 |

加载最大点不自动等于首次屈曲峰，具体过程看原始曲线，不能只用一个峰值判断路径一致。

![完整曲线、初段、保载与横向响应](force_and_lateral.png)

| 工程压缩量 | JAX上板力 N | 壳上板力 N | 相对偏差 |
| --- | --- | --- | --- |
{rows}

| 分段 | RMS / 同一本次壳峰 |
| --- | --- |
{segments}

## 变形和有限厚度边界的限制

变形用原中面位置比较，JAX采用完整三维Q2插值；Abaqus位移在相邻输出帧之间按相同时间插值，时间间隔见field_comparison.json。未做刚体对齐、放大或缩放。最后一列是JAX参考平切端面高占据点φ≥0.5越过板面的最大值；它不等于被约束端缘点的罚接触穿透。这一差别是端缘加载的物理简化代价。

| 对应压缩量 | 位移差RMS mm | 位移差/壳位移RMS | 实体端面越板最大 mm |
| --- | --- | --- | --- |
{fields}

![实际位移尺度下的变形对照](deformation_10_20pct.png)

无完整壁面压板接触、无自接触或周期镜像接触。本次没有执行全过程自交搜索，不能把正材料J或相近曲线当作无接触证明。真实有限厚度实体参考仍为延期候选，没有并行生成或提交。

## 数值状态与成本

JAX接受块采样中材料域最小实际J={health['minimum_required_actual_J']:.6f}，全域最小实际J={health['minimum_all_actual_J']:.6f}，虚域负J采样点最大数{health['max_negative_void_J_samples']}。没有夹断真实F/J。原始拒绝、减步及5/10/15/20%局部稳定步长估计见jax目录；这些是有限采样及局部估计，不是全过程严格稳定性证明。

JAX采样功—能量偏差/最大采样功为{pct(health['energy_work_defect_over_max_sampled_work'])}；壳ETOTAL漂移/外功为{pct(m['Abaqus_health']['energy_drift_relative_work'])}。两指标定义不同，不直接当作软件精度排名。动能变化按动态路径解释，不预先称准静态。

![能量与实际J记录](energy_and_J.png)

| 成本 | 墙钟分钟 |
| --- | --- |
| Abaqus输入准备、检查、求解与原提取 | {abq:.2f} |
| JAX构建、编译、检查和推进 | {jm:.2f} |
| 额外只读位移帧提取 | {costs['extra_field_extraction_minutes']:.2f} |
| 本轮启动至结果整理的整体已用时间 | {elapsed:.2f} |

额外提取与JAX前期有时间重叠，所以分项之和不等于总墙钟。全部原科学保护文件{len(protected)}项未变。原有限宽2%对照、旧20%范围及旧设计梯度失败记录保留；本轮未验证新设计总梯度。

正式输入/状态/原始路径位于WSL `/home/xuehu/projects/tpms_jax/validation/platen_xy_G_20261010_r51/experiments/midsurface_rim_20pct_20261010`；Abaqus原生INP与ODB在E:/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus/r51_G_midsurface_rim_20pct_20261010。原生platen.inp与正式abaqus/platen_20pct.inp哈希对应。metrics.json、costs.json、force_comparison.csv和逐端路径CSV随本报告保留。
'''
    if not (O/'deformation_10_20pct.png').exists():
        text=text.replace('![实际位移尺度下的变形对照](deformation_10_20pct.png)','本次路径未达到预定变形保存点，无对应变形图。')
    save(O/'README.md',text)
    formal=re.sub(r'\]\(([^)]+\.png)\)',r'](analysis/\1)',text)
    save(E/'README.md',formal)
    active={'stage':'complete_pair_analyzed' if complete else 'partial_pair_analyzed','active_jobs':[],'full20_completed':complete,
            'three_scalar_targets_within_10pct':passed,'new_design_AD':False,'curve_RMS_over_shell_peak':m['curve_RMS_over_new_shell_peak'],
            'work_relative_bias':m['common_loading_work_N_mm']['relative_bias'],'hold_mean_relative_bias':h,'overall_minutes':elapsed,
            'physical_finite_width_contact_validated':False,'report':'README.md'}
    write(E/'status.json',active)
    status=read(D/'status.json');status['active_jobs']=[];status['latest_activity']='midsurface_rim_20pct_pair_analyzed';status['midsurface_rim_20pct'].update(active);write(D/'status.json',status)
    note=f'2026-10-10端缘简化20%结果：{assessment} 全曲线RMS{pct(m["curve_RMS_over_new_shell_peak"])}、功差{pct(m["common_loading_work_N_mm"]["relative_bias"])}、保载均力差{pct(h)}；整体{elapsed:.2f} min。无新设计AD，无活动任务。原有限宽G未过状态保持。'
    for p,link in [(W/'START_HERE.md','output/r51_platen_xy_G_20261010/midsurface_rim_20pct_20261010/README.md'),
                   (R/'README.md','validation/platen_xy_G_20261010_r51/experiments/midsurface_rim_20pct_20261010/README.md'),
                   (W/'output/r51_platen_xy_G_20261010/README.md','midsurface_rim_20pct_20261010/README.md'),(D/'README.md','experiments/midsurface_rim_20pct_20261010/README.md')]:
        s=p.read_text(encoding='utf-8');s=re.sub(r'最新授权：[^\n]+',note+' [本轮详细结果]('+link+')。',s,count=1);save(p,s)
    for p,link in [(W/'PROJECT_OVERVIEW.md','output/r51_platen_xy_G_20261010/midsurface_rim_20pct_20261010/README.md'),
                   (R/'docs/RESEARCH_STATUS.md','../validation/platen_xy_G_20261010_r51/experiments/midsurface_rim_20pct_20261010/README.md')]:
        s=p.read_text(encoding='utf-8');s=re.sub(r'^# [^\n]+','# 当前状态：端缘20%对照已分析，具体范围见结果',s,count=1)
        s=re.sub(r'## 最新授权：[\s\S]*?(?=## 已完成证据)', '## 最新结果：中面端缘等效压板20%\n\n'+note+' [详细报告]('+link+')。\n\n',s,count=1);save(p,s)
    for p in [W/'RESEARCH_PLAN.md',R/'docs/RESEARCH_PLAN.md']:
        s=p.read_text(encoding='utf-8');s=s.replace('**2026-10-10用户已确认启动；独立INP已生成，Abaqus检查通过并推进中，JAX待其完成后启动；尚无20%成对结果。**','**本轮已执行并分析。'+note+'**')
        s=s.replace('## 三步执行方案','## 本轮三步执行方案与实际停点');save(p,s)
    for p in [W/'FILE_MAP.md',R/'docs/FILE_MAP.md']:
        s=p.read_text(encoding='utf-8').replace('现仅README及input/protocol_planned.json','已保存实际input/protocol.json、独立INP、jax/abaqus路径、analysis图表及README')
        s=s.replace('尚未建立，新INP/ODB尚不存在','已建立，原生platen.inp/platen.odb保留').replace('本轮仅定方案，当前无活动任务。','本轮已执行并分析，当前无活动任务。');save(p,s)
    for p in [W/'AGENTS.md',R/'AGENTS.md']:save(p,p.read_text(encoding='utf-8')+'\n\n补充最新执行状态：'+note+' 新端缘简化工况的结论不覆盖真实有限宽端面；详细路径见r51/experiments/midsurface_rim_20pct_20261010，未追加任何算例或设计AD。')
    write(O/'status.json',active)
    print(json.dumps({'assessment':assessment,'curve_RMS':m['curve_RMS_over_new_shell_peak'],'hold_bias':h,'costs':costs,'protected_files':len(protected),'report':str(O/'README.md')},ensure_ascii=False,indent=2))
if __name__=='__main__':main()

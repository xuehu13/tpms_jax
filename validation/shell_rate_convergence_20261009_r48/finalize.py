"""Update current navigation after r48; preserve all frozen scientific evidence."""
from pathlib import Path
import hashlib,json,time

R=Path('/home/xuehu/projects/tpms_jax')
W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
D=R/'validation/shell_rate_convergence_20261009_r48'
O=W/'output/r48_shell_rate_convergence_20261009'

def main():
    result=json.loads((D/'result.json').read_text())
    rows=json.loads((D/'rate_summary.json').read_text())
    q={r['T_ms']:r for r in rows};b=q[80]
    stable=result['two_slow_pairs_pass_frozen_peak_gates']
    scope=('完成20%加载' if b['loading_complete_20pct'] else f"30min求解预算停止，实际至{b['covered_compression_pct']:.4f}%，无完整20%/保载")
    finding=f"新增diverse_04壳T20/80ms，与既有40ms对照。20/40/80ms观察峰位分别{q[20]['recorded_max_compression_pct']:.4f}%/{q[40]['recorded_max_compression_pct']:.4f}%/{q[80]['recorded_max_compression_pct']:.4f}%，峰力{q[20]['recorded_max_force_N']:.5f}/{q[40]['recorded_max_force_N']:.5f}/{q[80]['recorded_max_force_N']:.5f}N。两次放慢{'均通过' if stable else '未均通过'}预设峰位0.2个百分点/峰力2%工作标准；只界定当前离散壳观察峰的T稳定性，不认证物理真值或JAX20%。80ms{scope}。"
    boundary='本轮仅2个新Abaqus壳作业，无JAX前向/AD，不恢复N32/N64。531项旧科学及r47记录字节保持，生产源码不改。同T4ms原JAX/壳峰位差3.4146个百分点仍存在，T不是全部差异的充分解释。'
    changed=[]
    def formal_links(s):
        mapping={
          'r48_shell_rate_convergence_20261009/README.md':'shell_rate_convergence_20261009_r48/README.md',
          'r47_simulation_atlas_20261009/README.md':'simulation_atlas_20261009_r47/README.md',
          'r46_binary_mesh_reassessment_20261009/README.md':'binary_mesh_reassessment_20261009_r46/REVIEW.md',
          'r45_binary_n64_20261009/README.md':'binary_n64_20261009_r45/REVIEW.md',
          'r44_binary_full20_20261008/README.md':'binary_full20_20261008_r44/REVIEW.md'}
        for a,b in mapping.items():s=s.replace('output/'+a,'../validation/'+b)
        return s
    def save(p,s):
        p.write_text(s,encoding='utf-8');changed.append(str(p))
    # Current state has an explicit r48 section installed before submission.
    p=W/'PROJECT_OVERVIEW.md';s=p.read_text()
    s=s.replace('下一轮见[唯一规划](RESEARCH_PLAN.md)，本次不启动计算。','这是r46历史范围；本轮r48新增壳T对照见下，不把历史“零新作业”用于当前轮。')
    s=s.replace('先查看图表，后续测试等待讨论。','其后按用户授权执行r48壳T对照，原图集保持。')
    a=s.index('## 本轮r48：');z=s.index('## 最近两条',a)
    s=s[:a]+'## 本轮r48：Abaqus壳加载时长对照已执行\n\n'+finding+'\n\n'+boundary+'\n\n[完整报告与曲线](output/r48_shell_rate_convergence_20261009/README.md)。两作业并行墙钟'+f"{result['batch_receipt']['UTC_wall_seconds']/60:.2f}min"+'，另有停止后只读提取和绘图；原提取器的峰后能量/惯性/人工能量失败保留。当前任务完成后等待讨论，不自动加T160ms或其他因素。\n\n'+s[z:]
    old='下一轮候选为已有场定位→一个同T1ms N32短对照→有需要才精确接续N64峰→一项有依据的确认/可导修正。四项均尚未执行；本次没有恢复N32或延长N64，也没有新接触/塑性/训练/AD。'
    s=s.replace(old,'旧r46候选路线已被本轮壳T诊断后置，并未自动执行。r48两项壳任务及只读后处理已完成到实际预算范围；后续先讨论证据，不自动新增仿真/AD/接触/塑性/训练。')
    save(p,s);save(R/'docs/RESEARCH_STATUS.md',formal_links(s))
    p=W/'RESEARCH_PLAN.md';s=p.read_text();marker='## 第1步：'
    s=s.replace(marker,'## 本轮执行结果（不生成新的待办）\n\n'+finding+'\n\n'+boundary+'\n\n四步已按实际覆盖范围执行并完成分析；80ms是否完整见上。运行前协议和原预算保持，下一次新增计算等待讨论。[r48完整报告](output/r48_shell_rate_convergence_20261009/README.md)。\n\n'+marker,1)
    save(p,s);save(R/'docs/RESEARCH_PLAN.md',formal_links(s))
    for p,link in [(W/'START_HERE.md','output/r48_shell_rate_convergence_20261009/README.md'),(R/'README.md','validation/shell_rate_convergence_20261009_r48/README.md')]:
        s=p.read_text();a=s.index('更新2026-10-09：');z=s.index('\n\n',a)
        s=s[:a]+'更新2026-10-09：[r48壳加载时长对照]('+link+')已执行。'+finding+'\n\n'+boundary+s[z:]
        s=s.replace('下一轮四步尚未执行，不自动接续N64或恢复N32。','r48四步分析已收口，后续等待讨论；不自动接续N64或恢复N32。')
        save(p,s)
    for p,link in [(W/'FILE_MAP.md','output/r48_shell_rate_convergence_20261009/README.md'),(R/'docs/FILE_MAP.md','../validation/shell_rate_convergence_20261009_r48/README.md')]:
        s=p.read_text();a=s.index('更新2026-10-09：');z=s.index('\n\n',a)
        s=s[:a]+'更新2026-10-09：r48两项Abaqus壳T对照已执行并分析，80ms实际范围与预算停止见[r48报告]('+link+')。r47图集及r46复盘保持。R=`/home/xuehu/projects/tpms_jax`为唯一生产仓库；原科学文件、INP/ODB和论文不搬迁。'+s[z:]
        marker='| validation/simulation_atlas_20261009_r47/'
        row='| validation/shell_rate_convergence_20261009_r48/ | 2项同模型壳T20/80ms、协议/收据/实际覆盖、峰/能量/模式CSV及3图；零新JAX/AD | [r48]('+link+') |\n'
        s=s.replace(marker,row+marker,1)
        s=s.replace('- `binary_n64_20261009_r45_shell_T0p001000/`','- `shell_rate_convergence_20261009_r48_T0p020000/`、`shell_rate_convergence_20261009_r48_T0p080000/`：本轮两项壳INP/ODB；80ms预算范围见报告。\n- `binary_n64_20261009_r45_shell_T0p001000/`',1)
        save(p,s)
    p=W/'output/README.md';s=p.read_text();a=s.index('先看[');z=s.index('\n\n',a)
    s=s[:a]+'先看[r48壳加载时长与峰值](r48_shell_rate_convergence_20261009/README.md)：同模型20/40/80ms、实际覆盖、峰前与峰后质量分开。再看[r47全历史图集](r47_simulation_atlas_20261009/README.md)和[r46复盘](r46_binary_mesh_reassessment_20261009/README.md)。执行只见[唯一规划](../RESEARCH_PLAN.md)，当前范围见[状态](../PROJECT_OVERVIEW.md)，位置见[文件地图](../FILE_MAP.md)。'+s[z:];save(p,s)
    p=R/'validation/README.md';s=p.read_text();a=s.index('更新2026-10-09：');z=s.index('\n\n',a)
    s=s[:a]+'更新2026-10-09：[r48壳T对照](shell_rate_convergence_20261009_r48/README.md)已执行并分析；20ms完整，80ms实际范围与预算停止见报告。新增2个壳作业，零新JAX/AD，原r47图集、r46复盘与r44/r45失败/预算保持。'+s[z:]
    s=s.replace('| [r47历史图集','| [r48壳加载时长](shell_rate_convergence_20261009_r48/README.md) | 同模型20/40/80ms峰稳定性、能量/模式、3图/4CSV；原预算与质量失败保持 |\n| [r47历史图集',1);save(p,s)
    hashes=json.loads((D/'frozen_before.json').read_text());bad=[k for k,v in hashes.items() if hashlib.sha256((R/k).read_bytes()).hexdigest()!=v];assert not bad,bad
    receipt={'updated_current_entries':changed,'protected_count':len(hashes),'all_protected_unchanged':True,'new_JAX_forward':0,'AD':0,'new_shell_jobs':2,'additional_jobs_after_batch':0,'finished_unix':time.time(),'finding':finding}
    for folder in [D,O]:save(folder/'finalization_receipt.json',json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

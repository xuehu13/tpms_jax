"""Maintain existing entry points; no solver or recurring automation."""
from pathlib import Path
import argparse,json,shutil
R=Path('/home/xuehu/projects/tpms_jax');W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
B=R/'validation/equation_coefficients_10pct_20261013_r58';O=W/'output/r58_equation_coefficients_10pct_20261013'
link='output/r58_equation_coefficients_10pct_20261013/README.md（WSL见validation/equation_coefficients_10pct_20261013_r58/README.md）'

def notice(p,n):
    lines=p.read_text().splitlines();where=next((i for i,x in enumerate(lines) if x.startswith('2026-10-13 r58当前')),None)
    if where is None:lines[1:1]=['',n]
    else:lines[where]=n
    p.write_text('\n'.join(lines)+'\n')

def main(start=False):
    windows=['AGENTS.md','RESEARCH_BACKGROUND.md','RESEARCH_PLAN.md','PROJECT_OVERVIEW.md','FILE_MAP.md','START_HERE.md','output/README.md','work/README.md']
    formal=['README.md','AGENTS.md','docs/RESEARCH_BACKGROUND.md','docs/RESEARCH_PLAN.md','docs/RESEARCH_STATUS.md','docs/FILE_MAP.md']
    if start:
        for base,names,archive in [(W,windows,W/'history/execution_equation_coefficients_10pct_20261013_r58/before/windows'),
            (R,formal,R/'docs/history/execution_equation_coefficients_10pct_20261013_r58/before')]:
            archive.mkdir(parents=True,exist_ok=False)
            for name in names:
                dst=archive/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base/name,dst)
        text='2026-10-13 r58当前有限授权：diverse_28三个真实方程系数(c1,c2,c0)扩展到10%，正在执行，尚无新通过结论。复用原0–0.5%终态/切向/检查点，完整反向仍经过原前缀；累计功不清零。原4ms到20%的加载、模型和dt保持，终点2ms。预计新增约6h，上限8h；结果见'+link+'。r57已通过范围及r53失败不改；heartbeat、Abaqus、训练、Git保持暂停。'
        plan='''# 唯一主规划：曲面方程三个系数的0–10%梯度 r58

用户授权：“都算一下压缩到10%的，0–0.5%压缩太小了。”固定测试diverse_28的(c1,c2,c0)=(-3.4,4.6,2)，不扩大其他构型、材料或厚度参数。原r57的0.5%结果和r53失败保持。

1. 核实并冻结r57原模型、参数、实际几何差分终态、c2切向及40个检查点的身份；同一p/h、加载曲线、dt和共享力学。对接点的功/力及切向重新核对，短块对偶/成本通过后续推。
2. 从各自原状态续推到原20%/4ms加载曲线的2ms，即10%。完整路径34485步，原前缀10113步，新增24372步；原dt=5.799774533e−8s，最后一小步事前冻结约5.75e−9s。c2完整JVP和全部12条实际几何FD保留累计功，不做0.5%之后的截断导数。
3. 三系数功FD双档自一致及c2完整功JVP通过后，执行一次完整加载功VJP。反向先过新增检查点，再过原40块，累加全部字段cotangent，只做一次零初态和参数pullback。分别核对三系数FD、c2 JVP及共同缩放零方向；相对1%及原近零门槛不变。末端力仅c2有AD，其余FD，不称三系数力全部通过。
4. 保存完整/部分路径、失败、图表、参数单位和费用，更新现有入口后停止。预计新增约6h，独立总预算8h；不继承r57原3h，也不降终点来换通过。不恢复heartbeat、Abaqus、训练或Git，不自动加FD阶梯、放宽门槛或修改加载/步长。

正式R/validation/equation_coefficients_10pct_20261013_r58；Windows阅读output/r58_equation_coefficients_10pct_20261013。科学输入以input/protocol.json、input/execution.json为准，实际状态看status.json。原0.5%计算费用单列历史复用，新增费用包含原前缀的再次反向，不能重复算作新增前向。

r57已完成：0–0.5%三系数完整功VJP/双档FD通过，c2功/力JVP通过。r53仍有形态及G12瞬时力失败，包含旧a的三参数完整反向未启动。新r58不改这些事实。原长规划与入口字节在history/execution_equation_coefficients_10pct_20261013_r58/before；旧下一步不自动恢复。
'''
        (W/'RESEARCH_PLAN.md').write_text(plan);(R/'docs/RESEARCH_PLAN.md').write_text(plan)
        for base,names in [(W,windows),(R,formal)]:
            for name in names:
                if 'RESEARCH_PLAN' not in name:notice(base/name,text)
        # Remove stale r57 "running" language only in navigation, never scientific reports.
        p=W/'START_HERE.md';s=p.read_text();s=s.replace('正在按[唯一主规划](RESEARCH_PLAN.md)执行；完整梯度尚无结论。','已完成0.5%三系数完整加载功梯度验证。');p.write_text(s)
        p=W/'output/README.md';s=p.read_text().replace('正在执行独立小压缩完整路径验证，尚无完整梯度通过结论；','已通过0.5%三系数完整加载功梯度；');p.write_text(s)
    else:
        status=json.loads((B/'status.json').read_text())
        if status.get('active_jobs'):raise RuntimeError('Do not close navigation while solver active')
        metrics=json.loads((B/'analysis/metrics.json').read_text())
        passed=metrics.get('final_acceptance_passed',False)
        text='2026-10-13 r58当前收口：'+('0–10%三系数完整加载功梯度通过' if passed else '本轮已停止，未取得完整通过结论，失败/部分结果保留')+'。详细证据见'+link+'。无新队列；原r57与r53科学记录不改，heartbeat、Abaqus、训练、Git继续暂停。'
        for base,names in [(W,windows),(R,formal)]:
            for name in names:notice(base/name,text)
    print('Existing navigation updated; no scientific computation launched')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--start',action='store_true');args=ap.parse_args();main(args.start)

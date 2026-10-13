"""One-off, CPU-only result analysis. Never starts FEM, AD, FD or automation."""
from pathlib import Path
import argparse,csv,hashlib,json,os,shutil,time
import numpy as np
R=Path('/home/xuehu/projects/tpms_jax')
B=R/'validation/equation_coefficients_20261012_r57'
W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
O=W/'output/r57_equation_coefficients_20261012'
A=B/'analysis'

def read(p):return json.loads(p.read_text()) if p.exists() else None
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def gate(ad,coarse,fine,scale):
    near=abs(fine)*scale<1e-6
    ae=abs(ad-fine)*scale if near else abs(ad-fine)/abs(fine)
    fe=abs(coarse-fine)*scale if near else abs(coarse-fine)/abs(fine)
    return dict(near_zero=bool(near),AD_FD_error=float(ae),FD_halving_error=float(fe),passed=bool(ae<=(1e-5 if near else .01) and fe<=(1e-5 if near else .01)))

def main(wait_pid=None):
    protocol=read(B/'input/protocol.json')
    if wait_pid:
        # Completion hook for this already-authorized process; no recurring scheduler.
        limit=protocol['started_unix']+protocol['total_wall_budget_s']+900
        while Path(f'/proc/{wait_pid}').exists() and time.time()<limit:time.sleep(10)
        if Path(f'/proc/{wait_pid}').exists():raise RuntimeError('Worker still present beyond analysis wait bound; do not analyze as finished')
    A.mkdir(exist_ok=True)
    status=read(B/'status.json');ex=read(B/'input/execution.json')
    geo=read(B/'checks/geometry.json');bench=read(B/'checks/benchmark.json')
    jvp=read(B/'AD/c2/result.json');vjp=read(B/'AD/work_VJP_result.json')
    jvp_check=read(B/'checks/full_JVP_c2.json')
    fd={p.parent.name:read(p) for p in (B/'FD').glob('*/result.json')}
    p=np.array(protocol['baseline']);scales=np.array(protocol['parameter_scales']);h=scales*protocol['relative_fd_step']
    rows=[];aux=[];c2_duality=None;path_check=None;null=None
    if jvp and ex:
        term=jvp['terminal'];counts=jvp['counts']
        path_check=dict(steps=sum(counts),expected_steps=ex['steps'],terminal_time=term['time_s'],expected_time=ex['end_time_s'],
            terminal_compression=term['compression'],expected_compression=ex['compression'])
        path_check['passed']=sum(counts)==ex['steps'] and abs(term['time_s']-ex['end_time_s'])<1e-12 and abs(term['compression']-ex['compression'])<1e-10
    for i,c in enumerate(protocol['parameters']):
        labels=[c+'_'+n for n in ['plus_h','minus_h','plus_half','minus_half']]
        if not all(n in fd for n in labels):continue
        vals=[fd[n]['terminal'] for n in labels]
        for obj in ['work','force']:
            coarse=(vals[0][obj]-vals[1][obj])/(2*h[i]);fine=(vals[2][obj]-vals[3][obj])/h[i]
            row=dict(coefficient=c,baseline_coefficient=float(p[i]),objective=obj,FD_h=coarse,FD_half=fine,
                h=float(h[i]),units='N mm / coefficient' if obj=='work' else 'N / coefficient',AD=None,AD_method='not computed')
            if obj=='work' and vjp:row.update(AD=vjp['gradient'][i],AD_method='complete VJP')
            if obj=='force' and i==1 and jvp:row.update(AD=jvp['terminal']['gradient'][1],AD_method='complete JVP')
            row.update(gate(row['AD'] if row['AD'] is not None else fine,coarse,fine,scales[i]))
            if row['AD'] is None:row['passed']=None
            if jvp:
                value=jvp['terminal'][obj]
                row['baseline_response']=value
                row['signed_logabs_sensitivity']=float(p[i]*fine/value) if value!=0 else None
            (rows if obj=='work' else aux).append(row)
    if vjp and jvp:
        ad=vjp['gradient'][1];other=jvp['terminal']['gradient'][0]
        c2_duality=dict(VJP=ad,JVP=other,**gate(ad,other,other,scales[1]))
        g=np.array(vjp['gradient']);contraction=float(g@p)
        null=dict(contraction_Nmm=contraction,relative_to_sum_abs_terms=float(abs(contraction)/max(np.sum(np.abs(g*p)),1e-30)),
            note='Common coefficient scaling leaves the zero surface unchanged; diagnostic, no post-hoc tolerance added')
    guard=read(B/'checks/source_guard_before.json')
    changed=[n for n,s in guard.items() if sha(R/n)!=s]
    hashes_changed=[]
    if ex:
        hashes_changed=[n for n,s in ex['source_hashes'].items() if sha(B/'tools'/n)!=s]
    result_complete=bool(vjp and jvp and vjp.get('complete') and jvp.get('complete') and len(rows)==3 and len(fd)==12 and path_check is not None)
    fd_consistency=bool(ex and all(x.get('complete') and x['terminal']['steps']==ex['steps'] and abs(x['terminal']['time_s']-ex['end_time_s'])<1e-12 and abs(x['terminal']['compression']-ex['compression'])<1e-10 for x in fd.values()))
    passed=bool(result_complete and all(x['passed'] for x in rows) and c2_duality['passed'] and path_check['passed'] and fd_consistency and geo['passed'] and bench['short_duality_passed'] and jvp_check and jvp_check['work']['passed'] and not changed and not hashes_changed)
    paths=([jvp] if jvp else [])+list(fd.values())
    metrics=dict(scope='diverse_28 equation coefficients; complete loading-work gradients; terminal force auxiliary c2 only',
        status=status,execution=ex,geometry_passed=geo['passed'] if geo else None,benchmark=bench,work=rows,terminal_force=aux,
        complete_work_result=result_complete,final_acceptance_passed=passed,representative_full_JVP_checks=jvp_check,c2_VJP_JVP=c2_duality,common_scaling=null,path_consistency=path_check,
        completed_FD_paths=len(fd),FD_terminal_consistency=fd_consistency,protected_original_changes=changed,frozen_r57_source_changes=hashes_changed,
        costs=dict(JVP_s=jvp['wall_s'] if jvp else None,FD_s=sum(x['wall_s'] for x in fd.values()),
            VJP_s=vjp['reverse_wall_s'] if vjp else None,total_s=vjp['total_wall_s'] if vjp else None,
            checkpoint_bytes=sum(x.stat().st_size for x in (B/'AD/c2/checkpoints').glob('*.npz'))),
        sampled_min_required_J=min((r['min_required_J'] for x in paths for r in x['path']),default=None),
        failure=read(B/'failure.json'))
    write(A/'metrics.json',metrics)
    columns=['coefficient','baseline_coefficient','objective','AD_method','AD','FD_h','FD_half','h','units','near_zero','AD_FD_error','FD_halving_error','signed_logabs_sensitivity','passed']
    with (A/'gradient_table.csv').open('w',newline='') as f:
        out=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');out.writeheader();out.writerows(rows+aux)
    figs=[]
    if jvp:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
        pp=jvp['path'];x=np.array([v['compression'] for v in pp])*100
        fig,ax=plt.subplots(1,2,figsize=(10,3.5),constrained_layout=True)
        ax[0].plot(x,[v['force'] for v in pp],color='black',label='Baseline')
        ax[1].plot(x,[v['work'] for v in pp],color='black',label='Baseline')
        for sign in ['plus_half','minus_half']:
            q=fd.get('c2_'+sign)
            if q:
                xp=[v['compression']*100 for v in q['path']]
                for a,obj in zip(ax,['force','work']):a.plot(xp,[v[obj] for v in q['path']],ls='--',label='c2 '+sign)
        for a,y in zip(ax,['Compressive force (N)','Loading work (N mm)']):a.set(xlabel='Compression (%)',ylabel=y);a.grid(alpha=.2);a.legend(fontsize=8)
        fig.suptitle('r57 actual equation perturbations: small-compression prefix')
        fig.savefig(A/'response_curves.png');plt.close(fig);figs.append('response_curves.png')
        fig,axes=plt.subplots(2,3,figsize=(12,6),constrained_layout=True)
        for i,c in enumerate(protocol['parameters']):
            for r,obj in enumerate(['work','force']):
                ax=axes[r,i]
                for names,den,label,style in [('h',2*h[i],'FD h','-'),('half',h[i],'FD h/2','--')]:
                    a=fd.get(c+'_plus_'+names);b=fd.get(c+'_minus_'+names)
                    if a and b:
                        xx=[v['compression']*100 for v in a['path']]
                        yy=(np.array([v[obj] for v in a['path']])-np.array([v[obj] for v in b['path']]))/den
                        ax.plot(xx,yy,style,label=label)
                if i==1:ax.plot(x,[v['gradient'][r] for v in pp],':',color='black',label='Full JVP')
                ax.set(title=c,xlabel='Compression (%)',ylabel=('dW/dc (N mm)' if r==0 else 'dF/dc (N)'));ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.suptitle('Trajectory sensitivities: real-geometry finite differences')
        fig.savefig(A/'trajectory_gradients.png');plt.close(fig);figs.append('trajectory_gradients.png')
    if rows and vjp:
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(1,2,figsize=(10,3.7),constrained_layout=True)
        xx=np.arange(3)
        for off,key,label in [(-.23,'AD','Full VJP'),(0,'FD_h','FD h'),(.23,'FD_half','FD h/2')]:
            ax[0].bar(xx+off,[r[key] for r in rows],width=.23,label=label)
        ax[0].set(xticks=xx,xticklabels=[r['coefficient'] for r in rows],ylabel='dW/dc (N mm / coefficient)');ax[0].legend()
        for off,key,label in [(-.16,'AD_FD_error','VJP vs fine FD'),(.16,'FD_halving_error','FD halving')]:
            ax[1].bar(xx+off,[max(r[key]/(1e-5 if r['near_zero'] else .01),1e-10) for r in rows],width=.32,label=label)
        ax[1].axhline(1,color='red',ls='--',label='Applicable gate');ax[1].set(xticks=xx,xticklabels=[r['coefficient'] for r in rows],ylabel='Error / applicable gate',yscale='log');ax[1].legend(fontsize=8)
        fig.suptitle('Complete loading-work gradients; near-zero rows use scaled absolute gate')
        fig.savefig(A/'work_gradient_checks.png');plt.close(fig);figs.append('work_gradient_checks.png')
    def num(v):return '未计算' if v is None else f'{v:.8g}'
    state='完整加载功三系数梯度通过' if passed else ('完整验收未通过' if result_complete else ('计算进行中，尚无完整结论' if status['active_jobs'] else '执行已停止，验证未完成，尚无完整通过结论'))
    text=f'''# r57：真实曲面方程系数梯度验证

**{state}。** 当前阶段：{status['stage']}。本轮只测试diverse_28三个真实方程系数，原r53失败与暂停状态保持，不恢复heartbeat、Abaqus、训练或Git。

## 研究对象与求导范围

方程为 f=c1 cosX cosZ+c2 sinX sinY sinZ+c0=0，X=2πx/L（Y、Z同理）；基线(c1,c2,c0)=(-3.4,4.6,2.0)，均无量纲。c1和c2改变两种周期基函数的权重，c0改变零面的水平位置。厚度仍由到中面的真实三角面距离定义，固定0.5mm；界面宽度固定0.05mm。未把函数值当作真实距离，未把c0当作厚度。

三个原始系数只有两个独立形态自由度：把三个系数同时乘相同非零正数，零面不变。共同缩放是应为零的检查方向；不能把三个有梯度的坐标解释为三个独立结构参数。

原三角中面顶点先严格投影到f=0，最大移动约0.0042715mm；此新基线不同于原近似三角面，但未改变原r55文件。随后沿受边界约束的固定法向求局部非线性根，保持三角连接。隐式顶点导数进入三角面距离，再进入sigmoid占据、材料、质量、XY宏观惯性、移动中面端缘接触插值与面积、初始加速度及全部时间推进。真实FD从每个扰动的非线性几何和零初态重新运行，不以线性化几何代替FD。

加载为最新共同中面端缘等效压板、XY周期、横向宏观伸缩自由；N32 HEX27/27点/HRZ及共享NH材料与内力未改。无完整壁面或自接触。终点按短块成本事前从1%/0.5%选择，使用原20%/4ms加载曲线的前缀，未加快到新的加载率。

## 实际范围与门槛

冻结终点：{num(ex['compression']*100) if ex else '尚未冻结'}%；时间：{num(ex['end_time_s']*1000) if ex else '未定'}ms；步数：{ex['steps'] if ex else '未定'}；固定步长：{num(ex['dt_s']) if ex else '未定'}s。
完整目标是加载功W=∫F dδ，单位N·mm。对三个系数做完整VJP，并分别与±h、±h/2的实际几何全路径FD核对；c2额外做完整JVP。末端力F只对c2做AD核对，c1/c0只有FD。

非近零导数的AD/细FD差与FD减半差均需≤1%；当|细FD|×参数尺度/目标尺度<1e−6时用≤1e−5的缩放绝对误差。参数尺度(3.4,4.6,2)，目标尺度功为1N·mm、力为1N。近零项不能把缩放绝对误差叫相对百分比。

几何字段检查：{geo['passed'] if geo else '未完成'}；短块JVP/VJP对偶相对误差：{num(bench['short_duality_relative']) if bench else '未完成'}。短块通过不能代替完整路径。
完整FD路径已完成{len(fd)}/12；三系数完整功VJP：{'已完成' if vjp else '未完成'}。

## 系数结果

|系数|完整功VJP|功FD(h)|功FD(h/2)|AD/细FD核对量|FD减半核对量|通过|
|---|---:|---:|---:|---:|---:|---|
'''
    for row in rows:text+='|'+ '|'.join([row['coefficient'],num(row['AD']),num(row['FD_h']),num(row['FD_half']),num(row['AD_FD_error']),num(row['FD_halving_error']),str(row['passed'])])+'|\n'
    text+='\n功导数单位N·mm/单位系数。核对量为小数（非近零0.01=1%）；近零标志、辅助力导数、无量纲敏感度及原值详见[完整表](gradient_table.csv)。\n'
    if c2_duality:text+=f"\nc2完整功VJP/JVP核对量{num(c2_duality['AD_FD_error'])}，通过：{c2_duality['passed']}。共同缩放方向∇W·p={num(null['contraction_Nmm'])}N·mm，相对各项绝对值之和为{num(null['relative_to_sum_abs_terms'])}。\n"
    if aux:
        text+='\n末端力辅助结果：\n\n'
        for row in aux:text+=f"- {row['coefficient']}：AD={num(row['AD'])}，细FD={num(row['FD_half'])}N/单位系数；AD核对通过={row['passed']}（None表示未做AD）。\n"
    for f in figs:text+=f'\n![{f}]({f})\n'
    text+=f'''
## 成本、失败与证据

完整JVP {num(metrics['costs']['JVP_s']/60) if jvp else '未完成'}min，12条FD累计{metrics['costs']['FD_s']/60:.2f}min，完整VJP {num(metrics['costs']['VJP_s']/60) if vjp else '未完成'}min；总墙钟{num(metrics['costs']['total_s']/60) if vjp else '尚无最终值'}min。总预算180min，从首次协议开始计，包括保存计划错误的时间。检查点占用{metrics['costs']['checkpoint_bytes']/1024**3:.3f}GiB；反向复用代表JVP的原始正向检查点，未另算基线。

首次在保存execution计划时出现字典自引用，发生于完整路径启动之前；原源码、日志、几何和短块结果保留在history/serialization_stop_before_full_paths。只修复复制字典并继续原预算，模型、FD步长和门槛未改。当前失败文件：{metrics['failure']['message'] if metrics['failure'] else '无新的执行失败'}。

原科学文件保护检查：{len(guard)}项，变更{len(changed)}项；冻结r57执行源码变更{len(hashes_changed)}项。完整检查点步数/终态一致：{path_check['passed'] if path_check else '尚未完成'}。数值采样中的最小必要域J={num(metrics['sampled_min_required_J'])}，有限采样不等于全过程证明。

## 能得出什么，不能得出什么

若完整验收通过，证据支持本构型基线附近、此次小压缩固定推进路径下，对真实曲面方程系数的完整加载功设计梯度可正确回传。局部几何延拓只在固定三角连接、相同根分支、稳定最近端缘分区和背景面单元内成立；最近三角面切换仍会限制导数光滑性。小扰动不代表任意大幅系数变化或拓扑变化可导。

AD/FD核对验证的是同一离散动态模型的导数，不能单独证明它等于Abaqus或真实结构的敏感度，也不能推广到20%、屈曲跳跃、自接触或所有瞬时力目标。固定路径导数不含接受/拒绝、减步或步数取整决策的导数。负或正梯度表示增加该原始有符号系数时目标的局部增减，不能自动解释为增加该系数绝对值的效果。

正式源码、输入、大数组、原路径和检查点：WSL /home/xuehu/projects/tpms_jax/validation/equation_coefficients_20261012_r57。Windows为阅读副本；协议与执行输入在../input，详细数值在metrics.json。旧r53门槛和失败未改，不自动恢复其他计算。
'''
    (A/'README.md').write_text(text)
    # Root entry points target analysis; relative image/table links must remain in analysis.
    root='# r57：真实曲面方程三个系数\n\n'+f'**{state}。**\n\n'+ '[完整方法、结果与图](analysis/README.md) · [数值表](analysis/gradient_table.csv) · [实际状态](status.json)\n\n正式执行仅WSL；本目录为阅读副本。原r53与heartbeat、Abaqus、训练、Git继续暂停。\n'
    (B/'README.md').write_text(root)
    O.mkdir(exist_ok=True)
    for rel in ['status.json','README.md','failure.json','input/model.json','input/protocol.json','input/execution.json','checks/geometry.json','checks/benchmark.json','checks/full_JVP_c2.json','AD/work_VJP_result.json']:
        src=B/rel
        if src.exists():dst=O/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    for src in A.iterdir():
        if src.is_file() and src.suffix in ['.json','.csv','.png','.md']:
            dst=O/'analysis'/src.name;dst.parent.mkdir(exist_ok=True);shutil.copy2(src,dst)
    write(O/'source_guard_recheck.json',dict(protected_files=len(guard),changed=changed))
    if not status['active_jobs']:
        write(A/'worker_status_before_analysis.json',status)
        status.update(analysis='analysis/README.md',final_acceptance_passed=passed)
        if result_complete:status['stage']='completed' if passed else 'completed_not_passed'
        write(B/'status.json',status);write(O/'status.json',status)
        notice=f'2026-10-12新增授权r57已收口：{state}；真实diverse_28(c1,c2,c0)，终点{num(ex["compression"]*100) if ex else "未冻结"}%。详细证据见output/r57_equation_coefficients_20261012/README.md（WSL见validation/equation_coefficients_20261012_r57/README.md）。无新队列；原r53失败与暂停不改，heartbeat、Abaqus、训练、Git仍暂停。'
        for directory,names in [(W,['AGENTS.md','RESEARCH_BACKGROUND.md','PROJECT_OVERVIEW.md','FILE_MAP.md']),
            (R,['README.md','AGENTS.md','docs/RESEARCH_BACKGROUND.md','docs/RESEARCH_STATUS.md','docs/FILE_MAP.md'])]:
            for name in names:
                target=directory/name;lines=target.read_text().splitlines()
                index=next((i for i,line in enumerate(lines) if line.startswith('2026-10-12新增授权r57') or line.startswith('2026-10-12用户新增授权r57')),None)
                if index is not None:lines[index]=notice;target.write_text('\n'.join(lines)+'\n')
        for directory,name in [(W,'RESEARCH_PLAN.md'),(R,'docs/RESEARCH_PLAN.md')]:
            target=directory/name;s=target.read_text();first=s.find('\n')+1
            lines=s.splitlines();index=next((i for i,line in enumerate(lines) if line.startswith('2026-10-12新增授权r57已收口')),None)
            if index is None:target.write_text(s[:first]+'\n'+notice+'\n\n以下四步为本轮原定安排，实际完成范围和失败以r57报告为准；不自动追加计算。\n'+s[first:])
            else:lines[index]=notice;target.write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(stage=status['stage'],complete=result_complete,passed=passed,FD_paths=len(fd),report=str(O/'analysis/README.md'))))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--wait-pid',type=int);args=ap.parse_args();main(args.wait_pid)

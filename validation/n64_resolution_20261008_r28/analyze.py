"""Extract one completed/partial r28 path; no mechanics or design-AD solve."""
from pathlib import Path
import hashlib,json,shutil,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/xuehu/projects/tpms_jax');D=R/'validation/n64_resolution_20261008_r28'
W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
out=D/'full_from_zero';protocol=json.loads((D/'protocol.json').read_text());T=.002
def write(p,value):p.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
shell=json.loads((D/'abaqus/explicit_T0p002/results/shell.json').read_text())
rows=json.loads((out/'accepted_path.json').read_text())
assert len(rows)>1, 'No nonzero accepted trajectory available'
old=json.loads((R/'validation/step_control_20261007_r18/controlled_forward/accepted_path.json').read_text())
fastold=json.loads((R/'validation/shell_rate_20261007_r20/abaqus/explicit_T0p004/results/shell.json').read_text())
jload=[r for r in rows if r['time']<=T+1e-12]
sload=[r for r in shell['force_path'] if r['time']<=T+1e-12]
def curve(path):return np.array([r['compression'] for r in path]),-np.array([r['Fz_N'] for r in path])
jx,jf=curve(jload);sx,sf=curve(sload)
amax=min(float(jx.max()),float(sx.max()))
complete=(out/'result.json').exists() and rows[-1]['time']>=1.1*T*(1-1e-8)
record=json.loads((out/('result.json' if complete else 'failure.json')).read_text()) if (out/('result.json' if complete else 'failure.json')).exists() else {}
matched=float(protocol['denominators_N']['matched_shell_peak']);legacy=float(protocol['denominators_N']['historical_fixed'])
def observed_max(path):return max(path,key=lambda r:-r['Fz_N'])
sjpeak=observed_max([r for r in sload if r['compression']>=.01])
jpeak=observed_max([r for r in jload if r['compression']>=.01]) if amax>=.01 else None
metrics={'scope':'same .002s dynamic load; signed compressive force retained, original strain alignment',
 'covered_compression':[0.,amax],'full_20_and_hold_complete':complete}
if amax>=.01:
    grid=np.linspace(.01,amax,1001);delta=np.interp(grid,jx,jf)-np.interp(grid,sx,sf)
    rms=float(np.sqrt(np.mean(delta**2)))
    metrics.update(RMS_N=rms,RMS_over_matched_peak=rms/matched,RMS_over_historical_fixed_peak=rms/legacy)
grid=np.linspace(0.,amax,1001)
jwork=float(10*np.trapezoid(np.interp(grid,jx,jf),grid));swork=float(10*np.trapezoid(np.interp(grid,sx,sf),grid))
metrics.update(JAX_work_N_mm=jwork,shell_work_N_mm=swork,
    work_relative_gap=abs(jwork/swork-1) if abs(swork)>1e-20 else None)
jt=np.array([r['time'] for r in rows]);st=np.array([r['time'] for r in shell['force_path']])
force_all=-np.array([r['Fz_N'] for r in rows]);sforce_all=-np.array([r['Fz_N'] for r in shell['force_path']])
if complete:
    hg=np.linspace(1.05*T,1.1*T,1001)
    hmj=float(np.trapezoid(np.interp(hg,jt,force_all),hg)/(hg[-1]-hg[0]))
    hms=float(np.trapezoid(np.interp(hg,st,sforce_all),hg)/(hg[-1]-hg[0]))
    metrics.update(hold_last_half_JAX_mean_N=hmj,hold_last_half_shell_mean_N=hms,
        hold_mean_relative_gap=abs(hmj/hms-1),peak_strain_absolute_gap=abs(jpeak['compression']-sjpeak['compression']),
        peak_force_relative_gap=abs((-jpeak['Fz_N'])/matched-1))
numeric={'accepted_endpoints_finite':all(r['J_finite'] and np.isfinite([r['Fz_N'],r['energy_N_mm'],r['KE_N_mm']]).all() for r in rows),
 'invalid_uncontinued_NH_points_max':max(r['invalid_material_points'] for r in rows),
 'required_positive_J_min':min(r['required_positive_J_min'] for r in rows),
 'actual_all_J_min':min(r['J_min'] for r in rows),
 'negative_all_J_points_max':max(r['negative_J_points'] for r in rows),
 'min_accepted_dt_seconds':min(r.get('dt_seconds',float('inf')) for r in rows),
 'rejected_blocks':len(record.get('rejected_blocks',[])),
 'scope':'accepted endpoint samples and recorded guards only; not all internal positions or contact certified'}
work_path=np.zeros(len(rows))
for i in range(1,len(rows)):
    work_path[i]=work_path[i-1]+5*(force_all[i]+force_all[i-1])*(rows[i]['compression']-rows[i-1]['compression'])
total=np.array([r['energy_N_mm']+r['KE_N_mm'] for r in rows]);gap=total-total[0]-work_path
ref=max(1e-30,float(np.max(np.abs(work_path))))
energy={'terminal_total_minus_external_work_N_mm':float(gap[-1]),'max_balance_gap_over_max_work':float(np.max(np.abs(gap))/ref),
 'terminal_KE_over_U':rows[-1]['KE_over_U'],'max_KE_over_U_after_1pct':max((r['KE_over_U'] for r in rows if r['compression']>=.01),default=None),
 'original_1pct_energy_gate_pass':bool(np.max(np.abs(gap))/ref<=.01),
 'scope':'trapezoidal external work from accepted macro-reaction/compression samples; high KE is recorded, not automatically removed'}
gates=None
if complete:
    gates={'peak_strain_le_1_percentage_point':metrics['peak_strain_absolute_gap']<=.01,
     'peak_force_le_10pct':metrics['peak_force_relative_gap']<=.1,
     'curve_RMS_over_matched_peak_le_10pct':metrics['RMS_over_matched_peak']<=.1,
     'work_le_10pct':metrics['work_relative_gap']<=.1,'hold_last_half_mean_le_10pct':metrics['hold_mean_relative_gap']<=.1}
frozen=json.loads((D/'frozen_before.json').read_text())
integrity={name:hashlib.sha256((R/name).read_bytes()).hexdigest()==digest for name,digest in frozen.items()}
result={'round':'r28','N':64,'load_time_seconds':T,'hold_time_seconds':.1*T,
 'full_complete':complete,'last_accepted':rows[-1],
 'stop_reason':None if complete else record.get('message','No complete receipt; partial coverage only'),
 'observed_JAX_max_in_covered_loading':jpeak,'matched_shell_loading_peak':sjpeak,
 'matched_vs_old_shell_peak_shift_percentage_points':100*(sjpeak['compression']-max((r for r in fastold['force_path'] if r['time']<=.004 and r['compression']>=.01),key=lambda r:-r['Fz_N'])['compression']),
 'metrics':metrics,'numerical_samples':numeric,'energy_diagnostics':energy,'working_accuracy_gates':gates,
 'matched_shell_original_quality_checks':shell['checks'],'frozen_unchanged':integrity,
 'production_source_unchanged':all(integrity.values()),'new_design_AD':False,
 'not_pure_resolution_cause_isolation':True,'no_universal_replacement_or20pct_gradient_claim':True}
write(D/'comparison.json',result)
fig,axes=plt.subplots(2,2,figsize=(12,8.3));axes=axes.ravel()
ox,of=curve(old);fx,ff=curve([r for r in fastold['force_path'] if r['time']<=.004])
axes[0].plot(100*fx,ff,'--',color='0.65',lw=1,label='Shell T=.004 (history)')
axes[0].plot(100*ox,of,':',color='0.4',lw=1.2,label='N32 T=.004, partial')
axes[0].plot(100*sx,sf,color='#253E67',lw=1.6,label='Matched shell T=.002')
axes[0].plot(100*jx,jf,color='#B15A23',lw=1.6,label='N64 T=.002'+('' if complete else ', partial'))
axes[0].set(xlabel='Compression (%)',ylabel='Compressive macro force (N)',title='Raw response; no peak shift or smoothing')
axes[0].legend(fontsize=8)
axes[1].plot(jt*1000,total,label='U+KE');axes[1].plot(jt*1000,work_path,label='External work')
axes[1].plot(jt*1000,gap,label='Balance gap');axes[1].set(xlabel='Time (ms)',ylabel='Energy / work (N mm)',title='Energy balance at accepted endpoints');axes[1].legend(fontsize=8)
axes[2].plot(100*np.array([r['compression'] for r in rows]),[r.get('dt_seconds',np.nan)*1e9 for r in rows])
axes[2].set(xlabel='Compression (%)',ylabel='Accepted dt (ns)',title='State-bound step control retained')
axes[3].plot(jt*1000,[r['KE_over_U'] for r in rows]);axes[3].set_yscale('symlog',linthresh=.01);axes[3].set(xlabel='Time (ms)',ylabel='KE / U',title='Dynamics recorded; near-zero initial U is delicate')
for ax in axes:ax.grid(alpha=.2)
fig.suptitle('diverse_04, t=.5 mm, HEX27 / 27 points; rate and resolution changed',fontsize=13)
fig.tight_layout();fig.savefig(D/'response.png',dpi=170);plt.close(fig)
completion='完成20%和原比例保载' if complete else f"只完成到{100*rows[-1]['compression']:.4f}%压缩，未完成20%/保载"
gate_text='未计算完整验收；仅报告实际共同窗口' if gates is None else ('全部预定响应工作目标通过' if all(gates.values()) else '部分预定响应工作目标未通过，详见comparison.json')
report=f'''# r28结果：{completion}

本轮使用N64、HEX27/27点、同一0.5mm壁厚/材料/界面/HRZ与XYZ周期波动，加载从0.004s缩短到0.002s、保载0.0002s。用户固定总计算预算3小时；一次匹配壳参照和一次从零JAX尝试，不自动延长。{gate_text}。

实际覆盖到压缩{100*rows[-1]['compression']:.6f}%、时间{rows[-1]['time']:.9g}s。停止原因：{result['stop_reason'] or '完整时程结束'}。同速率壳观察峰{100*sjpeak['compression']:.4f}%、{-sjpeak['Fz_N']:.6f}N。JAX覆盖窗口内最大反力所在位置与数值见comparison.json；未完整时不把它认证为完整响应峰。

相对误差、实际J、dt、拒绝块、外功/总能量和高KE记录见comparison.json及response.png。壳原质量关口保持，不冒称真值。历史N32和0.004s壳仅作带明确时间标签的辅助曲线；本轮既改变空间分辨率又改变时程，不能只凭改善归因网格，不能将未完成当精度失败，也不认证设计梯度或任意构型替代。

正式位置：/home/xuehu/projects/tpms_jax/validation/n64_resolution_20261008_r28。完整/部分接受路径与检查点在full_from_zero；新壳原生INP/ODB在E:/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus/n64_resolution_20261008_r28_shell_T0p002，轻量提取在abaqus/explicit_T0p002/results。protocol.json为预定规则，before_rate_revision保留缩短加载前协议，resource_result.json保留0.004s的初始成本测量。

六生产/环境文件、安装库、旧关键证据及原壳数据哈希保持：{all(integrity.values())}。无训练、设计AD、新材料/阻尼/质量缩放/接触或GitHub提交。本次结果先用于判断能否改善复杂构型前向响应，后续只根据证据选择一项，不自动批量继续。
'''
(D/'REVIEW.md').write_text(report,encoding='utf-8')
read=W/'output/r28_n64_resolution_20261008';(read/'README.md').write_text(report,encoding='utf-8')
for name in ('comparison.json','response.png','protocol.json'):shutil.copy2(D/name,read/name)
for name,target in [('PROJECT_OVERVIEW.md',R/'docs/RESEARCH_STATUS.md'),('RESEARCH_PLAN.md',R/'docs/RESEARCH_PLAN.md'),('AGENTS.md',R/'AGENTS.md')]:
    text=(W/name).read_text(encoding='utf-8')
    note=f'> r28自动结果记录（2026-10-08）：{completion}；{gate_text}。原始比较及停止原因见output/r28_n64_resolution_20261008/README.md；禁止据此自动扩大预算/批量开展。\n\n'
    if text.startswith('#'):
        first,rest=text.split('\n',1);text=first+'\n\n'+note+rest.lstrip('\n')
    (W/name).write_text(text,encoding='utf-8');target.write_text(text,encoding='utf-8')
print(json.dumps({'full_complete':complete,'last_compression':rows[-1]['compression'],'working_gates':gates,'report':str(D/'REVIEW.md')}),flush=True)

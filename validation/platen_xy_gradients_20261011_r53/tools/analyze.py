"""Compare the predeclared two-level FD with full-path thickness AD."""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1];R=B.parents[1]

def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def compare(name,value,ad,fd_h,fd_half,limits,index=0):
    scale=limits.get('parameter_scale',limits.get('parameter_scale_mm'))/limits['objective_scales'][index]
    nearzero=abs(fd_half)*scale<limits['nearzero_dimensionless_gradient_cutoff']
    err=abs(ad-fd_half);fd_err=abs(fd_h-fd_half)
    rel=err/max(abs(fd_half),1e-300);fd_rel=fd_err/max(abs(fd_half),1e-300)
    passed=(err*scale<=limits['nearzero_absolute_scaled_error'] and fd_err*scale<=limits['nearzero_absolute_scaled_error']) if nearzero else (rel<=limits['nonzero_gradient_relative_AD_FD'] and fd_rel<=limits['FD_h_halving_relative'])
    return dict(objective=name,baseline_value=float(value),AD=float(ad),FD_h=float(fd_h),FD_half=float(fd_half),
                AD_FD_relative_error=float(rel),FD_halving_relative_difference=float(fd_rel),nearzero=bool(nearzero),pass_target=bool(passed))

def main():
    cfg=json.loads((B/'input/protocol.json').read_text())
    names=['AD/thickness','FD/t_plus_h','FD/t_minus_h','FD/t_plus_half','FD/t_minus_half']
    cases=[json.loads((B/n/'result.json').read_text()) for n in names]
    assert all(x['complete'] and x['steps']==cases[0]['steps'] for x in cases)
    for name in names:
        with np.load(B/name/'final_state.npz') as z:
            assert all(np.isfinite(z[k]).all() for k in z.files),f'Nonfinite final state: {name}'
    with np.load(B/'AD/thickness/final_tangent.npz') as z:
        assert all(np.isfinite(z[k]).all() for k in z.files),'Nonfinite final design tangent'
    paths=[np.array([[row[k] for k in ['compression','force_N','work_N_mm']] for row in x['path']]) for x in cases]
    for p in paths[1:]:np.testing.assert_array_equal(p[:,0],paths[0][:,0])
    ad=np.array([cases[0]['terminal']['work_gradient_N'],cases[0]['terminal']['force_gradient_N_per_mm']])
    values=np.array([[x['terminal']['work_N_mm'],x['terminal']['force_N']] for x in cases])
    fd_h=(values[1]-values[2])/.005;fd_half=(values[3]-values[4])/.0025
    rows=[];limits=cfg['thresholds']
    for i,name in enumerate(['loading_work','terminal_upper_force']):
        rows.append(compare(name,values[0,i],ad[i],fd_h[i],fd_half[i],limits,i))
    old=R/'validation/platen_xy_G_20261010_r51/experiments/midsurface_rim_20pct_20261010/jax/result.json'
    op=json.loads(old.read_text())['path'];ox=np.array([r['compression'] for r in op]);of=np.array([r['force_N'] for r in op])
    delta=paths[0][:,1]-np.interp(paths[0][:,0],ox,of)
    metrics=dict(scope='G complete0-5pct thickness total derivative, fixed time discretization',
                 full_path_design_JVP=True,reverse_VJP_validated=False,shape_gradient_validated=False,
                 final_states_and_full_tangent_finite=True,
                 step1_pass=all(r['pass_target'] for r in rows),objectives=rows,
                 baseline_vs_original_G_fixed_dt_change=dict(max_absolute_force_difference_N=float(np.max(abs(delta))),
                     RMS_force_difference_N=float(np.sqrt(np.mean(delta**2))),interpretation='Also includes interpolation of original64-step samples; not a new spatial convergence claim'),
                 elapsed_total_minutes=(max((B/n/'result.json').stat().st_mtime for n in names)-cfg['batch_started_unix'])/60,
                 case_wall_minutes={n:x['wall_s']/60 for n,x in zip(names,cases)},
                 sampled_min_required_J=min(x['sampled_min_required_J'] for x in cases),
                 all_virtual_J_and_energy_retained=True)
    out=B/'analysis';write(out/'metrics.json',metrics)
    with (out/'gradient_comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    for n,p in zip(names,paths):axes[0].plot(p[:,0]*100,p[:,1],label=n,lw=1)
    axes[0].set(xlabel='Compression (%)',ylabel='Upper plate force (N)',title='Independent paths: thickness perturbations')
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    gradcurve=np.array([r['force_gradient_N_per_mm'] for r in cases[0]['path']])
    axes[1].plot(paths[0][:,0]*100,gradcurve,label='Full-path JVP',lw=1.6)
    axes[1].plot(paths[0][:,0]*100,(paths[1][:,1]-paths[2][:,1])/.005,label='FD h=0.0025 mm',ls='--')
    axes[1].plot(paths[0][:,0]*100,(paths[3][:,1]-paths[4][:,1])/.0025,label='FD h=0.00125 mm',ls=':')
    axes[1].set(xlabel='Compression (%)',ylabel='dF / d(thickness) (N/mm)',title='Thickness sensitivity along the complete path')
    axes[1].legend(fontsize=8);axes[1].grid(alpha=.2);fig.tight_layout();fig.savefig(out/'thickness_gradient.png');plt.close(fig)
    guard=json.loads((B/'checks/source_guard_before.json').read_text())
    assert all(sha(R/p)==h for p,h in guard.items()),'Protected source changed'
    write(B/'checks/source_guard_after.json',dict(unchanged=True,files=guard))
    lines=['# G5%厚度完整路径梯度结果','',f"第1步{'通过' if metrics['step1_pass'] else '未通过'}预先约定的1%导数核对目标。尚未验证反向回传或构型梯度。",'',
           '| 目标 | 基线值 | JVP | FD h | FD h/2 | AD/FD差 | 两档FD差 |','|---|---:|---:|---:|---:|---:|---:|']
    for x in rows:lines.append(f"| {x['objective']} | {x['baseline_value']:.9g} | {x['AD']:.9g} | {x['FD_h']:.9g} | {x['FD_half']:.9g} | {x['AD_FD_relative_error']*100:.5g}% | {x['FD_halving_relative_difference']*100:.5g}% |")
    lines+=['','![厚度扰动与梯度核对](thickness_gradient.png)','',
            '加载功单位N·mm，其厚度导数单位N；总力单位N，其厚度导数单位N/mm。FD是独立完整路径核对，JVP是自动微分；所有路径固定相同步长和加载前缀，包含占据、质量、宏观投影与端部面积的变化。',
            '',f"总成本约{metrics['elapsed_total_minutes']:.2f}min；数值守卫采样中主要材料实际J最小{metrics['sampled_min_required_J']:.6g}。虚域负J与全部材料能量保留。有限采样不等于每时每点的严格证明。",'',
            '这里求的是既定动态加载路径的敏感度，包含惯性与接触，不等于静力平衡导数。少量参数AD不保证比差分更快，当前先验证正确性；反向回传的多参数优势还须实际验证。',
            '', '本结果证明限定JAX离散模型的求导实现，不能替代物理设计敏感度或其他构型验证。原20%导数符号失败不变。']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    write(B/'status.json',dict(stage='step1_analyzed',active_jobs=[],step1='passed' if metrics['step1_pass'] else 'not_passed',step2='pending' if metrics['step1_pass'] else 'gated_not_started',step3='pending',step4='pending',new_reverse_AD=False))
    print(json.dumps(metrics,indent=2))

def interface():
    spec=json.loads((B/'input/interface_protocol.json').read_text())
    reverse=json.loads((B/'AD/reverse_work/result.json').read_text())
    assert reverse['complete'] and reverse['passed'],'Full reverse not passed'
    cases=[json.loads((B/c['label']/'result.json').read_text()) for c in spec['cases']]
    assert all(c['params_mm']==s['params_mm'] for c,s in zip(cases,spec['cases'])),'FD parameter mismatch'
    base=json.loads((B/'AD/thickness/result.json').read_text())
    assert all(c['complete'] and c['steps']==base['steps'] and c['dt_s']==base['dt_s'] for c in cases)
    paths=[np.array([[r[k] for k in ['compression','force_N','work_N_mm']] for r in c['path']]) for c in [base,*cases]]
    for p in paths[1:]:np.testing.assert_array_equal(p[:,0],paths[0][:,0])
    for case in spec['cases']:
        with np.load(B/case['label']/'final_state.npz') as z:
            assert all(np.isfinite(z[k]).all() for k in z.files),'Nonfinite interface FD state'
    h,half=spec['fd_steps_mm'];values=np.array([[c['terminal']['work_N_mm'],c['terminal']['force_N']] for c in cases])
    fd_h=(values[0]-values[1])/(2*h);fd_half=(values[2]-values[3])/(2*half)
    row=compare('loading_work',base['terminal']['work_N_mm'],reverse['gradient_N'][1],fd_h[0],fd_half[0],spec['thresholds'])
    guard=json.loads((B/'checks/source_guard_before.json').read_text())
    assert all(sha(R/p)==v for p,v in guard.items()),'Protected source changed'
    metrics=dict(scope='G complete0-5pct interface-width reverse work derivative',step2_interface_pass=row['pass_target'],
                 objective=row,reverse_thickness_pass=reverse['passed'],terminal_force_AD_validated=False,
                 terminal_force_FD_N_per_mm=dict(h=float(fd_h[1]),half=float(fd_half[1])),
                 interface_is_numerical_geometry_parameter=True,sampled_min_required_J=min(c['sampled_min_required_J'] for c in cases),
                 beta_per_mm=float(2*np.log(9)/.05),work_beta_derivative_N_mm2=float(-.05**2/(2*np.log(9))*row['AD']),
                 case_wall_minutes={c['case']:c['wall_s']/60 for c in cases},protected_sources_unchanged=True)
    out=B/'analysis/interface5';out.mkdir(exist_ok=True);write(out/'metrics.json',metrics)
    with (out/'gradient_comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerow(row)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    for label,p in zip(['Baseline',*[c['label'] for c in spec['cases']]],paths):axes[0].plot(p[:,0]*100,p[:,1],label=label,lw=1)
    axes[0].set(xlabel='Compression (%)',ylabel='Upper plate force (N)',title='Interface-width perturbations: complete paths')
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    bars=axes[1].bar(['Reverse VJP','FD h','FD h/2'],[row['AD'],row['FD_h'],row['FD_half']],color=['#245780','#628a9d','#94afba'])
    axes[1].bar_label(bars,fmt='%.7g',padding=3,fontsize=9)
    axes[1].set(ylabel='dW / d(interface width) (N)',title='Loading-work width sensitivity at5% compression')
    axes[1].margins(y=.2);axes[1].grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(out/'interface_gradient.png');plt.close(fig)
    text=['# G5%界面宽度完整反向导数','',f"界面参数核对{'通过' if row['pass_target'] else '未通过'}预定门槛；该参数控制数值过渡，不是实体壁厚。",'',
          '| 目标 | 反向VJP | FD h | FD h/2 | VJP/FD差 | 两档FD差 |','|---|---:|---:|---:|---:|---:|',
          f"| 加载功对w导数（N） | {row['AD']:.10g} | {row['FD_h']:.10g} | {row['FD_half']:.10g} | {row['AD_FD_relative_error']*100:.6g}% | {row['FD_halving_relative_difference']*100:.6g}% |",'',
          '![界面宽度核对](interface_gradient.png)','',
          'w为0.05mm的10–90%界面宽；厚度0.5mm、中面、材料、N32、固定时间步与原加载前缀保持。w改变Gauss占据、质量/宏观惯性和端缘面积权重；以上变化均进入完整路径反向链。四条FD均从零独立计算。',
          '',f"本定义β=2ln9/w={metrics['beta_per_mm']:.8g}mm⁻¹，因此dW/dβ=−w²/(2ln9)·dW/dw={metrics['work_beta_derivative_N_mm2']:.8g}N·mm²；不能与历史无量纲β混同。",'',
          '此次反向目标是加载功。末端力的宽度导数仅有差分记录，不称其AD已验证。厚度的功/力JVP与完整功VJP另见上级报告；10%/12%、P和构型系数仍需依规划推进。',
          '', '通过说明既定JAX离散动态模型的局部导数实现正确；不自动证明Abaqus或实物具有相同敏感度，不靠调整w拟合壳曲线。']
    (out/'README.md').write_text('\n'.join(text)+'\n')
    write(B/'checks/source_guard_after_interface.json',dict(unchanged=True,files=guard))
    write(B/'status.json',dict(stage='step2_interface_analyzed',active_jobs=[],step1='passed',step2='passed' if row['pass_target'] else 'not_passed',
                             step2_reverse='passed',step2_interface='passed' if row['pass_target'] else 'not_passed',
                             step3='pending' if row['pass_target'] else 'gated_not_started',step4='pending' if row['pass_target'] else 'gated_not_started',new_reverse_AD=True))
    print(json.dumps(metrics,indent=2))

def extension(spec_path):
    spec=json.loads((B/spec_path).read_text());names=[c['label'] for c in spec['cases']]
    cases=[json.loads((B/n/'result.json').read_text()) for n in names]
    assert len(cases)==5 and spec['cases'][0]['is_ad'] and not any(c['is_ad'] for c in spec['cases'][1:])
    for result,declared in zip(cases,spec['cases']):
        assert result['complete'] and result['steps']==spec['steps'] and result['dt_s']==spec['dt_s']
        assert result.get('parameters',result.get('params_mm'))==declared.get('parameters',declared.get('params_mm')) and result['seed']==declared.get('seed')
        with np.load(B/result['case']/'final_state.npz') as z:assert all(np.isfinite(z[k]).all() for k in z.files)
        if declared.get('initial_from'):
            restart=result['restart'];assert restart['source']==declared['initial_from']
            assert all(sha(B/restart['source']/p)==h for p,h in restart['hashes'].items()),'Saved prefix changed'
            prefix=json.loads((B/restart['source']/'result.json').read_text())
            assert result['path'][:len(prefix['path'])]==prefix['path'],'Lost full saved prefix'
    with np.load(B/names[0]/'final_tangent.npz') as z:assert all(np.isfinite(z[k]).all() for k in z.files)
    paths=[np.array([[r[k] for k in ['compression','force_N','work_N_mm']] for r in c['path']]) for c in cases]
    for path in paths[1:]:np.testing.assert_array_equal(path[:,0],paths[0][:,0])
    assert 0<paths[0][-1,0]<=spec['target_compression']+1e-10
    values=np.array([[c['terminal']['work_N_mm'],c['terminal']['force_N']] for c in cases])
    gradient_keys=['work_gradient_value','force_gradient_value'] if 'parameters' in spec['cases'][0] else ['work_gradient_N','force_gradient_N_per_mm']
    ad=np.array([cases[0]['terminal'][k] for k in gradient_keys])
    h,half=spec.get('fd_steps',spec.get('fd_steps_mm'));fd_h=(values[1]-values[2])/(2*h);fd_half=(values[3]-values[4])/(2*half)
    rows=[compare(name,values[0,i],ad[i],fd_h[i],fd_half[i],spec['thresholds'],i)
          for i,name in enumerate(['loading_work','terminal_upper_force'])]
    if 'gradient_units' in spec:
        for row,unit in zip(rows,spec['gradient_units']):row['gradient_unit']=unit
    guard=json.loads((B/'checks/source_guard_before.json').read_text())
    assert all(sha(R/p)==v for p,v in guard.items()),'Protected science changed'
    assert all(sha(R/p)==v for p,v in spec.get('protected_sources',{}).items()),'Protected migration source changed'
    required=spec.get('required_objectives',['loading_work','terminal_upper_force'])
    passed=all(row['pass_target'] for row in rows if row['objective'] in required)
    metrics=dict(scope=spec['scope'],passed=passed,actual_compression=float(paths[0][-1,0]),
                 complete_from_zero_design_JVP=True,full_prefix_state_tangent_reused=any(c.get('initial_from') for c in spec['cases']),
                 objectives=rows,steps=spec['steps'],dt_s=spec['dt_s'],
                 additional_case_wall_minutes={n:c['wall_s']/60 for n,c in zip(names,cases)},
                 sampled_min_required_J=min(c['sampled_min_required_J'] for c in cases),
                 sampled_min_all_J=min(c['sampled_min_actual_J'] for c in cases),
                 protected_sources_unchanged=True,reverse_at_this_endpoint_validated=False,required_objectives=required,
                 parameter=spec['active_parameter'])
    if spec.get('baseline_original_result'):
        original=json.loads((R/spec['baseline_original_result']).read_text())['path']
        ox=np.array([r['compression'] for r in original]);of=np.array([r['force_N'] for r in original])
        delta=paths[0][:,1]-np.interp(paths[0][:,0],ox,of)
        metrics['baseline_vs_original_forward']=dict(max_absolute_force_N=float(np.max(abs(delta))),RMS_force_N=float(np.sqrt(np.mean(delta**2))),includes_sample_interpolation=True)
    out=B/'analysis'/spec['analysis_name'];out.mkdir(exist_ok=True);write(out/'metrics.json',metrics)
    with (out/'gradient_comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    for name,path in zip(names,paths):axes[0].plot(path[:,0]*100,path[:,1],label=name.split('/')[-1],lw=1)
    axes[0].set(xlabel='Compression (%)',ylabel='Upper plate force (N)',title=spec['plot_title']+': independent paths')
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    unit=spec.get('active_parameter_unit','mm');force_unit=spec.get('gradient_units',['N','N/mm'])[1]
    axes[1].plot(paths[0][:,0]*100,[r[gradient_keys[1]] for r in cases[0]['path']],label='Complete-path JVP',lw=1.4)
    axes[1].plot(paths[0][:,0]*100,(paths[1][:,1]-paths[2][:,1])/(2*h),label=f'FD h={h:g} {unit}',ls='--')
    axes[1].plot(paths[0][:,0]*100,(paths[3][:,1]-paths[4][:,1])/(2*half),label=f'FD h={half:g} {unit}',ls=':')
    axes[1].set(xlabel='Compression (%)',ylabel=f"dF / d({spec.get('parameter_label','thickness')}) ({force_unit})",title='Full-path parameter sensitivity')
    axes[1].legend(fontsize=8);axes[1].grid(alpha=.2);fig.tight_layout();fig.savefig(out/'gradient.png');plt.close(fig)
    parameter=spec.get('parameter_label_cn','厚度')
    lines=[f"# {spec['plot_title']}完整{parameter}导数",'',f"{'通过' if passed else '未通过'}预定1%目标；实际终点压缩{metrics['actual_compression']*100:.8f}%。验收目标：{', '.join(required)}。",'',
           '| 目标 | 基线值 | JVP | FD h | FD h/2 | AD/FD差 | 两档FD差 |','|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:lines.append(f"| {row['objective']} | {row['baseline_value']:.9g} | {row['AD']:.9g} | {row['FD_h']:.9g} | {row['FD_half']:.9g} | {row['AD_FD_relative_error']*100:.6g}% | {row['FD_halving_relative_difference']*100:.6g}% |")
    lines+=['','![完整参数敏感度](gradient.png)','',
            f"加载功单位N·mm，其{parameter}导数单位{spec.get('gradient_units',['N','N/mm'])[0]}；力导数单位{force_unit}。材料、几何占据、HRZ质量、XY宏观惯性和端部面积均参与求导。",
            '', '复用前缀时同时载入从零得到的状态、完整参数切向和累计功；保存态不是固定常量，因此这是从零至终点的总导数。每个扰动仍使用自己的独立前缀。新计算成本只含本次延伸，既有前缀成本见5%报告。',
            '', '固定步长与原20%/4ms五次加载不变；实际终点见上述数值，不对自适应控制决策求导。本终点只验证JVP；实际VJP证据仍限于G5%。',
            '', f"主要材料实际J的采样最小值{metrics['sampled_min_required_J']:.6g}；保留全部虚域J和能量。通过证明本离散动态路径的局部导数，不能替代Abaqus/实物敏感度、自接触或通用屈曲分支验证。"]
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    status=json.loads((B/'status.json').read_text());status.update(stage=spec['id']+'_analyzed',active_jobs=[])
    status[spec['id']]='passed' if passed else 'not_passed'
    status[spec.get('plan_step','step3')]='in_progress' if passed else 'not_passed'
    if not passed:status['step4']='gated_not_started'
    write(B/'status.json',status);print(json.dumps(metrics,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--interface',action='store_true');parser.add_argument('--spec');args=parser.parse_args()
    extension(args.spec) if args.spec else interface() if args.interface else main()

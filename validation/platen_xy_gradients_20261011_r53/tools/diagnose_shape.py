"""Low-cost diagnosis using existing complete paths and saved final states.

No time steps, new complete FE solves, smoothing or acceptance changes.
Frozen-state derivatives below are diagnostics, never full-path validation.
"""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
from pathlib import Path
import argparse,json,time,hashlib,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shape import ShapeDesignPath,B,R,jax,jnp
from run import load,ready

parser=argparse.ArgumentParser();parser.add_argument('--spec',default='input/G_shape_diagnostic.json')
args=parser.parse_args()
start=time.time();spec=json.loads((B/args.spec).read_text())
out=B/'analysis'/spec.get('analysis_name','G5_shape_diagnostic');out.mkdir(exist_ok=False)
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def check():
    if time.time()-start>spec['wall_budget_seconds']:raise TimeoutError('Diagnostic budget exhausted')
for p,h in spec['protected_sources'].items():
    if hashlib.sha256((R/p).read_bytes()).hexdigest()!=h:raise ValueError('Protected source changed')
job=json.loads((B/spec.get('job_spec','input/G5_shape_protocol.json')).read_text())
cases=[json.loads((B/c['label']/'result.json').read_text()) for c in job['cases']]
paths=[np.array([[r[k] for k in ['compression','force_N','work_N_mm']] for r in c['path']]) for c in cases]
ad=np.array([[r[k] for k in ['work_gradient_value','force_gradient_value']] for r in cases[0]['path']])
h,half=job['fd_steps']
fd=(paths[1][:,[2,1]]-paths[2][:,[2,1]])/(2*h)
fd2=(paths[3][:,[2,1]]-paths[4][:,[2,1]])/(2*half)
rows=[]
for target in [.01,.02,.03,.04,.05]:
    k=np.argmin(abs(paths[0][:,0]-target))
    for i,name in enumerate(['loading_work','instantaneous_force']):
        rows.append(dict(target_compression=target,actual_sample_compression=float(paths[0][k,0]),objective=name,
            AD=float(ad[k,i]),FD_h=float(fd[k,i]),FD_half=float(fd2[k,i]),
            AD_FD_absolute=float(abs(ad[k,i]-fd2[k,i])),
            AD_FD_relative=float(abs(ad[k,i]-fd2[k,i])/max(abs(fd2[k,i]),1e-300)),
            FD_halving_relative=float(abs(fd[k,i]-fd2[k,i])/max(abs(fd2[k,i]),1e-300))))
with (out/'path_samples.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
model=ShapeDesignPath();p=jnp.array([.5,.05,0.]);seed=jnp.array([0.,0.,1.])
fields,dfields=ready(jax.jvp(model.fields,(p,),(seed,)))
state=load(B/job['cases'][0]['label']/'final_state.npz')
tangent=load(B/job['cases'][0]['label']/'final_tangent.npz')
zeros_s=jax.tree_util.tree_map(jnp.zeros_like,state)
zeros_f=jax.tree_util.tree_map(jnp.zeros_like,fields)
observe_jvp=jax.jit(lambda s,ds,f,df:jax.jvp(model._observe,(s,f),(ds,df))[1])
part_state=np.asarray(ready(observe_jvp(state,tangent,fields,zeros_f)))
part_field=np.asarray(ready(observe_jvp(state,zeros_s,fields,dfields)))

ex=model.ex
def gap(s):
    z=ex.z0+jnp.sum(s[0][ex.ids][ex.cids,2]*ex.sh,axis=1)
    return np.asarray(jnp.where(ex.side==0,z,1-ex.motion(s[-2])-z))
base_gap=gap(state);area=np.asarray(fields[-1]);darea=np.asarray(dfields[-1]);top=np.asarray(ex.side)==1
dz=np.asarray(jnp.sum(tangent[0][ex.ids][ex.cids,2]*ex.sh,axis=1))
pen=np.maximum(-base_gap,0.)
contributions=np.where(top,float(ex.knL)*ex.L**2*(area*(base_gap<0)*dz+pen*darea),0.)
sum_abs=float(abs(contributions).sum());net=float(contributions.sum())
contact=dict(top_active_count=int(np.count_nonzero(top&(base_gap<0))),
    top_force_derivative_by_point_sum_N=net,absolute_point_contributions_N=sum_abs,
    cancellation_ratio=abs(net)/max(sum_abs,1e-300),
    largest_positive_point_contribution_N=float(contributions.max()),
    largest_negative_point_contribution_N=float(contributions.min()),
    minimum_top_gap_abs_with_area_normalized=float(np.min(abs(base_gap[top&(area>1e-14)]))))

fd_diagnostics=[]
for step,plus,minus in [(h,1,2),(half,3,4)]:
    check();sp=load(B/job['cases'][plus]['label']/'final_state.npz');sm=load(B/job['cases'][minus]['label']/'final_state.npz')
    dstate=jax.tree_util.tree_map(lambda x,y:(x-y)/(2*step),sp,sm)
    fp=ready(model.fields(jnp.array(job['cases'][plus]['parameters'])))
    fm=ready(model.fields(jnp.array(job['cases'][minus]['parameters'])))
    df=jax.tree_util.tree_map(lambda x,y:(x-y)/(2*step),fp,fm)
    linearized=np.asarray(ready(observe_jvp(state,dstate,fields,df)))
    actual=(np.array([cases[plus]['terminal']['work_N_mm'],cases[plus]['terminal']['force_N']])-
            np.array([cases[minus]['terminal']['work_N_mm'],cases[minus]['terminal']['force_N']]))/(2*step)
    state_error={name:float(np.linalg.norm(np.asarray(x-y).ravel())/max(np.linalg.norm(np.asarray(y).ravel()),1e-300))
                 for name,x,y in zip(['q','macro','vq','vmacro','aq','amacro','time','work'],dstate,tangent)}
    fd_diagnostics.append(dict(h=step,actual_full_path_FD=actual.tolist(),
        final_observable_linearization_on_FD_state_and_field_tangent=linearized.tolist(),
        observable_nonlinearity_remainder=(actual-linearized).tolist(),
        FD_state_tangent_relative_L2_vs_AD=state_error,
        final_top_active_set_changes_plus=int(np.count_nonzero(top&((gap(sp)<0)!=(base_gap<0)))),
        final_top_active_set_changes_minus=int(np.count_nonzero(top&((gap(sm)<0)!=(base_gap<0))))))

# Smaller geometry-only differences: no dynamic response evaluated here.
geometry_rows=[];names=['scale','nodem','mass','macro_c','macro_schur','rim_area']
for small in spec['geometry_only_fd_steps']:
    check();fp=ready(model.fields(jnp.array([.5,.05,small])));fm=ready(model.fields(jnp.array([.5,.05,-small])))
    df=jax.tree_util.tree_map(lambda x,y:(x-y)/(2*small),fp,fm)
    errors={name:float(np.linalg.norm(np.asarray(x-y).ravel())/max(np.linalg.norm(np.asarray(y).ravel()),1e-300))
            for name,x,y in zip(names,df,dfields)}
    geometry_rows.append(dict(h=small,field_FD_relative_L2_vs_AD=errors))
    print(json.dumps(dict(geometry_only_h=small,wall_s=time.time()-start)),flush=True)

force_peak=float(np.max(abs(ad[:,1])))
result=dict(scope='Saved complete-path diagnosis and smaller geometry-only FD; no new dynamic solve',
    original_full_gate_passed=False,original_work_passed=True,original_terminal_force_passed=False,
    path_samples=rows,terminal_state_partials=part_state.tolist(),terminal_field_partials=part_field.tolist(),
    terminal_AD_sum=(part_state+part_field).tolist(),final_contact=contact,
    FD_diagnostics=fd_diagnostics,geometry_only=geometry_rows,
    force_gradient_path_peak_absolute_N=force_peak,
    terminal_force_gradient_AD_FD_absolute_N=float(abs(ad[-1,1]-fd2[-1,1])),
    actual_force_response_difference_plus_half_minus_half_N=float(paths[3][-1,1]-paths[4][-1,1]),
    normalized_loading_work_sensitivity_at_scale_0p01=float(.01*ad[-1,0]/paths[0][-1,2]),
    normalized_terminal_force_sensitivity_at_scale_0p01=float(.01*ad[-1,1]/paths[0][-1,1]),
    no_time_steps_executed=True,new_full_FE_submissions=0,protected_sources_unchanged=True,
    wall_minutes=(time.time()-start)/60)
write(out/'metrics.json',result)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
fig,axes=plt.subplots(2,2,figsize=(11,7))
for i,title,unit in [(0,'Loading-work shape sensitivity','N mm'),(1,'Instantaneous-force shape sensitivity','N')]:
    ax=axes[0,i]
    for series,label,style in [(ad,'Full-path JVP','-'),(fd,f'Full-path FD h={h:g}','--'),(fd2,f'Full-path FD h={half:g}',':')]:
        ax.plot(paths[0][:,0]*100,series[:,i],ls=style,lw=1.1,label=label)
    ax.set(xlabel='Compression (%)',ylabel=f'd objective / da ({unit})',title=title);ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[1,i].plot(paths[0][:,0]*100,ad[:,i]-fd2[:,i],label='JVP - FD h/2')
    axes[1,i].plot(paths[0][:,0]*100,fd[:,i]-fd2[:,i],label='FD h - FD h/2',ls='--')
    axes[1,i].set(xlabel='Compression (%)',ylabel=f'Absolute signed derivative difference ({unit})',title='Difference along saved complete paths')
    axes[1,i].grid(alpha=.2);axes[1,i].legend(fontsize=8)
fig.tight_layout();fig.savefig(out/'path_diagnosis.png');plt.close(fig)
lines=['# 构型末端力梯度未通过：已有路径与保存态诊断','',
    '原五条完整路径及1%门槛不变：加载功通过，末端瞬时力未通过，所以不启动长反向任务。本诊断没有执行新的时间步。','',
    '| 采样压缩 | 目标 | AD | FD h/2 | AD/FD差 | 两档FD差 |','|---:|---|---:|---:|---:|---:|']
for row in rows:lines.append(f"| {100*row['actual_sample_compression']:.5f}% | {row['objective']} | {row['AD']:.7g} | {row['FD_half']:.7g} | {100*row['AD_FD_relative']:.5g}% | {100*row['FD_halving_relative']:.5g}% |")
lines+=['','![完整路径诊断](path_diagnosis.png)','',
    f"5%末端力为{paths[0][-1,1]:.9g}N；较细差分两条实际力之差只有{result['actual_force_response_difference_plus_half_minus_half_N']:.5g}N。AD/FD导数绝对差{result['terminal_force_gradient_AD_FD_absolute_N']:.5g}N/单位a，仍未过原相对门槛，不能因绝对值小而改标。",'',
    f"按预定形态尺度0.01，无量纲加载功/末端力敏感度为{result['normalized_loading_work_sensitivity_at_scale_0p01']:.5g}/{result['normalized_terminal_force_sensitivity_at_scale_0p01']:.5g}。形态变化确实存在，但这个基线方向的宏观响应一阶变化很弱。",'',
    f"末端逐接触点力导数绝对值之和{sum_abs:.6g}N、净和{net:.6g}N，净和/绝对和{contact['cancellation_ratio']:.5g}。正负贡献相互抵消会放大净导数的相对误差；这不是自动证明误差全部来自舍入。",'',
    '最终观测的状态/字段偏导、FD保存态切向误差和接触活跃集合差异见metrics.json。该偏导仅用于定位问题，不替代原从零完整路径梯度。最终活跃集合相同也不能证明时间历史中始终未切换。较小步长仅重算几何字段，不能据此宣布完整响应梯度通过。']
(out/'README.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['path_samples']},indent=2),flush=True)

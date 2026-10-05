"""Read-only comparison; incomplete paths are never extrapolated to 20%."""
from pathlib import Path
import hashlib,json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/xuehu/projects/tpms_jax');E=R/'validation/jax_improvement_20261006_r8'
P=E/'T0p004_q6_cpu_geometry';Q=R/'validation/large_compression_20261005_r6/quadratic_candidate'
A=R/'validation/large_compression_20261005_r6/abaqus'
read=lambda p:json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def curve(rows):
    a=np.array([r['compression'] for r in rows]);f=-np.array([r['Fz_N'] for r in rows])
    a,idx=np.unique(a,return_index=True)
    return a,f[idx]
def main():
    complete=(P/'result.json').exists();recpath=P/('result.json' if complete else 'failure.json')
    rec=read(recpath);rows=rec['path'] if complete else rec['accepted_path']
    old=read(Q/'T0p004_compact/result.json');slow=read(A/'explicit_T0p040/shell.json')
    standard=read(A/'standard/shell.json');cfg=read(P/'input.json');oldcfg=read(Q/'T0p004_compact/input.json')
    for k in ['N','cell_size_mm','thickness_mm','E_MPa','nu','eta','interface_10_90_mm','mechanical_periodic_axes','load_time_seconds']:
        assert cfg[k]==oldcfg[k],k
    curves={'Original Q2 27-point':curve(old['path']),
            'Shell slow Explicit':curve(slow['force_path']),
            'Q2 64-point attempt':curve(rows)}
    aa,ff=curves['Q2 64-point attempt'];peak=float(np.max(curve(standard['force_path'])[1]))
    # Same 1000-point comparison grid as the original forward report.
    grid=np.linspace(.001,.2,1000);grid=grid[grid<=aa[-1]+1e-12]
    ref=np.interp(grid,*curves['Shell slow Explicit'])
    metric={'candidate_completed_20':complete,'fixed_physics_checked':True,
        'changed_factor':'27 -> 64 Gauss points; force and HRZ mass use same new rule',
        'last_accepted_compression':rows[-1]['compression'],'last_accepted_J_min':rows[-1]['J_min'],
        'minimum_monitored_J':min(r['J_min'] for r in rows),
        'common_range_curve_RMS_over_Standard_peak':float(np.sqrt(np.mean((np.interp(grid,aa,ff)-ref)**2))/peak),
        'common_range':[float(grid[0]),float(grid[-1])],
        'partial_curve_is_not_20pct_accuracy':not complete,
        'body_seconds':rec['body_seconds'],'rejected_blocks':len(rec['rejected_blocks']),
        'reference_quality_certified':False,'physical_replacement_certified':False,'gradient20_certified':False,
        'source_sha256':{str(p.relative_to(R)):sha(p) for p in [recpath,P/'input.json',A/'explicit_T0p040/shell.json',Q/'T0p004_compact/result.json']}}
    if complete:
        times=np.array([r['time'] for r in rows]);compression=np.array([r['compression'] for r in rows])
        forces=np.array([r['Fz_N'] for r in rows]);u=np.array([r['energy_N_mm'] for r in rows]);ke=np.array([r['KE_N_mm'] for r in rows])
        hold=times>=1.05*cfg['load_time_seconds']-1e-12
        mean=float(np.mean(forces[hold]));work=float(np.sum(.5*(forces[1:]+forces[:-1])*(-cfg['cell_size_mm']*np.diff(compression))))
        dt=np.r_[0.,np.diff(times)];loading=(compression>=.01)&(times<=cfg['load_time_seconds']+1e-12)
        ratio=ke/np.maximum(u,1e-30)
        metric.update(hold_mean_Fz_N=mean,hold_min_max_Fz_N=[float(forces[hold].min()),float(forces[hold].max())],
            terminal_force_difference_vs_shell=abs(mean/slow['hold_mean_Fz_N']-1),
            input_work_N_mm=work,input_work_difference_vs_shell=abs(work/slow['macro_work_from_RF_U_N_mm']-1),
            energy_work_gap=abs(u[-1]+ke[-1]-u[0]-ke[0]-work)/abs(work),
            loading_time_fraction_KE_below_5pct=float(np.sum(dt[loading]*(ratio[loading]<=.05))/np.sum(dt[loading])),
            terminal_KE_over_U=float(ratio[-1]),peak_magnitude_N=float(ff.max()),peak_compression=float(aa[ff.argmax()]),
            seconds_per_step_with_monitoring=rec['seconds_per_step_with_monitoring'],peak_RSS_GiB=rec['peak_RSS_GiB'])
    else:metric.update(failure_message=rec['message'],terminal_force_difference_vs_shell=None,input_work_difference_vs_shell=None)
    dense=read(E/'candidate_probe/rule_8.json')['states'][P.name]
    metric.update(dense_sampling_J_min=dense['J_min'],dense_sampling_nonpositive_points=dense['nonpositive_points'],
                  dense_sampling_nonpositive_cells=dense['nonpositive_cells'],dense_sampling_material_defined=dense['total_material_response_defined'])
    if complete:
        metric['response_target_only_pass']=bool(metric['terminal_force_difference_vs_shell']<=.1 and metric['common_range_curve_RMS_over_Standard_peak']<=.1 and metric['input_work_difference_vs_shell']<=.1)
    metric['trusted_improvement_adopted']=False  # Set only after evidence/rate review, never by closer curve alone.
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for name,(x,y) in curves.items():axes[0].plot(x*100,y,label=name,lw=1.5)
    axes[0].set(xlabel='Compression (%)',ylabel='Reaction magnitude (N)',title='Fixed physical inputs, XYZ periodic')
    axes[0].legend(fontsize=7)
    axes[1].plot([r['compression']*100 for r in rows],[r['KE_over_U']*100 for r in rows]);axes[1].axhline(5,color='gray',ls='--')
    axes[1].set(xlabel='Compression (%)',ylabel='KE / elastic energy (%)',ylim=(0,20),title='64-point monitored path')
    axes[2].plot([r['compression']*100 for r in rows],[r['J_min'] for r in rows]);axes[2].axhline(0,color='gray',ls='--')
    axes[2].set(xlabel='Compression (%)',ylabel='Minimum monitored detF',title='Dense terminal check reported separately')
    for ax in axes:ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(E/'response.png',dpi=160);plt.close(fig)
    (E/'comparison.json').write_text(json.dumps(metric,indent=2,allow_nan=False)+'\n')
    shutil.copy2(__file__,E/'compare_at_run.py');print(json.dumps(metric,indent=2))
if __name__=='__main__':main()

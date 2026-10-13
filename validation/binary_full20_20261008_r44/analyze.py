"""Read-only comparison of r44 accepted path with frozen same-rate references."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/xuehu/projects/tpms_jax');D=Path(__file__).resolve().parent
P=json.loads((D/'protocol.json').read_text());T=P['load_time_seconds'];END=T+P['hold_time_seconds']
def write(name,obj):(D/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def arrays(rows):
 keys=['time','compression','Fz_N','energy_N_mm','KE_N_mm']
 return {k:np.array([x.get(k,0.) for x in rows],float) for k in keys}
def interval(a,start,end):
 assert end<=a['time'][-1]+1e-12
 ts=np.unique(np.r_[start,a['time'][(a['time']>start)&(a['time']<end)],end])
 return {k:(ts if k=='time' else np.interp(ts,a['time'],v)) for k,v in a.items()}
def load(a,end=None):return interval(a,0.,min(T,a['time'][-1]) if end is None else end)
def work(a):return float(np.trapezoid(-a['Fz_N'],10*a['compression']))
def observed_peak(a):
 j=int(np.argmax(-a['Fz_N']));return {'compression':float(a['compression'][j]),'force_N':float(-a['Fz_N'][j]),'time':float(a['time'][j])}
def rms(a,b,upper):
 grid=np.linspace(0.,upper,P['comparison_grid_points']);fa=np.interp(grid,a['compression'],-a['Fz_N']);fb=np.interp(grid,b['compression'],-b['Fz_N']);delta=fa-fb
 val=float(np.sqrt(np.mean(delta**2)))
 return {'upper_compression':upper,'points':len(grid),'RMS_N':val,'relative_to_old_fixed_peak':val/P['old_fixed_curve_denominator_N'],'relative_to_matched_shell_peak':val/P['shell_matched_loading_peak_N']}
def inertia(a):
 ratio=a['KE_N_mm']/np.maximum(a['energy_N_mm'],1e-30);dt=np.diff(a['time']);mid=.5*(ratio[1:]+ratio[:-1]);j=np.argsort(mid);cdf=np.cumsum(dt[j])/dt.sum();p95=float(mid[j[min(np.searchsorted(cdf,.95),len(j)-1)]])
 return {'duration_seconds':float(dt.sum()),'terminal_KE_over_U':float(ratio[-1]),'loading_time_fraction_sampled_KE_below_5pct':float(np.sum(dt*.5*((ratio[:-1]<=.05).astype(float)+(ratio[1:]<=.05).astype(float)))/dt.sum()),'time_weighted_midpoint_KE_over_U_p95':p95,'maximum_sampled_KE_over_U_after_1pct':float(np.max(ratio[a['compression']>=.01])) if np.any(a['compression']>=.01) else None}
def main():
 F=D/'continuation_from_budget' if (D/'continuation_receipt.json').exists() else D/'full_from_zero';result_file=F/'result.json';failure_file=F/'failure.json'
 if not result_file.exists() and not failure_file.exists():raise RuntimeError('Forward still running; no final analysis')
 completed=result_file.exists();forward=json.loads((result_file if completed else failure_file).read_text());rows=json.loads((F/'accepted_path.json').read_text());a=arrays(rows)
 shell=json.loads(Path(P['shell_reference']).read_text());sp=shell['force_path'];b=arrays([dict(x,energy_N_mm=x['energies_N_mm']['ALLSE'] if 'energies_N_mm' in x else x.get('ALLSE',0.),KE_N_mm=x.get('ALLKE',0.)) for x in sp]);smooth=arrays(json.loads((R/'validation/step_control_20261007_r18/controlled_forward/accepted_path.json').read_text()))
 loading_completed=bool(a['time'][-1]>=T-1e-12 and abs(a['compression'][-1]-.2)<1e-12)
 al=load(a);bl=load(b);sl=load(smooth);upper=float(min(al['compression'][-1],bl['compression'][-1]));cl=interval(bl,0.,min(T,al['time'][-1]))
 pa=observed_peak(al);pb=observed_peak(bl);ps=observed_peak(sl)
 shared_upper=float(min(al['compression'][-1],sl['compression'][-1]));total_work=work(a);U=a['energy_N_mm'];KE=a['KE_N_mm'];cumulative=np.r_[0.,np.cumsum(.5*(-a['Fz_N'][1:]-a['Fz_N'][:-1])*10*np.diff(a['compression']))];balance=cumulative-(U+KE-U[0]-KE[0]);den=max(abs(total_work),1e-30)
 checks={'completed_to_end_time':bool(completed and abs(a['time'][-1]-END)<1e-10),'compression_20pct':loading_completed,'all_accepted_observables_finite':all(np.isfinite(v).all() for v in a.values()),'all_accepted_solid_Gauss_J_positive':all(x['invalid_material_points']==0 and x['required_positive_J_min']>0 for x in rows),'full_curve_within_10pct_matched_peak':loading_completed and rms(al,bl,.2)['relative_to_matched_shell_peak']<=.1,'global_work_energy_drift_le_1pct':float(np.max(np.abs(balance)))/den<=.01}
 data={'completed':completed,'full_20pct_loading_completed':loading_completed,'full_hold_completed':completed,'stop_reason':None if completed else forward['message'],'total_wall_seconds':(json.loads((D/'continuation_receipt.json').read_text())['total_elapsed_since_first_launch_seconds'] if (D/'continuation_receipt.json').exists() else json.loads((D/'receipt.json').read_text())['total_wall_seconds']),'body_seconds':forward['body_seconds'],'bound_seconds':forward['bound_seconds_total'],'accepted_end':rows[-1],'accepted_rows':len(rows),'rejected_blocks':len(forward['rejected_blocks']),'completed_steps':forward.get('steps',forward.get('completed_steps')),'binary_initial_K_N_per_mm':4.719388773981349,'smooth_initial_K_N_per_mm':4.916454681739766,'shell_initial_K_N_per_mm':4.651218897465464,'binary_initial_bias':4.719388773981349/4.651218897465464-1,'curve_vs_matched_shell':rms(al,bl,upper),'shared_smooth_binary_window':{'upper_compression':shared_upper,'binary_vs_shell':rms(al,bl,shared_upper),'smooth_vs_shell':rms(sl,bl,shared_upper),'binary_vs_smooth':rms(al,sl,shared_upper)},'observed_binary_peak':pa,'observed_smooth_partial_peak':ps,'observed_shell_peak':pb,'peak_compression_shift_percentage_points':100*(pa['compression']-pb['compression']),'peak_force_relative_difference':pa['force_N']/pb['force_N']-1,'covered_loading_work_N_mm':work(al),'same_time_covered_shell_work_N_mm':work(cl),'covered_loading_work_relative_difference':work(al)/work(cl)-1,'total_macro_work_N_mm':total_work,'terminal_internal_energy_N_mm':float(U[-1]),'terminal_KE_N_mm':float(KE[-1]),'max_sampled_energy_balance_residual_N_mm':float(np.max(np.abs(balance))),'max_energy_balance_residual_relative_total_work':float(np.max(np.abs(balance)))/den,'terminal_energy_balance_residual_relative_total_work':float(balance[-1])/den,'loading_inertia':inertia(al),'minimum_accepted_required_positive_J':min(x['required_positive_J_min'] for x in rows),'minimum_accepted_all_J':min(x['J_min'] for x in rows),'maximum_accepted_virtual_negative_J_points':max(x['negative_J_points'] for x in rows),'initial_required_positive_J_points':rows[0]['required_positive_J_points'],'minimum_accepted_dt_seconds':min(x['dt_seconds'] for x in rows),'minimum_dt_before_final_time_alignment_seconds':min(x['dt_seconds'] for x in rows[1:-1]),'final_time_alignment_dt_seconds':rows[-1]['dt_seconds'],'checks':checks,'shell_checks_unchanged':shell['checks'],'comparison_scope':'Binary27 numerical diagnostic, same physical loading rate, no fitted shift/scale. Smooth reference partial only. High KE permitted; energy and integration quality assessed separately; no full-path AD.'}
 if completed:
  hold_end=float(min(END,a['time'][-1],b['time'][-1]));hold=interval(a,T,hold_end);mean=float(np.trapezoid(hold['Fz_N'],hold['time'])/(hold_end-T));shell_hold=interval(b,T,hold_end);shell_mean=float(np.trapezoid(shell_hold['Fz_N'],shell_hold['time'])/(hold_end-T));data['hold_comparison_time_window_seconds']=[T,hold_end];data['shell_saved_terminal_time_seconds']=float(b['time'][-1]);data.update(hold_mean_Fz_N=mean,shell_hold_mean_Fz_N=shell_mean,shell_original_report_hold_mean_Fz_N=shell['hold_mean_Fz_N'],hold_mean_force_relative_difference=mean/shell_mean-1,hold_Fz_min_max_N=[float(hold['Fz_N'].min()),float(hold['Fz_N'].max())]);checks['hold_mean_force_within_10pct']=abs(data['hold_mean_force_relative_difference'])<=.1
 if (D/'continuation_receipt.json').exists():
  prior_failure=json.loads((D/'full_from_zero/failure.json').read_text());data['execution_phases']=[{'role':'from_zero_original_budget_stop','body_seconds':prior_failure['body_seconds'],'completed_steps':prior_failure['completed_steps'],'end_compression':prior_failure['last_valid']['compression'],'stop_reason':prior_failure['message']},{'role':'exact_checkpoint_continuation','body_seconds':forward['body_seconds'],'continuity_verified':json.loads((D/'continuity_check.json').read_text())['passed']}];data['bound_seconds']+=prior_failure['bound_seconds_total'];data['body_seconds']+=prior_failure['body_seconds'];data['selected_total_budget_seconds']=json.loads((D/'selected_continuation_budget.json').read_text())['total_budget_seconds']
 data['force_at_fixed_compressions_N']=[{'compression':x,'binary':float(np.interp(x,al['compression'],-al['Fz_N'])),'shell':float(np.interp(x,bl['compression'],-bl['Fz_N'])),'smooth_partial':float(np.interp(x,sl['compression'],-sl['Fz_N'])) if x<=sl['compression'][-1] else None} for x in (.01,.05,.10,.12,.14,.16,.18,.20) if x<=al['compression'][-1]+1e-12]
 data['early_0_5_curve_vs_shell']=rms(al,bl,min(.05,upper))
 write('comparison.json',data)
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
 fig,axes=plt.subplots(1,2,figsize=(11.7,4.3),gridspec_kw={'width_ratios':[1.4,1]})
 for ax in axes:
  ax.plot(bl['compression']*100,-bl['Fz_N'],color='#343f4b',lw=1.6,label='Abaqus shell, T=4 ms')
  ax.plot(sl['compression']*100,-sl['Fz_N'],color='#3878ae',lw=1.3,label='JAX smooth N32 (partial r18)')
  ax.plot(al['compression']*100,-al['Fz_N'],color='#bd542d',lw=1.6,label='JAX binary N32 (r44)');ax.set(xlabel='Compression (%)',ylabel='Compressive force (N)');ax.grid(alpha=.18)
 axes[0].set_xlim(0,20);axes[0].legend(fontsize=8);axes[0].set_title('Original compression axis; no shift or fitted scaling')
 early_top=max(float(np.max(-z['Fz_N'][z['compression']<=.05])) for z in (al,bl,sl))
 axes[1].set_xlim(0,5);axes[1].set_ylim(0,1.1*early_top);axes[1].set_title('Early loading (0-5%)')
 fig.tight_layout();fig.savefig(D/'force_comparison.png');plt.close(fig)
 fig,axs=plt.subplots(2,2,figsize=(11.7,7.4));tt=a['time']*1000
 axs[0,0].plot(tt,U,label='Internal U');axs[0,0].plot(tt,KE,label='Kinetic KE');axs[0,0].plot(tt,cumulative,label='Macro work');axs[0,0].set(ylabel='N mm',title='Whole modeled-domain energies');axs[0,0].legend(fontsize=8)
 axs[0,1].plot(tt,balance/den*100,color='#bd542d');axs[0,1].set(ylabel='Residual / total work (%)',title='W - (U + KE - initial energy)')
 axs[1,0].plot(tt,[x['required_positive_J_min'] for x in rows],label='Min solid-Gauss J');axs[1,0].axhline(0,color='k',lw=.7);axs[1,0].set(ylabel='Actual det F',title='Required positive J domain');axs[1,0].legend(fontsize=8)
 axs[1,1].plot(tt,np.array([x['dt_seconds'] for x in rows])*1e9,color='#3878ae');axs[1,1].set(ylabel='Accepted dt (ns)',title='State-dependent conservative time step')
 for ax in axs.ravel():ax.set_xlabel('Time (ms)');ax.axvline(T*1000,color='#999999',ls=':',lw=1);ax.grid(alpha=.18)
 fig.tight_layout();fig.savefig(D/'path_diagnostics.png');plt.close(fig)
 print(json.dumps({k:data[k] for k in ['completed','stop_reason','total_wall_seconds','accepted_end','observed_binary_peak','observed_shell_peak','curve_vs_matched_shell','covered_loading_work_relative_difference','max_energy_balance_residual_relative_total_work','checks']},indent=2))
if __name__=='__main__':main()

from pathlib import Path
import json,hashlib,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/xuehu/projects/tpms_jax');D=Path(__file__).resolve().parent
W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
F=D/'forward';new=json.loads((F/'accepted_path.json').read_text())
old=json.loads((R/'validation/step_control_20261007_r18/controlled_forward/accepted_path.json').read_text())
shell_data=json.loads((R/'validation/shell_rate_20261007_r20/abaqus/explicit_T0p004/results/shell.json').read_text())
shell=[p for p in shell_data['force_path'] if p['time']<=.0040000001]
receipt=json.loads((D/'receipt.json').read_text())
result=json.loads((F/'result.json').read_text()) if (F/'result.json').exists() else None
failure=json.loads((F/'failure.json').read_text()) if (F/'failure.json').exists() else None
def arrays(path,key='Fz_N'):
 a=np.array([p['compression'] for p in path]);y=np.array([p[key] for p in path])
 # Histories have monotone compression during loading; no hold/double x here.
 assert np.all(np.diff(a)>=-1e-12)
 mask=np.r_[True,np.diff(a)>1e-12]
 return a[mask],y[mask]
def peak(path,upper=.17):
 ps=[p for p in path if .01<=p['compression']<=upper+1e-12]
 if not ps:return None
 p=max(ps,key=lambda r:-r['Fz_N']);idx=path.index(p)
 after=[r for r in path[idx+1:] if r['compression']<=upper+1e-12]
 drop=max((1-r['Fz_N']/p['Fz_N'] for r in after),default=0.)
 span=max((r['compression']-p['compression'] for r in after),default=0.)
 return {'compression':p['compression'],'compressive_force_N':-p['Fz_N'],
  'post_peak_span_percentage_points':100*span,'largest_post_peak_fraction_drop':drop,
  'observed_drop_peak_qualified':span>=.005 and drop>=.1,
  'definition':'largest saved dynamic compressive force within declared loading window; not an eigenvalue buckling point'}
sp=peak(shell,.2);den=sp['compressive_force_N']
assert abs(den-5.229988098144531)<1e-10
def pair(p1,p2,window,key='Fz_N'):
 a,y=arrays(p1,key);b,z=arrays(p2,key);lo=max(window[0],a[0],b[0]);hi=min(window[1],a[-1],b[-1])
 if hi<=lo:return None
 grid=np.linspace(lo,hi,1001);u=np.interp(grid,a,y);v=np.interp(grid,b,z)
 rms=float(np.sqrt(np.trapezoid((u-v)**2,grid)/(hi-lo)))
 wu=float(np.trapezoid(-u*10,grid));wv=float(np.trapezoid(-v*10,grid))
 return {'requested_window':window,'actual_common_window':[float(lo),float(hi)],
   'full_requested_window_covered':bool(hi>=window[1]-1e-12 and lo<=window[0]+1e-12),
   'curve_RMS_N':rms,'RMS_relative_original_fixed_2p25891N':rms/2.25891,
   'RMS_relative_matched_shell_peak':rms/den,'work_first_N_mm':wu,'work_second_N_mm':wv,
   'work_relative_second':wu/wv-1 if abs(wv)>1e-12 else None,
   'integration':'linear interpolation to common uniform 1001-point compression grid, trapezoid, no smoothing or shift'}
windows={'main':[.01,.17],'peak_zone':[.10,.17],'early':[.01,.08]}
comparison={name:{'new_vs_original_JAX':pair(new,old,win),
 'new_vs_shell':pair(new,shell,win),'original_JAX_vs_shell':pair(old,shell,[win[0],min(win[1],new[-1]['compression'])]),
 'original_JAX_vs_shell_full_requested_window':pair(old,shell,win),
 'new_vs_original_internal_force':pair(new,old,win,'internal_macro_Fz_N')} for name,win in windows.items()}
npk=peak(new);opk=peak(old)
full=new[-1]['compression']>=.17-1e-12 and result is not None
move=100*(npk['compression']-opk['compression']) if npk and opk and npk['observed_drop_peak_qualified'] else None
curve_sensitive=comparison['main']['new_vs_original_JAX']['RMS_relative_matched_shell_peak']>=.02
interpretable=full and bool(npk and npk['observed_drop_peak_qualified'])
if interpretable:
 decision='visible_sensitivity' if abs(move)>=.5 or curve_sensitive else 'low_occupancy_support_priority_reduced'
else:decision='incomplete_or_no_qualified_drop_peak_no_insensitivity_conclusion'
work=np.r_[0.,np.cumsum(.5*(np.array([-p['Fz_N'] for p in new[1:]])+np.array([-p['Fz_N'] for p in new[:-1]]))*10*np.diff([p['compression'] for p in new]))]
energy=np.array([p['energy_N_mm']+p['KE_N_mm'] for p in new]);drift=energy-energy[0]-work
frozen=json.loads((D/'freeze_before.json').read_text())
unchanged={p:hashlib.sha256((R/p).read_bytes()).hexdigest()==h for p,h in frozen.items()}
bounds=json.loads((F/'stability_path.json').read_text())
ratios=[p['dt_seconds']*np.sqrt(b['R_s_minus2']) for p,b in zip(new[1:],bounds[1:])]
summary={'scope':'one from-zero diagnostic, no default/20pct/design-AD certification',
 'support_energy_factor_deep_void':.1,'complete_17pct_diagnostic':full,'end':new[-1],
 'accepted_endpoints':len(new),'new_peak':npk,'original_JAX_peak':opk,'matched_shell_peak':sp,
 'new_minus_original_peak_percentage_points':move,
 'new_minus_shell_peak_percentage_points':100*(npk['compression']-sp['compression']) if npk and npk['observed_drop_peak_qualified'] else None,
 'original_minus_shell_peak_percentage_points':100*(opk['compression']-sp['compression']),
 'peak_force_relative_shell_new':npk['compressive_force_N']/den-1 if npk and npk['observed_drop_peak_qualified'] else None,
 'comparisons':comparison,'decision':decision,'peak_sensitivity_interpretable':interpretable,
 'invalid_required_NH_points_max':max(p['invalid_material_points'] for p in new),
 'required_actual_J_min':min(p['required_positive_J_min'] for p in new),
 'all_actual_J_min':min(p['J_min'] for p in new),'negative_J_points_max':max(p['negative_J_points'] for p in new),
 'accepted_endpoint_positive_frequency_ratio_max':float(max(ratios,default=0)),
 'rejected_blocks':len(json.loads((F/'rejected_blocks.json').read_text())),
 'energy_balance':{'terminal_work_N_mm':float(work[-1]),'terminal_U_plus_KE_minus_work_N_mm':float(drift[-1]),
  'max_absolute_drift_relative_max_work':float(np.max(np.abs(drift))/max(np.max(np.abs(work)),1e-30)),
  'scope':'saved endpoint trapezoidal work; not exact stepwise energy certification'},
 'cost':receipt,'control_result':{k:v for k,v in (result or failure or {}).items() if k not in ('path','accepted_path','rejected_blocks')},
 'old_shell_checks_preserved':shell_data['checks'],'frozen_files_unchanged':unchanged}
(D/'comparison.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
fig,axs=plt.subplots(3,1,figsize=(9,10),sharex=True,layout='constrained')
for path,label,col in [(shell,'Abaqus shell, r20, T=0.004 s','C2'),(old,'Original JAX, r18 (partial)','C0'),(new,'JAX low-occupancy support x0.1, r26','C3')]:
 a,y=arrays(path);mask=(a>=.01)&(a<=.1755)
 axs[0].plot(100*a[mask],-y[mask],label=label,color=col)
 pk=peak(path,.17)
 if pk and pk['observed_drop_peak_qualified']:
  axs[0].plot(100*pk['compression'],pk['compressive_force_N'],'o',color=col)
  axs[0].annotate(f"{100*pk['compression']:.3f}%",(100*pk['compression'],pk['compressive_force_N']),xytext=(5,7),textcoords='offset points',color=col)
 if path is not shell:
  axs[1].plot(100*a[mask],np.array([r['energy_N_mm'] for r in path])[mask],label=label+' : U',color=col)
  axs[1].plot(100*a[mask],np.array([r['KE_N_mm'] for r in path])[mask],label=label+' : KE',color=col,linestyle=':')
  axs[2].plot(100*a[mask],np.array([r['required_positive_J_min'] for r in path])[mask],label=label,color=col)
axs[0].set_ylabel('Compressive macro reaction (N)');axs[1].set_ylabel('Energy (N mm)')
axs[2].set_ylabel('Min actual J, occupancy >=0.01');axs[2].set_xlabel('Macroscopic compression (%)')
for ax in axs:
 ax.grid(alpha=.25);ax.legend(fontsize=8);ax.set_xlim(1,17.55)
 ax.axvline(100*new[-1]['compression'],color='C3',linestyle='--',linewidth=.7)
fig.suptitle('diverse_04: one low-occupancy support-energy factor\nSame wall, occupancy, HRZ mass and loading; saved endpoints only')
fig.savefig(D/'response.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
for path,label,col in [(old,'Original JAX r18','C0'),(new,'Low-occupancy support x0.1 r26','C3')]:
 a=np.array([p['compression'] for p in path[1:]])
 ax.plot(100*a,1e6*np.array([p['dt_seconds'] for p in path[1:]]),label=label,color=col)
ax.axhline(1e6*1.6832955867050256e-8,color='grey',linestyle=':',label='Unchanged initial/16 safeguard')
ax.set_xlabel('Macroscopic compression (%)');ax.set_ylabel('Accepted time step (microseconds)')
ax.set_xlim(0,17.6);ax.grid(alpha=.25);ax.legend();ax.set_title('Same state-bound control policy; different trajectories and time steps')
fig.savefig(D/'stepsize.png',dpi=180);plt.close(fig)
out=W/'output/r26_void_support_diagnostic_20261007';out.mkdir(exist_ok=True)
for n in ['comparison.json','response.png','stepsize.png','PROTOCOL.md','local_checks.json','receipt.json','bound_at_checkpoint10.json','bound_at_checkpoint10_original.json','occupancy_partition.json']:shutil.copy2(D/n,out/n)
print(json.dumps({k:summary[k] for k in ['complete_17pct_diagnostic','accepted_endpoints','new_peak','new_minus_original_peak_percentage_points','new_minus_shell_peak_percentage_points','decision','required_actual_J_min','rejected_blocks','energy_balance']},indent=2))

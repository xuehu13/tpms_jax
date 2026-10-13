"""Read-only analysis of the explicitly partial N64 peak diagnostic."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/xuehu/projects/tpms_jax');D=Path(__file__).resolve().parent
P=json.loads((D/'protocol.json').read_text());T=P['load_time_seconds']
def access(p):
 v=str(p).replace(chr(92),'/')
 if v.startswith('//wsl.localhost/Ubuntu-24.04/'):v='/'+v.split('/Ubuntu-24.04/',1)[1]
 return Path(v)
def write(n,x):(D/n).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def arrays(rows):return {k:np.array([x.get(k,0.) for x in rows],float) for k in ('time','compression','Fz_N','energy_N_mm','KE_N_mm')}
def loading(rows,t):return [r for r in rows if r['time']<=t+1e-12]
def peak(rows):
 r=max((r for r in rows if r['compression']>=.01),key=lambda r:-r['Fz_N']);return dict(compression=r['compression'],force_N=-r['Fz_N'],time=r['time'])
def rms(a,b,upper):
 grid=np.linspace(0,upper,2001);v=np.interp(grid,a['compression'],-a['Fz_N'])-np.interp(grid,b['compression'],-b['Fz_N']);r=float(np.sqrt(np.mean(v*v)));return dict(upper_compression=upper,points=2001,RMS_N=r,relative_to_matched_shell_peak=r/P['matched_peak_N'],relative_to_old_fixed_peak=r/P['old_fixed_peak_N'])
def work(a):return float(np.trapezoid(-a['Fz_N'],10*a['compression']))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 F=D/'peak_N64';monitor=json.loads((D/'peak_monitor_receipt.json').read_text());assert 'finished_unix' in monitor
 rows=json.loads((F/'accepted_path.json').read_text());assert len(rows)>1
 final=json.loads((F/('result.json' if (F/'result.json').exists() else 'failure.json')).read_text())
 manual=json.loads((D/'peak_manual_stop_request.json').read_text()) if (D/'peak_manual_stop_request.json').exists() else None
 planned=monitor.get('planned_stop') or manual
 sh=json.loads(access(P['matched_shell_json']).read_text());sr=loading(sh['force_path'],T);lr=loading(rows,T);a=arrays(lr);b=arrays(sr)
 upper=float(a['compression'][-1]);grid=np.unique(np.r_[0,b['compression'][(b['compression']>0)&(b['compression']<upper)],upper]);bc={'compression':grid,'Fz_N':np.interp(grid,b['compression'],b['Fz_N'])}
 pa=peak(lr);pb=peak(sr);cum=np.r_[0,np.cumsum(.5*(-a['Fz_N'][1:]-a['Fz_N'][:-1])*10*np.diff(a['compression']))];res=cum-(a['energy_N_mm']+a['KE_N_mm']-a['energy_N_mm'][0]-a['KE_N_mm'][0]);den=max(abs(work(a)),1e-30)
 bounds=json.loads((F/'stability_path.json').read_text());limit=max(r['dt_seconds']*np.sqrt(q['R_s_minus2']) for r,q in zip(rows[1:],bounds[1:]))
 frozen=json.loads((D/'frozen_before.json').read_text());zero=json.loads((D/'stopped_zero_state_hashes.json').read_text())
 sameT=[r for r in rows if r['compression']>=.01];after=[r for r in sameT if r['time']>=pa['time']]
 old=json.loads((R/'validation/binary_full20_20261008_r44/comparison.json').read_text())
 result={'N':64,'occupancy':'binary Gauss, eta/C2 void retained','load_time_seconds':T,'full_20pct_completed':False,'hold_performed':False,'new_N32_runs':0,'observed_loading_maximum':pa,'matched_shell_peak':pb,'observed_maximum_minus_shell_peak_compression_pp':100*(pa['compression']-pb['compression']),'observed_maximum_vs_shell_peak_force_relative_difference':pa['force_N']/pb['force_N']-1,'last_accepted':rows[-1],'covered_curve_comparison':rms(a,b,upper),'covered_work_N_mm':work(a),'same_compression_shell_covered_work_N_mm':work(bc),'covered_work_relative_difference':work(a)/work(bc)-1,'postpeak_compression_increment':rows[-1]['compression']-pa['compression'],'accepted_rows':len(rows),'accepted_steps':final.get('steps',final.get('completed_steps')),'rejected_blocks':len(final['rejected_blocks']),'original_solver_status':final.get('status',final.get('type')),'stop_request':planned,'stop_is_numerical_failure':not bool(planned) and final.get('type') not in ('TimeoutError',None),'total_wall_seconds':monitor['total_elapsed_seconds'],'N64_wall_seconds':monitor['forward_wall_seconds'],'body_seconds':final['body_seconds'],'bound_seconds':final['bound_seconds_total'],'max_sampled_whole_domain_energy_residual_N_mm':float(np.max(np.abs(res))),'max_energy_residual_relative_covered_work':float(np.max(np.abs(res)))/den,'terminal_KE_over_U':rows[-1]['KE_over_U'],'minimum_sampled_solid_J':min(r['required_positive_J_min'] for r in rows),'minimum_sampled_all_J':min(r['J_min'] for r in rows),'maximum_virtual_negative_J_points':max(r['negative_J_points'] for r in rows),'initial_solid_required_J_points':rows[0]['required_positive_J_points'],'max_accepted_endpoint_dt_sqrt_R':float(limit),'sampled_dt_min_seconds':min(r['dt_seconds'] for r in rows),'all_accepted_finite':all(np.isfinite(v).all() for v in a.values()),'all_accepted_solid_J_positive':all(r['required_positive_J_min']>0 and r['invalid_material_points']==0 for r in rows),'shell_quality_checks_preserved':sh['checks'],'prior_scientific_files_unchanged':all(sha(R/n)==h for n,h in frozen.items()),'prior_files_count':len(frozen),'stopped_zero_state_files_unchanged':all(sha(D/n)==h for n,h in zero.items()),'old_N32_T4ms_context':{'peak':old['observed_binary_peak'],'shell_peak':old['observed_shell_peak'],'delay_percentage_points':old['peak_compression_shift_percentage_points'],'comparison_limit':'Different T; no matched new N32 as user cancelled it. Mesh and loading rate cannot be separated.'},'no_AD':True,'scope':'Observed dynamic loading maximum within covered interval, followed by planned postpeak advance if achieved. No future/global peak, static critical load, pure mesh causality, full20/contact or design AD certificate.'}
 if (D/'peak_stop_request.json').exists():result['planned_stop_evidence']=json.loads((D/'peak_stop_request.json').read_text())
 j=min(range(len(lr)),key=lambda j:abs(lr[j]['time']-pa['time']));near=lr[max(0,j-1):min(len(lr),j+2)]
 result['observed_maximum_state']={k:lr[j][k] for k in ('KE_over_U','energy_N_mm','KE_N_mm','required_positive_J_min','dt_seconds')}
 result['peak_neighbor_sample_compressions']=[r['compression'] for r in near]
 result['peak_neighbor_scope']='Accepted endpoint sampling only; neighbor spacing is not a rigorous peak uncertainty bound.'
 result['observed_maximum_is_interior']=bool(j<len(lr)-1)
 result['accepted_postpeak_rows']=len(lr)-1-j
 result['peak_turnover_with_three_lower_rows']=bool(len(lr)-1-j>=3 and all(-r['Fz_N']<pa['force_N'] for r in lr[-3:]))
 result['if_peak_at_endpoint']='Endpoint maximum is still rising/unconfirmed; do not report it as captured peak.'
 result['hold_started']=bool(rows[-1]['time']>T+1e-12)
 result['force_at_fixed_compressions_N']=[{'compression':c,'N64':float(np.interp(c,a['compression'],-a['Fz_N'])),'matched_shell':float(np.interp(c,b['compression'],-b['Fz_N']))} for c in (.01,.05,.10,.12,.14,.16) if c<=upper]
 result['endpoint_matched_shell_force_N']=float(np.interp(upper,b['compression'],-b['Fz_N']))
 result['endpoint_same_compression_force_relative_difference']=(-float(a['Fz_N'][-1])/result['endpoint_matched_shell_force_N']-1)
 result['captured_N64_peak']=None if not result['peak_turnover_with_three_lower_rows'] else pa
 result['interpretation']='Observed maximum is the still-rising endpoint, NOT a captured N64 peak. The 0.80095pp is an endpoint-vs-shell-peak comparison only. No pure mesh cause or full20 accuracy claim.'
 write('comparison.json',result)
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
 fig,axs=plt.subplots(1,2,figsize=(12,4.5),gridspec_kw={'width_ratios':[1.25,1]})
 axs[0].plot(b['compression']*100,-b['Fz_N'],color='#39434e',label='Abaqus shell, T=1 ms (full loading)')
 axs[0].plot(a['compression']*100,-a['Fz_N'],color='#bd542d',lw=1.7,label='JAX binary N64, T=1 ms (partial)')
 for p,c,n in [(pa,'#bd542d','N64'),(pb,'#39434e','shell')]:
  axs[0].plot(p['compression']*100,p['force_N'],'o',color=c);axs[0].annotate(("N64 endpoint" if n=="N64" and not result["observed_maximum_is_interior"] else n)+f" {p['compression']*100:.2f}%",(p['compression']*100,p['force_N']),xytext=(0,10 if n=='N64' else -18),textcoords='offset points',ha='center',color=c,fontsize=8)
 axs[0].axvline(upper*100,color='#bd542d',ls=':',lw=.8);axs[0].set_title('Same-rate peak diagnostic; no shift or fitted scaling');axs[0].set_xlim(0,20);axs[0].legend(fontsize=8)
 oldrows=json.loads((R/'validation/binary_full20_20261008_r44/continuation_from_budget/accepted_path.json').read_text());oa=arrays(loading(oldrows,.004));old_protocol=json.loads((R/'validation/binary_full20_20261008_r44/protocol.json').read_text());osh=json.loads(access(old_protocol['shell_reference']).read_text())
 if osh:
  ob=arrays(loading(osh['force_path'],.004));axs[1].plot(ob['compression']*100,-ob['Fz_N'],color='#39434e',label='Old shell, T=4 ms')
 axs[1].plot(oa['compression']*100,-oa['Fz_N'],color='#3878ae',label='Old binary N32, T=4 ms')
 axs[1].set_title('Old context: different rate, not a mesh-only comparison');axs[1].set_xlim(0,20);axs[1].legend(fontsize=8)
 for ax in axs:ax.set(xlabel='Compression (%)',ylabel='Compressive force (N)');ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(D/'peak_comparison.png');plt.close(fig)
 fig,axs=plt.subplots(2,2,figsize=(11,7));t=a['time']*1000
 axs[0,0].plot(t,a['energy_N_mm'],label='Internal U');axs[0,0].plot(t,a['KE_N_mm'],label='Kinetic KE');axs[0,0].plot(t,cum,label='Macro work');axs[0,0].set(ylabel='N mm',title='Whole modeled domain');axs[0,0].legend(fontsize=8)
 axs[0,1].plot(t,res/den*100);axs[0,1].set(ylabel='Residual / covered work (%)',title='Work-energy accounting')
 axs[1,0].plot(t,[r['required_positive_J_min'] for r in lr]);axs[1,0].set(ylabel='Actual det F',title='Minimum sampled solid J')
 axs[1,1].plot(t[1:],np.array([r['dt_seconds'] for r in lr[1:]])*1e9);axs[1,1].set(ylabel='Accepted advance dt (ns)',title='State-dependent stable time step')
 for ax in axs.ravel():ax.set_xlabel('Time (ms)');ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(D/'path_diagnostics.png');plt.close(fig)
 print(json.dumps({k:result[k] for k in ('observed_loading_maximum','matched_shell_peak','observed_maximum_minus_shell_peak_compression_pp','observed_maximum_vs_shell_peak_force_relative_difference','last_accepted','covered_curve_comparison','total_wall_seconds','max_energy_residual_relative_covered_work','prior_scientific_files_unchanged')},indent=2))
if __name__=='__main__':main()

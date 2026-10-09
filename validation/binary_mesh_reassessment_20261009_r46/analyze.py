"""Retrospective comparison of existing binary paths; no mechanics solve or AD."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/xuehu/projects/tpms_jax');D=Path(__file__).resolve().parent

def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def access(v):
 s=str(v).replace(chr(92),'/')
 return Path('/'+s.split('/Ubuntu-24.04/',1)[1]) if s.startswith('//wsl.localhost/Ubuntu-24.04/') else Path(s)
def path(rows,T):
 a=[r for r in rows if r['time']<=T+1e-12]
 return {'compression':np.array([r['compression'] for r in a]),'force':np.array([-r['Fz_N'] for r in a]),'rows':a}
def at(a,c,k='force'):
 y=a[k] if k in a else np.array([r[k] for r in a['rows']])
 return float(np.interp(c,a['compression'],y))
def work(a,c):
 x=a['compression'];g=np.r_[0,x[(x>0)&(x<c)],c]
 return float(np.trapezoid(np.interp(g,x,a['force']),10*g))
def shell_peak(a):
 i=int(np.argmax(a['force']));return {'compression':float(a['compression'][i]),'force_N':float(a['force'][i])}
def main():
 start=time.perf_counter();p=load(D/'protocol.json');a44=R/'validation/binary_full20_20261008_r44';a45=R/'validation/binary_n64_20261009_r45'
 p32=load(a44/'protocol.json');p64=load(a45/'protocol.json');c32=load(a44/'comparison.json');c64=load(a45/'comparison.json')
 m32=path(load(a44/'continuation_from_budget/accepted_path.json'),.004);m64=path(load(a45/'peak_N64/accepted_path.json'),.001)
 sh32_path=access(p32['shell_reference']);sh64_path=access(p64['matched_shell_json']);shfast_path=a45/'abaqus/explicit_T0p000250/results/shell.json'
 s32=path(load(sh32_path)['force_path'],.004);s64=path(load(sh64_path)['force_path'],.001);sf=path(load(shfast_path)['force_path'],.00025)
 upper=float(m64['compression'][-1]);assert abs(upper-p['upper_windows_compression'][-1])<1e-14
 pairs={'N32_T4ms':(m32,s32,p32['shell_matched_loading_peak_N']),'N64_T1ms':(m64,s64,p64['matched_peak_N'])}
 metrics=[]
 for cap in p['upper_windows_compression']:
  row={'upper_compression':cap}
  for name,(a,b,den) in pairs.items():
   g=np.linspace(0,cap,p['grid_points']);rms=float(np.sqrt(np.mean((np.interp(g,a['compression'],a['force'])-np.interp(g,b['compression'],b['force']))**2)))
   row[name]={'RMS_N':rms,'RMS_over_matched_shell_loading_peak':rms/den,'RMS_over_legacy_fixed_peak':rms/p32['old_fixed_curve_denominator_N'],'model_loading_work_N_mm':work(a,cap),'matched_shell_loading_work_N_mm':work(b,cap),'covered_work_relative_difference':work(a,cap)/work(b,cap)-1}
  metrics.append(row)
 fixed=[]
 for c in [.01,.05,.10,.12,.13,upper]:
  row={'compression':c}
  for name,(a,b,den) in pairs.items():
   f=at(a,c);ref=at(b,c);fi=-at(a,c,'internal_macro_Fz_N')
   row[name]={'force_N':f,'matched_shell_force_N':ref,'same_compression_force_relative_difference':f/ref-1,'internal_macro_force_on_dynamic_state_N':fi,'instantaneous_inertia_contribution_N':f-fi,'KE_over_U':at(a,c,'KE_over_U')}
  row['N64_vs_N32_force_relative_difference']=row['N64_T1ms']['force_N']/row['N32_T4ms']['force_N']-1
  fixed.append(row)
 vf32=p32['binary_volume_fraction'];vf64=load(a45/'preparation.json')['N64_binary_volume_fraction'];original=load(D/'frozen_before.json');intact=all(sha(R/n)==h for n,h in original.items())
 result={'action':'read-only; no new solve/AD','common_upper_compression':upper,'RMS_points':p['grid_points'],'window_metrics':metrics,'force_at_fixed_compressions':fixed,'N32_captured_dynamic_loading_peak':c32['observed_binary_peak'],'N64_captured_dynamic_loading_peak':None,'N64_endpoint_maximum':c64['observed_loading_maximum'],'matched_shell_peaks':{'T4ms':shell_peak(s32),'T1ms':shell_peak(s64),'T0p25ms_preliminary':shell_peak(sf)},'N64_peak_unobserved':True,'N32_full20_and_hold':True,'N64_full20_and_hold':False,'binary_volume_fraction':{'N32':vf32,'N64':vf64,'N64_vs_N32_relative_difference':vf64/vf32-1},'force_source':'scripts/thin_target_explicit.py observables: macro reaction includes internal force plus nodal full acceleration. internal_macro_Fz_N is evaluated on the dynamic state, not a statically re-equilibrated response.','mesh_rate_confounded':True,'original_gates_unchanged':True,'old_N32_full_window_metrics_preserved':c32['curve_vs_matched_shell'],'source_SHA256':{str(sh32_path.relative_to(R)):sha(sh32_path),str(sh64_path.relative_to(R)):sha(sh64_path),str(shfast_path.relative_to(R)):sha(shfast_path)},'new_N32_runs':0,'N64_resumed':False,'new_Abaqus_jobs':0,'new_design_AD':0,'original_protected_files_count':len(original),'original_protected_files_unchanged':intact,'production_entry_sha256':sha(R/'scripts/thin_target_explicit.py')}
 assert intact and c64['captured_N64_peak'] is None
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
 fig,ax=plt.subplots(2,2,figsize=(12.8,8))
 for a,b,t,c in [(m32,s32,'N32, T=4 ms','#3377a7'),(m64,s64,'N64, T=1 ms','#ba5332')]:
  z=ax[0,0] if t.startswith('N32') else ax[0,1]
  xa=np.r_[a['compression'][a['compression']<upper],upper];xb=np.r_[b['compression'][b['compression']<upper],upper]
  z.plot(xa*100,np.interp(xa,a['compression'],a['force']),color=c,label='JAX binary')
  z.plot(xb*100,np.interp(xb,b['compression'],b['force']),color='#37414b',ls='--',label='Own matched-rate shell')
  title=t+(': common interval' if t.startswith('N32') else ': peak not captured')
  z.axvline(shell_peak(b)['compression']*100,color='#37414b',ls=':',lw=.8);z.set(title=title,ylabel='Compression reaction (N)',xlim=(0,upper*100),ylim=(0,6.8));z.legend(fontsize=9)
 z=ax[1,0];xx=np.arange(len(metrics));width=.34
 for name,c,off in [('N32_T4ms','#3377a7',-width/2),('N64_T1ms','#ba5332',width/2)]:
  yy=[100*r[name]['RMS_over_matched_shell_loading_peak'] for r in metrics];bars=z.bar(xx+off,yy,width,color=c,label=name.replace('_',' '));z.bar_label(bars,fmt='%.2f',padding=3,fontsize=8)
 z.set_xticks(xx,[f"0-{r['upper_compression']*100:.2f}%" for r in metrics]);z.set(ylabel='RMS / own full shell peak (%)',title='Early difference and reference unloading',ylim=(0,25.5));z.legend(fontsize=8)
 z=ax[1,1]
 for a,t,c in [(m32,'N32, T=4 ms','#3377a7'),(m64,'N64, T=1 ms','#ba5332')]:
  g=np.linspace(.01,upper,600);d=np.array([at(a,q)+at(a,q,'internal_macro_Fz_N') for q in g]);z.plot(g*100,d,color=c,label=t)
 z.axhline(0,color='#777777',lw=.6);z.set(title='Instantaneous inertial contribution to reaction',ylabel='Reaction minus internal macro force (N)',xlim=(1,upper*100));z.legend(fontsize=8)
 for z in ax.ravel():z.set_xlabel('Compression (%)');z.grid(alpha=.2)
 ax[1,0].set_xlabel('Common comparison window')
 fig.tight_layout();fig.savefig(D/'common_window_comparison.png');plt.close(fig)
 result['read_only_analysis_seconds']=time.perf_counter()-start
 (D/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'common_upper_compression':upper,'shared_RMS_N':{k:metrics[-1][k]['RMS_N'] for k in pairs},'N64_peak_unobserved':True,'protected_count':len(original),'intact':intact,'seconds':result['read_only_analysis_seconds']},indent=2))
if __name__=='__main__':main()

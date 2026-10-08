"""Existing-path postprocessing only; no FEM, tangent, new trajectory, AD or ODB."""
from pathlib import Path
import hashlib,json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/xuehu/projects/tpms_jax')
W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
D=R/'validation/initial_stiffness_review_20261008_r29';D.mkdir(exist_ok=False)
WO=W/'output/r29_initial_stiffness_review_20261008';WO.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source={
 'd04_N32_T004':'validation/step_control_20261007_r18/controlled_forward/accepted_path.json',
 'd04_N64_T002':'validation/n64_resolution_20261008_r28/full_from_zero/accepted_path.json',
 'd04_shell_T004':'validation/shell_rate_20261007_r20/abaqus/explicit_T0p004/results/shell.json',
 'd04_shell_T002':'validation/n64_resolution_20261008_r28/abaqus/explicit_T0p002/results/shell.json',
 'd28_N32_T004':'validation/void_continuation_20261006_r12/T0p004/result.json',
 'd28_N32_T008':'validation/void_continuation_20261006_r12/T0p008/result.json',
 'd28_shell_T040':'validation/large_compression_20261005_r6/abaqus/explicit_T0p040/shell.json',
 'd28_shell_Standard':'validation/large_compression_20261005_r6/abaqus/standard/shell.json',
 'historical_linear_XYZ':'validation/thin_target_20261004_r5/step2/diagnostic_xyz/comparison.json',
 'historical_linear_XYZ_input':'validation/thin_target_20261004_r5/step2/diagnostic_xyz/input.json',
 'historical_geometry_diagnostic':'validation/simulation_error_20261006_r7/step1/result.json',
 'r28_comparison':'validation/n64_resolution_20261008_r28/comparison.json',
 'r27_comparison':'validation/plate_bending_20261008_r27/comparison.json'}
inputs={k:json.loads((R/v).read_text()) for k,v in source.items()}
before={v:sha(R/v) for v in source.values()}
protected=json.loads((R/'validation/n64_resolution_20261008_r28/frozen_before.json').read_text())
for k,v in protected.items():assert sha(R/k)==v
paths={}
for name,obj in inputs.items():
    if name.startswith('historical') or name in ('r28_comparison','r27_comparison'):continue
    rows=obj if isinstance(obj,list) else obj.get('path',obj.get('force_path'))
    duration={'d04_shell_T004':.004,'d04_shell_T002':.002,'d28_shell_T040':.04,'d28_N32_T004':.004,'d28_N32_T008':.008}.get(name)
    if duration:rows=[r for r in rows if r['time']<=duration+1e-12]
    a=np.array([r['compression'] for r in rows]);ix=np.unique(a,return_index=True)[1];ix=ix[np.argsort(a[ix])]
    paths[name]=[rows[int(i)] for i in ix]
def values(name,a,key='Fz_N'):
    rows=paths[name]
    return np.interp(a,[r['compression'] for r in rows],[-r[key] for r in rows])
pairs={'d04_N32_T004':'d04_shell_T004','d04_N64_T002':'d04_shell_T002',
 'd28_N32_T004':'d28_shell_Standard','d28_N32_T008':'d28_shell_Standard'}
points=[];fit=[]
for case,ref in pairs.items():
    for a in (.002,.005,.01,.05,.10):
        fj,fi,fs=[float(z) for z in (values(case,a),values(case,a,'internal_macro_Fz_N'),values(ref,a))]
        points.append({'case':case,'reference':ref,'compression':a,'dynamic_force_N':fj,'internal_force_N':fi,
          'shell_force_N':fs,'dynamic_excess_over_shell':fj/fs-1,'internal_excess_over_shell':fi/fs-1,
          'explicit_inertia_fraction_of_dynamic_force':(fj-fi)/fj,
          'scope':'Internal force is evaluated in the saved dynamic state, not a static equilibrium force.'})
    for lo,hi in ((.002,.01),(.01,.05),(.05,.10)):
        a=np.linspace(lo,hi,1001);u=10*a
        for key in ('Fz_N','internal_macro_Fz_N'):
            f=values(case,a,key);s=values(ref,a)
            coef=np.polyfit(u,f,1);rcoef=np.polyfit(u,s,1)
            fit.append({'case':case,'reference':ref,'compression_window':[lo,hi],'force_key':key,
             'slope_N_per_mm':float(coef[0]),'intercept_N':float(coef[1]),
             'shell_slope_N_per_mm':float(rcoef[0]),'shell_intercept_N':float(rcoef[1]),
             'slope_relative_excess':float(coef[0]/rcoef[0]-1),
             'fit_RMS_N':float(np.sqrt(np.mean((f-np.polyval(coef,u))**2))),
             'scope':'Descriptive free-intercept least-squares fit of existing dynamic path; not K0 at zero strain or a new acceptance gate.'})
overview={'stage':'read_only_existing_initial_force_paths','points':points,'window_fits':fit,
 'old_static_linear_XYZ':inputs['historical_linear_XYZ'],
 'old_static_scope':'diverse_28 historical N64 HEX8 static linear XYZ, not current N32 HEX27 certification',
 'd28_references':'Fits use existing shell Standard loading curve as a static reference. JAX .004/.008s and shell .04s are different time histories; no matched-rate claim.',
 'old_shell_dynamic_crosscheck_at_points':[{'compression':a,'slow_Explicit_force_N':float(values('d28_shell_T040',a)),
   'Standard_force_N':float(values('d28_shell_Standard',a))} for a in (.002,.005,.01,.05,.1)],
 'current_N32_T002_not_run':True,'N64_T002_vs_N32_T004_not_pure_mesh_cause_test':True,
 'new_forward_jobs':0,'new_Abaqus_jobs':0,'new_tangent_or_AD':False,
 'input_sha256':before,'protected_sha256':protected,
 'prior_dynamic_gates_and_failures_unchanged':True}
(D/'analysis.json').write_text(json.dumps(overview,indent=2,ensure_ascii=False)+'\n')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(11.5,4.8))
a=np.linspace(.002,.1,1001)
for name,col,style,label in (('d04_shell_T004','#8a9aa7','--','Shell 0.004 s'),('d04_shell_T002','#286494','-','Shell 0.002 s'),('d04_N32_T004','#a87755','--','JAX N32, 0.004 s'),('d04_N64_T002','#a93339','-','JAX N64, 0.002 s')):
    axs[0].plot(a*100,values(name,a),color=col,ls=style,lw=2,label=label)
axs[0].set(xlabel='Macroscopic compression (%)',ylabel='Compressive force (N)',title='diverse_04: early force paths');axs[0].legend(frameon=False,fontsize=9);axs[0].grid(alpha=.15)
a=np.linspace(.01,.1,1001)
for case,col,label in (('d04_N32_T004','#a87755','N32 / own 0.004 s shell'),('d04_N64_T002','#a93339','N64 / own 0.002 s shell')):
    ref=pairs[case];fs=values(ref,a)
    axs[1].plot(a*100,100*(values(case,a)/fs-1),color=col,lw=2,label=label+' (dynamic)')
    axs[1].plot(a*100,100*(values(case,a,'internal_macro_Fz_N')/fs-1),color=col,ls='--',lw=1.5,label=label+' (internal)')
axs[1].set(xlabel='Macroscopic compression (%)',ylabel='Force excess over shell (%)',title='Internal-force bias remains after the direct inertia term');axs[1].legend(frameon=False,fontsize=8);axs[1].grid(alpha=.15)
fig.text(.08,.018,'Existing saved paths only. Internal force still belongs to a dynamic state; N32 and N64 have different load times.',fontsize=9,color='#444444')
fig.tight_layout(rect=(0,.055,1,1));fig.savefig(D/'initial_response.png',dpi=180);plt.close(fig)
shutil.copy2(Path(__file__),D/'read_existing_paths.py')
for n in ('analysis.json','initial_response.png'):shutil.copy2(D/n,WO/n)
assert all(sha(R/k)==v for k,v in before.items()) and all(sha(R/k)==v for k,v in protected.items())
print(json.dumps({'points_1_5_10pct':[r for r in points if r['compression'] in (.01,.05,.10)],
 'fits_0p2_to_1pct':[r for r in fit if r['compression_window']==[.002,.01]],
 'fits_1_to_5pct':[r for r in fit if r['compression_window']==[.01,.05]],
 'no_new_solver':True},indent=2))

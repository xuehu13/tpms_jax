"""Classify actual stored evidence. Never infer precision from a pilot."""
from pathlib import Path
import json,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path('/home/xuehu/projects/tpms_jax');O=P/'validation/mechanics_trust_20261003_r4/step4';HERE=Path(__file__).resolve().parent
ledger=json.loads((O/'execution.json').read_text());cases=[]
for entry in ledger:
    tag=entry['tag'];full=O/(tag+'.json');progress=O/(tag+'.progress.json')
    v=json.loads(full.read_text()) if full.exists() else {'status':entry['status'],'rows':json.loads(progress.read_text()) if progress.exists() else []}
    rows=v['rows'];last=rows[-1] if rows else None
    profile=(O/(tag+'.log')).read_text(errors='replace')
    newton_times=[{'count':int(k),'wall_seconds':float(w),'linear_seconds':float(t),'linear_percent':float(percent)}
        for k,w,t,percent in re.findall(r'Timing summary .*? (\d+) Newton iter, ([\d.]+) s wall[\s\S]*?jax_fem:\s+linear\s+([\d.]+) s\s+([\d.]+)%',profile)]
    if rows and 'work' not in v:
        a=np.array([r['compression'] for r in rows]);force=np.array([r['Fz_top'] for r in rows]);stored=rows[-1]['energy']
        work=float(np.sum(-.5*(force[1:]+force[:-1])*np.diff(a)))
        v['work']={'path_trapezoid':work,'stored_energy':stored,'relative_gap':abs(work-stored)/stored if stored else 0.,
                   'uses_half_Fu':False,'checked_interval_end':float(a[-1]),'partial_path_only':True}
    case={'tag':tag,'N':entry['N'],'status':entry['status'],'kind':entry['kind'],'stored_states':len(rows),
        'last_completed_compression':last['compression'] if last else None,'record':entry,
        'last_response':{k:last[k] for k in ('Fz_top','energy','J_min','detF_by_occupancy','pure_void_energy_fraction','energy_uniform_floor','energy_projected_material')} if last else None,
        'maximum_completed_state_seconds':max((r['solve_seconds'] for r in rows),default=0.),
        'maximum_reduced_residual_l2':max((r['reduced_residual_l2'] for r in rows),default=0.),
        'maximum_balance_error':max((abs(r['Fz_top']+r['Fz_bottom']) for r in rows),default=0.),
        'maximum_reaction_Piola_error':max((abs(r['Fz_top']-r['mean_first_piola'][2][2]) for r in rows),default=0.),
        'maximum_void_energy_fraction':max((r['pure_void_energy_fraction'] for r in rows),default=0.),
        'all_completed_checks_pass':all(all(r['checks'].values()) for r in rows),'work':v.get('work'),
        'runtime':v.get('runtime'),'Newton_profile':newton_times,'failure':v.get('failure')}
    cases.append(case)
    entry['saved_equilibrium_states']=len(rows)
    if last:entry['last_completed_compression']=last['compression']
    current=O/(tag+'.current.json')
    if current.exists():entry['last_attempted_compression']=json.loads(current.read_text())['compression']
classification='step4_resource_boundary' if any(r['resource_stop'] for r in ledger) else 'step4_requires_precision_and_reference'
if any(v['failure'] for v in cases):classification='step4_numerical_or_physical_boundary'
summary={'stage':classification,'cases':cases,'budget_counts':{'JAX_paths_attempted':len(ledger),'saved_equilibrium_states':sum(r['saved_equilibrium_states'] for r in ledger),
    'preflight_paths':sum(r['kind']=='JAX_resource_preflight' for r in ledger),'failure_diagnostic_paths':sum('diagnostic' in r['kind'] for r in ledger),
    'precision_paths_attempted':sum(r['kind']=='JAX_precision_path' for r in ledger),'Abaqus_analyses':0,'datachecks':0,'training':0,'design_gradient':0},
    'TPMS_finite_strain_verified_interval':None,'important':'A successful coarse continuation is implementation/feasibility evidence only; no independent nonlinear reference or mesh convergence yet',
    'research_decision':'Identify and resolve measured numerical/resource bottleneck in a separately bounded next stage before extending compression or training'}
(O/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(O/'execution.json').write_text(json.dumps(ledger,indent=2)+'\n')
fig,axes=plt.subplots(1,3,figsize=(11.6,3.5),layout='constrained')
colors={16:'#7692a2',32:'#287c8e',48:'#c7743b',64:'#843b79'}
for case in cases:
    tag=case['tag'];f=O/(tag+'.json');rows=json.loads(f.read_text())['rows'] if f.exists() else json.loads((O/(tag+'.progress.json')).read_text())
    a=np.array([r['compression'] for r in rows]);n=case['N'];label=f'N{n}'+(' half increment' if 'half' in tag else '')
    axes[0].plot(a*100,[-r['Fz_top'] for r in rows],marker='o',ms=3,color=colors[n],label=label)
    axes[1].plot(a*100,[r['detF_by_occupancy']['void']['min'] for r in rows],color=colors[n],label=label+' void')
    axes[1].plot(a*100,[r['detF_by_occupancy']['solid']['min'] for r in rows],'--',color=colors[n],label=label+' solid')
    nonzero=[r for r in rows if r['compression']>0]
    axes[2].plot([r['compression']*100 for r in nonzero],[r['solve_seconds'] for r in nonzero],marker='o',ms=3,color=colors[n],label=label)
    if case['record']['resource_stop']=='state_timeout':
        axes[2].plot(case['record'].get('last_attempted_compression',a[-1])*100,300,'x',ms=8,color=colors[n],label=label+' timed out')
axes[0].set(xlabel='Compression (%)',ylabel='Compression force magnitude',title='Stored equilibria; precision pending')
axes[1].axhline(.1,ls=':',color='#888');axes[1].set(xlabel='Compression (%)',ylabel='Minimum local det(F)',title='Solid: rho >= .95; void: rho <= .05')
axes[2].axhline(300,ls=':',color='#888');axes[2].set(xlabel='Compression (%)',ylabel='Seconds per completed state',title='Locked state limit: 300 seconds')
for ax in axes:ax.legend(fontsize=7);ax.spines[['top','right']].set_visible(False)
fig.savefig(O/'gyroid_preflight.png',dpi=180);fig.savefig(HERE/'gyroid_preflight.png',dpi=180);plt.close(fig)
(O/'analyze.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(summary,indent=2))

from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path(__file__).resolve().parent;R=D.parent.parent
read=lambda p:json.loads(p.read_text())
p=read(D/'protocol.json');selection=read(D/'selection.json');fixed=read(D/'fixed_check.json')
s0=read(R/p['smooth_baseline']/'jax_result.json');b0=read(D/'original27/jax_result.json')
s1=read(D/'smooth_mixed12/result.json');b1=read(D/'binary_mixed12/result.json');a=read(D/'energy_accounting/result.json')
assert all(x['status']=='ok' for x in [fixed,s0,b0,s1,b1,a])
assert all(all(x['checks'].values()) for x in [fixed,s0,b0,s1,b1])
rows=[]
for rule,ss,bb in [('original27',s0,b0),('own_selected12_rest27',s1,b1)]:
 rows.append({'rule':rule,'smooth_K_N_per_mm':ss['stiffness_N_per_mm'],'binary_K_N_per_mm':bb['stiffness_N_per_mm'],
  'smooth_relative_to_shell':ss['stiffness_N_per_mm']/p['shell_stiffness_N_per_mm']-1,
  'binary_relative_to_shell':bb['stiffness_N_per_mm']/p['shell_stiffness_N_per_mm']-1,
  'binary_relative_to_same_rule_smooth':bb['stiffness_N_per_mm']/ss['stiffness_N_per_mm']-1})
recorded_costs={'selection':selection['wall_seconds'],'binary27':b0['wall_seconds'],'fixed_check':fixed['wall_seconds'],
 'smooth_mixed12':s1['wall_seconds'],'binary_mixed12':b1['wall_seconds'],'energy_accounting':a['wall_seconds']}
out={'status':'accepted_initial_partial_scope','rows':rows,'selected_cells':selection['selected_cells'],
 'original_smooth_energy_fraction_selected':selection['selected_fraction_full_energy'],
 'max_abs_8_12_relative':max(abs(v) for k,v in fixed['relative_8_over_12_minus_one'].items() if 'floor' not in k),
 'energy_accounting':a['pairs'],'recorded_successful_stage_seconds':recorded_costs,
 'recorded_successful_stage_total_seconds':sum(recorded_costs.values()),
 'cost_note':'stage-internal timings; excludes preparation scripting, input-adapter repairs, process startup, some file IO and documentation time',
 'new_equilibrium_solves':3,'new_Abaqus_jobs':0,'production_changed':False,
 'all_initial_static_and_frozen_local_integration_gates_passed':True,'outside_selection_densely_verified':False,
 'sharp_interface_convergence_certified':False,'complete20pct_or_design_AD_certified':False}
(D/'comparison.json').write_text(json.dumps(out,indent=2)+'\n')
fig,axes=plt.subplots(1,2,figsize=(10,4.2));x=np.arange(2);width=.32
for offset,key,label,color in [(-.5,'smooth_K_N_per_mm','Smooth','#527d9b'),(.5,'binary_K_N_per_mm','Binary','#548473')]:
 values=[z[key] for z in rows];bars=axes[0].bar(x+offset*width,values,width,label=label,color=color)
 axes[0].bar_label(bars,labels=[f'{v:.4f}' for v in values],padding=3,fontsize=9)
axes[0].axhline(p['shell_stiffness_N_per_mm'],color='#8a5b42',linestyle='--',label='Matched shell')
axes[0].set_ylim(3.5,4.08);axes[0].set_ylabel('Initial K (N/mm); truncated axis')
axes[0].set_xticks(x,['27 points/cell','Selected 12^3 / rest 27']);axes[0].legend(frameon=False,fontsize=9)
axes[0].set_title('diverse_28: same-rule comparison',fontsize=11)
for offset,key,label,color in [(-.5,'smooth_relative_to_shell','Smooth','#527d9b'),(.5,'binary_relative_to_shell','Binary','#548473')]:
 values=[100*z[key] for z in rows];bars=axes[1].bar(x+offset*width,values,width,label=label,color=color)
 axes[1].bar_label(bars,labels=[f'{v:+.2f}%' for v in values],padding=4,fontsize=9)
axes[1].axhline(0,color='#555',linewidth=.7);axes[1].set_ylim(-3,4.6)
axes[1].set_ylabel('Difference / fixed matched-shell K (%)');axes[1].set_xticks(x,['27 points/cell','Selected 12^3 / rest 27'])
axes[1].set_title('Quadrature changes the binary discrepancy sign',fontsize=11)
for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.tight_layout();fig.savefig(D/'initial_occupancy_comparison.png',dpi=170,bbox_inches='tight')
print(json.dumps({'rows':rows,'recorded_stage_seconds':sum(recorded_costs.values())},indent=2))

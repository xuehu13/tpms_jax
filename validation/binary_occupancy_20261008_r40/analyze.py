"""Summarize only completed initial experiments; no new mechanics."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/xuehu/projects/tpms_jax')
B=R/'validation/binary_occupancy_20261008_r40'
T=R/'validation/initial_diverse28_20261008_r41'
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
old=read(R/'validation/initial_tangent_20261008_r30/jax_result.json')
oldmixed=read(R/'validation/local_reequilibrium_20261008_r34/result.json')
hard=read(B/'original27/jax_result.json');hardmixed=read(B/'mixed12/result.json')
post=read(B/'fixed_check.json');k04=read(R/'validation/initial_tangent_20261008_r30/abaqus/shell_result.json')['stiffness_N_per_mm']
current=read(T/'jax_result.json');sh=read(T/'abaqus/shell_result.json')
assert all(v['status']=='ok' for v in [old,oldmixed,hard,hardmixed,current,sh])
assert all(all(v['checks'].values()) for v in [hard,hardmixed,current,sh])
for directory in [B,T]:
    frozen=read(directory/'frozen_before.json')
    assert all(hashlib.sha256((R/name).read_bytes()).hexdigest()==h for name,h in frozen.items())
pairs=[]
for name,smooth,binary in [('original27',old,hard),('selected12_rest27',oldmixed,hardmixed)]:
    a=smooth['stiffness_N_per_mm'];b=binary['stiffness_N_per_mm']
    pairs.append({'rule':name,'smooth_K_N_per_mm':a,'binary_K_N_per_mm':b,
        'same_rule_binary_over_smooth_minus_one':b/a-1,'smooth_over_shell_minus_one':a/k04-1,
        'binary_over_shell_minus_one':b/k04-1,'shell_K_N_per_mm':k04})
out={'status':'completed_bounded_initial_diagnostic','case':'diverse_04','pairs':pairs,
    'binary_mixed_over_binary27_minus_one':hardmixed['stiffness_N_per_mm']/hard['stiffness_N_per_mm']-1,
    'binary_fixed_integration':post,'smooth_original_weighted_volume_mm3':old['weighted_volume_mm3'],
    'binary_original_weighted_volume_mm3':hard['weighted_volume_mm3'],
    'interpretation':'Occupancy treatment affects discrete initial K, but original27 apparent closeness is strongly quadrature-sensitive; partial dense rule leaves bias and is not full-domain sharp-interface certification.',
    'binary_production_adopted':False,'full5pct_or20pct_or_geometry_AD_certified':False,
    'cost_seconds':{'binary27':hard['wall_seconds'],'fixed_check':post['wall_seconds'],'mixed12':hardmixed['wall_seconds']}}
write(B/'comparison.json',out)
shell_summary={k:v for k,v in sh.items() if k!='nodes'}
write(T/'abaqus/shell_summary.json',shell_summary)
out28={'status':'ok','case':'diverse_28','N':32,'compression':1e-4,
    'background_K_N_per_mm':current['stiffness_N_per_mm'],'shell_K_N_per_mm':sh['stiffness_N_per_mm'],
    'background_over_shell_minus_one':current['stiffness_N_per_mm']/sh['stiffness_N_per_mm']-1,
    'historical_initial_shell_reproduced_relative':sh['stiffness_N_per_mm']/read(R/'validation/thin_target_20261004_r5/step2/diagnostic_xyz/shell.json')['stiffness_N_per_mm']-1,
    'two_cases_have_same_bias_sign_not_proof_of_same_cause':True,'diverse04_current_bias':old['stiffness_N_per_mm']/k04-1,
    'full_path_or_design_AD_certified':False,'new_native_job':True,'production_changed':False,
    'cost_seconds':{'background':current['wall_seconds'],**read(T/'abaqus/receipt.json')}}
write(T/'comparison.json',out28)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,3,figsize=(14,4.8))
fig.subplots_adjust(left=.065,right=.98,bottom=.23,top=.81,wspace=.35)
colors=['#315b80','#ab7544'];xx=np.arange(2)
for i,key in enumerate(['smooth_over_shell_minus_one','binary_over_shell_minus_one']):
    vals=np.array([r[key]*100 for r in pairs]);pos=xx+(i-.5)*.32
    axs[0].bar(pos,vals,.32,color=colors[i],label=['Smooth .05 mm','Binary GP'][i])
    for x,y in zip(pos,vals):axs[0].text(x,y+.17,f'{y:.2f}%',ha='center',fontsize=9)
axs[0].set_xticks(xx,['27 points/cell','Selected 12^3\nrest 27']);axs[0].set_ylim(0,8)
axs[0].set_ylabel('Initial K relative to matched shell (%)');axs[0].legend(fontsize=8,frameon=False)
axs[0].set_title('A  diverse_04: occupancy affects K')
ks=np.array([[k04,old['stiffness_N_per_mm']],[sh['stiffness_N_per_mm'],current['stiffness_N_per_mm']]])
for i in range(2):
    pos=xx+(i-.5)*.32;axs[1].bar(pos,ks[:,i],.32,color=['#8d99a4','#315b80'][i],label=['S3R Standard','Current smooth N32'][i])
    for x,y in zip(pos,ks[:,i]):axs[1].text(x,y+.08,f'{y:.3f}',ha='center',fontsize=8)
axs[1].set_xticks(xx,['diverse_04','diverse_28']);axs[1].set_ylim(0,6.1);axs[1].set_ylabel('Initial K (N/mm)')
axs[1].text(0,float(ks[0].max())+.5,'+5.70%',ha='center');axs[1].text(1,float(ks[1].max())+.5,f'+{out28["background_over_shell_minus_one"]:.2%}',ha='center')
axs[1].legend(fontsize=8,frameon=False,loc='upper right');axs[1].set_title('B  Static bias depends on geometry')
d=np.linspace(.16,.34,1000);phi=1/(1+np.exp((d-.25)/(.05/(2*np.log(9)))))
axs[2].plot(d,phi,color=colors[0],label='Smooth .05 mm')
axs[2].step([.16,.25,.34],[1,0,0],where='post',color=colors[1],label='Binary')
axs[2].axvline(.25,color='#bbbbbb',ls=':',lw=1);axs[2].set_xlabel('Distance to midsurface (mm)')
axs[2].set_ylabel('Occupancy phi');axs[2].set_ylim(-.05,1.15);axs[2].legend(fontsize=8,frameon=False)
axs[2].set_title('C  Remove transition, retain eta=1e-4')
fig.suptitle('Binary test: partial confirmation lowers K by 2.12%, but a 4.24% bias remains',fontsize=13,y=.95)
fig.text(.065,.105,'Initial static tangent only. Dense quadrature covers the original 1,941 cells; other cells remain at 27 points.',fontsize=10)
fig.text(.065,.06,'Binary is a causal diagnostic of the discrete model, not an adopted geometry-gradient method or a 20% validation.',fontsize=10)
fig.savefig(B/'binary_initial_comparison.png',dpi=180);plt.close(fig)
print(json.dumps({'binary_comparison':pairs,'diverse28':out28},indent=2))

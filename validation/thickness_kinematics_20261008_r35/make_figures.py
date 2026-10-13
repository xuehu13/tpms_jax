"""Plot saved r35 observations. No new field evaluation or solve."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path('/home/xuehu/projects/tpms_jax/validation/thickness_kinematics_20261008_r35')
r=json.loads((D/'result.json').read_text());assert r['status']=='ok'
z=np.array(r['line_geometry']['signed_z_mm']);a=1e-4
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
cols={'r30':'#1e526c','r34':'#687e60'}
specs=[('normal_vs_ps_prediction_fit',1.,'Normal strain / ideal plane-stress prediction','Weighted fitted amplitude'),
       ('normal_mismatch_rms',a,'Normal strain mismatch through thickness','RMS mismatch / macro strain'),
       ('shear_strain_vector_rms',a,'Transverse shear through thickness','RMS tensor shear / macro strain')]
for axis,(key,scale,title,ylabel) in zip(ax,specs):
    axis.axvspan(-.225,.225,color='#f0f2f3',zorder=0)
    for boundary in [-.25,.25]:axis.axvline(boundary,color='#777777',linestyle=':',linewidth=.8)
    for label in ['r30','r34']:axis.plot(z,np.array(r['line_profiles'][label][key])/scale,color=cols[label],label=f'{label} saved state',linewidth=1.8)
    axis.set_xlabel('Signed normal distance from midsurface (mm)');axis.set_ylabel(ylabel)
    axis.set_title(title,fontsize=10);axis.grid(axis='y',alpha=.15)
ax[0].axhline(1.,color='#aa7d5c',linewidth=.8,linestyle='--');ax[0].legend(fontsize=9,loc='lower center')
fig.suptitle('Wall-interior mismatch persists after the integration-only intervention',fontsize=12)
fig.text(.5,-.05,'Frozen 3,122 faces; area-weighted probes of existing displacements. Ideal plane stress is a diagnostic prediction, not a new shell result.',ha='center',fontsize=9)
fig.savefig(D/'through_thickness_profiles.png',dpi=180,bbox_inches='tight');plt.close(fig)

fig,ax=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
for j,label in enumerate(['r30','r34']):
    s=r['states'][label];parts=s['partitions'];x=np.arange(3)+(j-.5)*.3
    groups=[[parts[-1]],[parts[2]],parts[:2]]
    norm=np.array([sum(v['normal_fraction_selected_total'] for v in group) for group in groups])*100
    shear=np.array([sum(v['shear_fraction_selected_total'] for v in group) for group in groups])*100
    ax[0].bar(x,norm,width=.27,color=cols[label],label=label+' normal mismatch')
    ax[0].bar(x,shear,width=.27,bottom=norm,color=cols[label],alpha=.42,hatch='//',label=label+' transverse shear')
ax[0].set_xticks(np.arange(3),['Solid core (phi >= .9)','Transition (.01-.9)','Soft region (< .01)'])
ax[0].set_ylabel('Fraction of same selected-cell energy (%)');ax[0].set_title('Residual energy is mainly inside the solid core',fontsize=10)
ax[0].legend(fontsize=8,loc='upper right');ax[0].grid(axis='y',alpha=.15)
for j,label in enumerate(['r30','r34']):
    s=r['states'][label];windows=s['normal_line_windows'];vals=[windows[0]['normal_vs_ps_prediction_least_squares_factor'],windows[2]['normal_vs_ps_prediction_least_squares_factor']]
    x=np.arange(2)+(j-.5)*.32;ax[1].bar(x,vals,width=.29,color=cols[label],label=label)
    for xx,y in zip(x,vals):ax[1].text(xx,y+.02,f'{y:.3f}',ha='center',fontsize=9)
ax[1].set_xticks(np.arange(2),['Midwall |z| < .1 mm','Near interface .2-.275 mm']);ax[1].set_ylim(0,1.12)
ax[1].axhline(1,color='#aa7d5c',linewidth=.8,linestyle='--');ax[1].set_ylabel('Normal strain / ideal prediction: fitted amplitude')
ax[1].set_title('Interface improves; wall interior remains similar',fontsize=10);ax[1].legend(loc='lower right',fontsize=9)
fig.suptitle('Same original Gauss rule for the two states; energy splits are diagnostic',fontsize=12)
fig.text(.5,-.05,'Pointwise energy release is not a compatible deformation or a causal decomposition of the 5.70% stiffness bias.',ha='center',fontsize=9)
fig.savefig(D/'partition_and_normal_fit.png',dpi=180,bbox_inches='tight');plt.close(fig)

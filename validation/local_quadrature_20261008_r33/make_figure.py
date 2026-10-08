"""Plot the completed fixed-state integration diagnostic; no FEM computation."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D=Path('/home/xuehu/projects/tpms_jax/validation/local_quadrature_20261008_r33')
r=json.loads((D/'result.json').read_text())
keys=['total_energy_N_mm','occupancy_volume_mm3','occupancy_distance_moment_mm5']
labels=['Strain energy','Occupied volume','Distance moment']
change=np.array([r['comparison'][k]['dense12_over_original_minus_one'] for k in keys])*100
stable=np.array([r['comparison'][k]['dense8_over_dense12_minus_one'] for k in keys])*100
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
for ax,data,title in [(axs[0],change,'Dense integration vs original 27 points'),
                       (axs[1],stable,'8 vs 12 points per axis: local agreement')]:
    colors=['#1d526c','#6d8893','#8b979e']
    ax.bar(np.arange(3),data,width=.6,color=colors)
    ax.axhline(0,color='#555555',linewidth=.7)
    ax.set_xticks(np.arange(3),labels)
    ax.set_ylabel('Relative difference (%)')
    ax.set_title(title,fontsize=11)
    span=max(abs(data).max(),.001)
    ax.set_ylim(min(0,data.min())-.20*span,max(0,data.max())+.35*span)
    for i,v in enumerate(data):
        ax.text(i,v+(.055*span if v>=0 else -.055*span),f'{v:+.4f}%',ha='center',
                va='bottom' if v>=0 else 'top',fontsize=9)
fig.suptitle('diverse_04 N32 | Same saved displacement, same material and geometry',fontsize=12)
fig.text(.5,-.035,'1,941 frozen cells; 64.62% of original strain energy. No new equilibrium or stiffness result.',
         ha='center',fontsize=9,color='#444444')
fig.savefig(D/'fixed_state_quadrature.png',dpi=190,bbox_inches='tight')
plt.close(fig)

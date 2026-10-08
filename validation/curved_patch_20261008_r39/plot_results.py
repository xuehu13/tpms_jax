"""Render one curved-patch result figure; no new mechanics."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
D=Path(__file__).resolve().parent
a=json.loads((D/'analytic.json').read_text())
s=json.loads((D/'abaqus/solid_result.json').read_text())
h=json.loads((D/'abaqus/shell_result.json').read_text())
b=json.loads((D/'background_result.json').read_text())
plt.rcParams.update({'font.size':10.5,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(11.7,5.0),layout='constrained')
fig.get_layout_engine().set(rect=(0,.12,1,.88))
ax=axs[0]
for v in np.arange(0,10.01,.3125):
    ax.axhline(v,lw=.45,alpha=.15,color='gray');ax.axvline(v,lw=.45,alpha=.15,color='gray')
for r,color,ls,label in [(2.25,'#24557a','-','Inner free surface'),(2.5,'#ae5933','--','Same loaded mid-surface'),(2.75,'#24557a','-','Outer free surface')]:
    ax.add_patch(Circle((5,5),r,fill=False,color=color,ls=ls,lw=1.5,label=label))
for th in np.linspace(0,2*np.pi,9)[:-1]:
    n=np.array([np.cos(th),np.sin(th)])
    ax.annotate('',xy=5+3.05*n,xytext=5+2.5*n,arrowprops={'arrowstyle':'->','color':'#ae5933','lw':1.3})
ax.set(xlim=(1.6,8.4),ylim=(1.6,8.4),aspect='equal',xlabel='x (mm)',ylabel='y (mm)',title='One controlled cylinder: R = 2.5 mm, t = 0.5 mm')
ax.legend(loc='lower center',fontsize=8.5,framealpha=.9)
ax=axs[1]
labels=['3D analytic','Abaqus CAX8','Abaqus S3R','Background*']
values=[a['solid_K_N_per_mm'],s['K_N_per_mm'],h['K_N_per_mm'],b['K_N_per_mm']]
colors=['#444444','#39785c','#24557a','#ae5933']
for i,(v,c) in enumerate(zip(values,colors)):
    ax.scatter(v,i,color=c,s=65);ax.text(v+.012,i,f'{v:.6f}',va='center',fontsize=10)
ax.axvline(s['K_N_per_mm'],color='#39785c',ls=':',lw=1)
ax.set(yticks=range(4),yticklabels=labels,xlabel='Generalized radial stiffness K (N/mm)',xlim=(4.1,4.52),ylim=(3.7,-.7),title='Solid vs shell: +0.6209%; background vs solid: -3.1837%')
ax.grid(axis='x',alpha=.2)
fig.suptitle('r39: same mid-surface work; an exploratory patch, not the TPMS compression stiffness',fontsize=12)
fig.text(.53,.018,'* Original basis-cancellation gate failed; saved field audit found roundoff only.\nOriginal not_accepted status retained; no new solve or relaxed gate.',fontsize=8.5,color='#ae5933')
fig.savefig(D/'curved_patch_comparison.png',dpi=190,bbox_inches='tight')
plt.close(fig)

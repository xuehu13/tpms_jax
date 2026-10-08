"""Actual saved states: accepted JAX and existing shell, no amplification."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
R=Path('/home/xuehu/projects/tpms_jax');O=R/'validation/mechanism_20261007_r17';A=O/'analysis'
rows=json.loads((A/'mode_comparison.json').read_text())
with np.load(O/'shell_frames/reference.npz') as f:xyz=f['xyz_mm'];tri=f['triangles']
fields=[]
for r in rows:
    with np.load(O/'shell_frames'/r['shell_nearest_file']) as f:
        shell=(xyz+f['u_mm'],f['normal_fluctuation_mm'])
    with np.load(A/r['file']) as f:
        jax=(f['jax_current_xyz_mm'],f['jax_normal_fluctuation_mm'])
    fields.append((shell,jax))
limit=max(float(np.max(np.abs(v))) for pair in fields for _,v in pair)
norm=Normalize(-limit,limit);cmap=plt.get_cmap('coolwarm')
fig=plt.figure(figsize=(14,8))
for i,(pair,r) in enumerate(zip(fields,rows)):
    for j,((current,normal),label,a,rms) in enumerate(zip(pair,['Shell','JAX'],
       [r['shell_nearest_compression'],r['compression']],
       [r['shell_nearest_fluctuation_RMS_mm'],r['JAX_fluctuation_RMS_mm']])):
        ax=fig.add_subplot(2,4,j*4+i+1,projection='3d')
        ax.add_collection3d(Poly3DCollection(current[tri],facecolors=cmap(norm(normal[tri].mean(axis=1))),edgecolors='none'))
        ax.set(xlim=(0,10),ylim=(0,10),zlim=(0,10))
        if j:ax.set(xlabel='X (mm)',ylabel='Y (mm)',zlabel='Z (mm)' if i==0 else '')
        ax.set_box_aspect((1,1,1));ax.view_init(24,-58)
        ax.set_title('{}: {:.2f}% compression\nFluctuation RMS {:.3f} mm'.format(label,100*a,rms),fontsize=10)
fig.suptitle('diverse_04: actual saved deformed midsurfaces\nShared color = reference-normal nonaffine displacement; no amplification. Nearest shell frames have different compression.',fontsize=11)
fig.subplots_adjust(left=.015,right=.92,bottom=.09,top=.84,wspace=.12,hspace=.27)
bar=fig.add_axes([.931,.2,.013,.55]);cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=bar)
cb.set_label('Normal fluctuation (mm)',fontsize=9,labelpad=4)
fig.savefig(A/'mode_comparison.png',dpi=170);plt.close(fig)

path=json.loads((O/'original_diagnostic/accepted_path.json').read_text())
shell=json.loads((R/'validation/geometry_transfer_20261006_r15/abaqus/explicit_T0p040/results/shell.json').read_text())['force_path']
fig,axes=plt.subplots(1,2,figsize=(11,4.3))
axes[0].plot([r['compression']*100 for r in shell],[-r['Fz_N'] for r in shell],label='Existing shell (quality gates failed)')
axes[0].plot([r['compression']*100 for r in path],[-r['Fz_N'] for r in path],label='JAX accepted blocks',marker='.',ms=3)
axes[0].set(xlabel='Compression (%)',ylabel='Compression force (N)',xlim=(0,20));axes[0].legend(fontsize=8)
axes[1].plot([r['shell_nearest_compression']*100 for r in rows],[r['shell_nearest_fluctuation_RMS_mm'] for r in rows],'-o',label='Saved shell fields')
axes[1].plot([r['compression']*100 for r in rows],[r['JAX_fluctuation_RMS_mm'] for r in rows],'-o',label='Accepted JAX fields')
axes[1].set(xlabel='Actual compression (%)',ylabel='Midsurface fluctuation RMS (mm)');axes[1].legend(fontsize=8)
for ax in axes:ax.grid(alpha=.25)
fig.suptitle('Earlier response/mode divergence is separate from the later numerical failure',fontsize=11)
fig.tight_layout();fig.savefig(A/'response_and_modes.png',dpi=170);plt.close(fig)
print(str(A/'mode_comparison.png'));print(str(A/'response_and_modes.png'))

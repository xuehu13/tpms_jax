"""Read-only final rejection diagnosis and plots from stored solutions."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path('/home/xuehu/projects/tpms_jax');O=P/'validation/mechanics_trust_20261003_r4/step2'
W=Path(__file__).resolve().parents[2]
source=(P/'binary_gyroid.py').read_text()
needle="raise ValueError('Cut point is not on an original tetrahedral edge')"
diagnostic='raise ValueError(json.dumps({"origin":origin.tolist(),"permutation":list(permutation),"tetra_G":g.tolist(),"threshold":float(c),"point":point.tolist(),"candidate_cut_points":choices.tolist(),"distance_squared":float(distances[selected]),"tolerance_squared":1e-24}))'
assert source.count(needle)==1
namespace={'json':json,'__name__':'read_only_final_rejection'}
exec(compile(source.replace(needle,diagnostic),'read_only_final_rejection','exec'),namespace)
try:
    namespace['linear_domain'](48,c=.25)
    raise RuntimeError('Final rejection not reproduced')
except ValueError as exc:
    detail=json.loads(str(exc))
detail['source_sha256']=hashlib.sha256((P/'binary_gyroid.py').read_bytes()).hexdigest()
detail['no_new_mesh_or_FEM_solves']=True
(O/'thin_gyroid/reference_final_failure_detail.json').write_text(json.dumps(detail,indent=2)+'\n')
print('FINAL_REJECTION',json.dumps(detail))
row=json.loads((O/'primitive/comparison.json').read_text())
with np.load(O/'primitive/mode_sample.npz') as data:
    pts=data['points'];affine=pts@data['H'].T
    wr=data['reference_u']-affine;wb=data['background_u']-affine
    wr[:,:2]-=wr[:,:2].mean(axis=0);wb[:,:2]-=wb[:,:2].mean(axis=0)
    fig,axes=plt.subplots(1,2,figsize=(9,3.7),layout='constrained')
    values=[row[k] for k in ('K_N48','K_N64','K_G32','K_G48')]
    axes[0].bar(['JAX N48','JAX N64','Abaqus G32','Abaqus G48'],values,color=['#2675aa','#2675aa','#dd9259','#dd9259'])
    axes[0].set_ylabel('Axial stiffness K = |Fz| / |uz|')
    axes[0].set_ylim(0,2.1);axes[0].tick_params(axis='x',labelrotation=20)
    axes[0].set_title('Primitive |P| <= 0.60; linear elasticity')
    axes[0].text(.03,.95,f'N64 vs G48: {100*row["model_difference"]:.3f}%',transform=axes[0].transAxes,va='top')
    # Deterministic uniform subset of the already stored 4096 solid-node sample.
    sample=np.arange(0,len(pts),4)
    colors=['#2675aa','#dd9259','#6b8c4b']
    for i in range(3):axes[1].scatter(wr[sample,i]/.01,wb[sample,i]/.01,s=5,alpha=.4,label='xyz'[i],color=colors[i])
    limit=1.05*max(np.abs(wr).max(),np.abs(wb).max())/.01
    axes[1].plot([-limit,limit],[-limit,limit],color='#333333',lw=1)
    axes[1].set(xlabel='Abaqus non-affine displacement / |uz|',ylabel='JAX non-affine displacement / |uz|',xlim=(-limit,limit),ylim=(-limit,limit))
    axes[1].set_aspect('equal');axes[1].legend(title='Component',fontsize=8)
    axes[1].set_title('Stored solid-node displacement comparison')
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.savefig(O/'primitive_response.png',dpi=180)
    fig.savefig(Path(__file__).with_name('primitive_response.png'),dpi=180)
    plt.close(fig)
(O/'analyze.py').write_bytes(Path(__file__).read_bytes())
print('FIGURE_SAVED',str(O/'primitive_response.png'))

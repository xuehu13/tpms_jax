"""Additional force-integral check and plot of the completed static intervention."""
from pathlib import Path
import json,ast
import numpy as np
import basix
from scipy.special import expit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
R=Path('/home/xuehu/projects/tpms_jax');sys.path.insert(0,str(R))
from jax_fem.basis import get_elements
from surface_distance import PeriodicSurfaceDistance
D=R/'validation/local_reequilibrium_20261008_r34';Q=R/'validation/local_quadrature_20261008_r33'
r=json.loads((D/'result.json').read_text());assert r['status']=='ok'
# Check the intervention matrix against direct stress integration for deterministic
# random displacements, independently of the saved r30 displacement used at launch.
fam,cell,_,_,deg,order=get_elements('HEX27');el=basix.create_element(fam,cell,deg)
c=np.load(R/'validation/geometry_transfer_20261006_r15/gauss_field.npz')
surf=PeriodicSurfaceDistance(c['surface_vertices'],c['surface_triangles'])
sel=np.load(Q/'selected_cells.npy');sample=sel[np.linspace(0,len(sel)-1,8).astype(int)]
lam=10*.3/(1.3*.4);MU=10/(2*1.3)
tree=ast.parse((D/'diagnostic.py').read_text());fun=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='element_matrix')
ns={'np':np,'lam':lam,'MU':MU};exec(compile(ast.Module(body=[fun],type_ignores=[]),'element_matrix','exec'),ns)
matrix=ns['element_matrix'];errs=[];rng=np.random.default_rng(34)
for level in [3,12]:
    if level==3:
        ref=c['physical_quad_points'][0]*32
        rho=c['rho'][sample];weights=c['JxW'][sample]
    else:
        z,w=np.polynomial.legendre.leggauss(level);z=(z+1)/2;w/=2
        ref=np.array([[a,b,c] for a in z for b in z for c in z])
        weights=np.array([a*b*c for a in w for b in w for c in w])[None,:]/32**3
        origin=c['physical_quad_points'][sample,0,:]-c['physical_quad_points'][0,0,:]
        rho=expit((.25-10*surf.query(origin[:,None,:]+ref[None,:,:]/32))/(.05/(2*np.log(9))))
    g=el.tabulate(1,ref)[1:,:,:,:][:,:,:,0].transpose(1,2,0)[:,order,:]*32
    coef=(1e-4+(1-1e-4)*rho)*weights;ke=matrix(g,coef)
    for trial in range(3):
        u=rng.normal(size=(8,27,3))*1e-4
        grad=np.einsum('cni,qnj->cqij',u,g);eps=.5*(grad+grad.swapaxes(-1,-2))
        sig=lam*np.trace(eps,axis1=-2,axis2=-1)[:,:,None,None]*np.eye(3)+2*MU*eps
        direct=np.einsum('cqij,qnj,cq->cni',sig,g,coef)
        bymatrix=np.einsum('cij,cj->ci',ke,u.reshape(8,81)).reshape(8,27,3)
        errs.append({'level':level,'trial':trial,'force_relative_error':float(np.linalg.norm(bymatrix-direct)/np.linalg.norm(direct))})
out={'scope':'Additional implementation verification after the frozen intervention; no new equilibrium or acceptance relaxation',
 'sample_cells':sample.tolist(),'random_seed':34,'checks':{'matrix_vs_direct_stress_integral_le_1e-12':all(v['force_relative_error']<=1e-12 for v in errs)},'errors':errs}
(D/'additional_operator_validation.json').write_text(json.dumps(out,indent=2)+'\n')
assert all(out['checks'].values()),out
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(7,3.8),layout='constrained')
k=[r['shell_stiffness_N_per_mm'],r['original_stiffness_N_per_mm'],r['stiffness_N_per_mm']]
ax.bar([0,1,2],k,color=['#87939b','#1d526c','#657b60'],width=.6)
ax.set_xticks([0,1,2],['Original shell reference','Original background','Local dense equilibrium'])
ax.set_ylabel('Initial stiffness (N/mm)');ax.set_ylim(0,5.75)
for i,v in enumerate(k):
    txt=f'{v:.4f}'
    if i:txt+=f'\n{100*(v/k[0]-1):+.2f}% vs shell'
    ax.text(i,v+.08,txt,ha='center',va='bottom',fontsize=10)
ax.set_title('One integration-only intervention did not remove the initial stiffness bias',fontsize=11)
fig.text(.5,-.035,'Background intervention only: same displacement space, material, geometry and periodic boundaries',ha='center',fontsize=9)
fig.savefig(D/'initial_stiffness_intervention.png',dpi=190,bbox_inches='tight');plt.close(fig)
print(json.dumps(out,indent=2))

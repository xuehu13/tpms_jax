"""Frozen-state tangent Rayleigh bounds using the shared energy and HRZ mass.

Assemble a restricted tangent only around the largest event displacement. All
cells incident to those periodic DOFs contribute. Its largest eigenvalue is a
lower bound on the full symmetric mass-normalized tangent's largest eigenvalue;
it cannot certify global stability when below the central-difference limit.
"""
from pathlib import Path
import json, os, sys, time
os.environ['JAX_PLATFORMS']='cpu'
R=Path('/home/xuehu/projects/tpms_jax');sys.path.insert(0,str(R))
import numpy as np
import jax
import jax.numpy as jnp
import basix
from jax_fem.basis import get_elements
from hyperelastic_fem import make_density_hyperelastic_problem,objective_void_first_piola
jax.config.update('jax_enable_x64',True)
O=R/'validation/mechanism_20261007_r17';D=O/'original_diagnostic';old=R/'validation/geometry_transfer_20261006_r15'
started=time.perf_counter();N=32;levels=65
p=make_density_hyperelastic_problem(1,rho_quad=1.,eta=1e-4,periodic_axes=(0,1,2),
    element_degree=2,quadrature_order=4,material_model='objective_void')
grads=np.asarray(p.fe.shape_grads[0])*N
family,cell,_,_,degree,order=get_elements('HEX27')
local=np.rint(2*basix.create_element(family,cell,degree).points[order]).astype(int)
origins=2*np.indices((N,)*3).reshape(3,-1).T
indices=origins[:,None,:]+local[None,:,:]
cells=(indices[...,0]*levels+indices[...,1])*levels+indices[...,2]
with np.load(old/'hrz_mass.npz') as f:
    # Cached ex.mass is already rho*L^2 times the normalized-volume integral,
    # exactly the denominator in acceleration(). The stored extra factor L
    # converts to physical nodal mass and does not belong in this equation.
    ids=f['class_ids'];mass=f['periodic_class_mass_normalized']
with np.load(old/'gauss_field.npz') as f:rho=f['rho'];weights=f['JxW']
cid=ids[cells]
with np.load(D/'first_rejection_replay/first_invalid.npz') as f:qbad=f['q']
with np.load(D/'first_rejection_replay/before_first_invalid.npz') as f:qbefore=f['q']
event=int(np.argmax(np.linalg.norm(qbad-qbefore,axis=1)))
eventcells=np.flatnonzero((cid==event).any(axis=1))
# Central cell chosen by actual event, not by shell response. Periodic DOFs
# ensure neighbours across the boundary are included.
central=int(eventcells[0]);target=np.unique(cid[central]);target=target[target!=0]
included=np.flatnonzero(np.isin(cid,target).any(axis=1))
mapping=np.full(len(mass),-1,dtype=int);mapping[target]=np.arange(len(target))
Hbatch=jax.jit(jax.vmap(jax.jacfwd(objective_void_first_piola,argnums=0)))
def compression(t):
    s=np.clip(t/.004,0,1);return .2*(10*s**3-15*s**4+6*s**5)
rows=[]
paths=[None,*sorted(D.glob('accepted_a*.npz')),D/'first_rejection_replay/before_first_invalid.npz']
for path in paths:
    if path is None:q=np.zeros_like(qbefore);t=0.;dt=2.693272938728041e-7;name='undeformed_same_material'
    else:
        with np.load(path) as f:q=f['q'];t=float(f['time']);dt=float(f['dt'])
        name=str(path.relative_to(O))
    a=float(compression(t))
    F=np.eye(3)+np.diag([0.,0.,-a])+np.einsum('cni,qnj->cqij',q[cid[included]],grads)
    H=np.asarray(Hbatch(F.reshape(-1,3,3),rho[included].ravel())).reshape(len(included),27,3,3,3,3)
    assert np.isfinite(H).all()
    K=np.zeros((3*len(target),3*len(target)))
    for n,c in enumerate(included):
        kc=np.einsum('qijkl,qaj,qbl,q->aibk',H[n],grads,grads,weights[c]).reshape(81,81)
        dest=mapping[cid[c]];keep=np.flatnonzero(dest>=0)
        source=(keep[:,None]*3+np.arange(3)).ravel()
        sink=(dest[keep,None]*3+np.arange(3)).ravel()
        K[np.ix_(sink,sink)]+=kc[np.ix_(source,source)]
    symmetry=float(np.max(np.abs(K-K.T))/max(np.max(np.abs(K)),1e-30))
    m=np.repeat(mass[target],3)
    B=(K+K.T)/2/np.sqrt(m[:,None]*m[None,:])
    eig,vec=np.linalg.eigh(B)
    omega=float(np.sqrt(max(eig[-1],0)))
    direction=vec[:,-1]/np.sqrt(m)
    # Rayleigh quotient includes all cells touching the support; record both
    # direct quotient and eigendecomposition for a check on the restriction.
    rayleigh=float(direction@K@direction/(np.sum(m*direction**2)))
    row={'state':name,'compression':a,'dt_s':dt,'restricted_dofs':len(m),
      'incident_cells':len(included),'tangent_relative_asymmetry':symmetry,
      'largest_restricted_eigenvalue_s_minus2':float(eig[-1]),
      'rayleigh_s_minus2':rayleigh,'dt_times_omega_lower_bound':dt*omega,
      'upper_bound_on_positive_frequency_dt_limit_s':2/omega if omega else None,
      'frozen_positive_frequency_limit_violated_by_bound':bool(dt*omega>2),
      'negative_restricted_eigenvalues':int((eig<0).sum())}
    rows.append(row);print(json.dumps(row),flush=True)
result={'event_class':event,'central_cell':central,'event_incident_cells':eventcells.tolist(),
 'target_periodic_classes':target.tolist(),'rows':rows,'wall_seconds':time.perf_counter()-started,
 'scope':'Frozen-state undamped central-difference positive-frequency test. Exact restricted shared-energy tangent and original global HRZ mass; >2 establishes this frozen tangent violates the original dt limit. <=2 does not certify the full system. Negative eigenvalues are retained; no stiffness projection, trajectory or design AD.'}
(O/'analysis/patch_frequency.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

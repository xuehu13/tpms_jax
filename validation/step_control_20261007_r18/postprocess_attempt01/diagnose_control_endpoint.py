"""Read-only endpoint localization using shared material tangents and actual HRZ.

The restriction supplies a LOWER frequency bound, independent of the conservative
upper row bound. It never projects stiffness, changes parameters or advances time.
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
from hyperelastic_fem import make_density_hyperelastic_problem
jax.config.update('jax_enable_x64',True)
O=R/'validation/step_control_20261007_r18';D=O/'controlled_forward';A=O/'analysis'
assert (O/'forward_receipt.json').exists() and A.exists()
read=lambda p:json.loads(p.read_text())
started=time.perf_counter();N=32;levels=65
p=make_density_hyperelastic_problem(1,rho_quad=1.,eta=1e-4,periodic_axes=(0,1,2),
    element_degree=2,quadrature_order=4,material_model='objective_void')
grads=np.asarray(p.fe.shape_grads[0])*N
family,cell,_,_,degree,order=get_elements('HEX27')
local=np.rint(2*basix.create_element(family,cell,degree).points[order]).astype(int)
origins=2*np.indices((N,)*3).reshape(3,-1).T;indices=origins[:,None,:]+local[None,:,:]
cells=(indices[...,0]*levels+indices[...,1])*levels+indices[...,2]
old=R/'validation/geometry_transfer_20261006_r15'
with np.load(old/'hrz_mass.npz') as z:ids=z['class_ids'];mass=z['periodic_class_mass_normalized']
with np.load(old/'gauss_field.npz') as z:rho=z['rho'];weights=z['JxW']
cid=ids[cells];scale=1e-4+(1-1e-4)*rho
field=D/('field.npz' if (D/'result.json').exists() else 'last_valid_field.npz')
with np.load(field) as z:q=z['q'];t=float(z['time']);dt=float(z['dt'])
s=np.clip(t/.004,0,1);a=float(.2*(10*s**3-15*s**4+6*s**5))
F=np.eye(3)+np.diag([0.,0.,-a])+np.einsum('cni,qnj->cqij',q[cid],grads)
J=np.linalg.det(F);domains={}
for name,mask in {'deep_void':rho<=.001,'mixed_tail':(rho>.001)&(rho<.01),
    'uncontinued_NH':rho>=.01,'geometric_core':rho>=.5}.items():
    domains[name]={'points':int(mask.sum()),'minimum_J':float(J[mask].min()),
        'negative_J_points':int((J[mask]<=0).sum()),'maximum_F_norm':float(np.linalg.norm(F[mask],axis=(-2,-1)).max())}
nh=np.argwhere(rho>=.01);flat=int(np.argmin(J[rho>=.01]));where=tuple(nh[flat])
minimum_point={'cell':int(where[0]),'quadrature':int(where[1]),'rho':float(rho[where]),'J':float(J[where])}
b=read(D/'stability_path.json')[-1];event=int(b['max_periodic_class'])
central=int(np.flatnonzero((cid==event).any(axis=1))[0]);target=np.unique(cid[central]);target=target[target!=0]
included=np.flatnonzero(np.isin(cid,target).any(axis=1));mapping=np.full(len(mass),-1,dtype=int);mapping[target]=np.arange(len(target))
tangent=jax.jit(jax.vmap(jax.jacfwd(p.material_stress,argnums=0)))
H=np.asarray(tangent(jnp.asarray(F[included].reshape(-1,3,3)),jnp.asarray(scale[included].ravel()))).reshape(len(included),27,3,3,3,3)
assert np.isfinite(H).all()
K=np.zeros((3*len(target),3*len(target)));Kdomain={n:np.zeros_like(K) for n in ('deep_void','mixed_tail','uncontinued_NH')}
for n,c in enumerate(included):
    dest=mapping[cid[c]];keep=np.flatnonzero(dest>=0)
    source=(keep[:,None]*3+np.arange(3)).ravel();sink=(dest[keep,None]*3+np.arange(3)).ravel()
    masks={'deep_void':rho[c]<=.001,'mixed_tail':(rho[c]>.001)&(rho[c]<.01),'uncontinued_NH':rho[c]>=.01}
    for name,mask in masks.items():
        kc=np.einsum('qijkl,qaj,qbl,q->aibk',H[n],grads,grads,weights[c]*mask).reshape(81,81)
        Kdomain[name][np.ix_(sink,sink)]+=kc[np.ix_(source,source)]
for kd in Kdomain.values():K+=kd
m=np.repeat(mass[target],3);B=(K+K.T)/2/np.sqrt(m[:,None]*m[None,:]);eig,vec=np.linalg.eigh(B)
omega=float(np.sqrt(max(eig[-1],0)));direction=vec[:,-1]/np.sqrt(m)
contributions={name:float(direction@kd@direction/(np.sum(m*direction**2))) for name,kd in Kdomain.items()}
result={'field':str(field.relative_to(R)),'time_s':t,'compression':a,'actual_dt_s':dt,
    'material_domains':domains,'minimum_NH_point':minimum_point,'bound_limiting_class':event,
    'restricted_dofs':len(m),'incident_cells':len(included),'largest_restricted_eigenvalue_s_minus2':float(eig[-1]),
    'conservative_upper_R_s_minus2':b['R_s_minus2'],'upper_to_restricted_lower_eigenvalue_ratio':b['R_s_minus2']/float(eig[-1]),
    'dt_times_positive_frequency_lower_bound':dt*omega,'positive_frequency_limit_dt_upper_bound_s':2/omega,
    'negative_restricted_eigenvalues':int((eig<0).sum()),
    'tangent_relative_asymmetry':float(np.max(np.abs(K-K.T))/max(np.max(np.abs(K)),1e-30)),
    'selected_direction_curvature_contributions_s_minus2':contributions,
    'selected_direction_curvature_fraction':{name:value/float(eig[-1]) for name,value in contributions.items()},
    'wall_seconds':time.perf_counter()-started,
    'scope':'Frozen accepted endpoint at the upper-bound-limiting class. Restricted lower bound; below 2 cannot certify full stability. Directional curvature fractions are not total energy or force shares. No time advance/design AD.'}
# Mechanical domain energy and static macro-force are evaluated through the
# same material functions, not inferred from the selected high-frequency mode.
energy=jax.jit(jax.vmap(p.material_energy));stress=jax.jit(jax.vmap(p.material_stress))
path=read(D/'accepted_path.json');domain_rows=[]
for statefile in [*sorted(D.glob('accepted_a*.npz')),field]:
    with np.load(statefile) as z:qs=z['q'];ts=float(z['time'])
    ss=np.clip(ts/.004,0,1);aa=float(.2*(10*ss**3-15*ss**4+6*ss**5))
    Fs=np.eye(3)+np.diag([0.,0.,-aa])+np.einsum('cni,qnj->cqij',qs[cid],grads)
    Ws=[];Ps=[]
    for first in range(0,len(Fs),1024):
        fs=jnp.asarray(Fs[first:first+1024].reshape(-1,3,3));sc=jnp.asarray(scale[first:first+1024].ravel())
        Ws.append(np.asarray(energy(fs)).reshape(-1,27))
        Ps.append(np.asarray(stress(fs))[:,2,2].reshape(-1,27))
    W=np.concatenate(Ws);P33=np.concatenate(Ps);assert np.isfinite(W).all() and np.isfinite(P33).all()
    parts={}
    for name,mask in {'deep_void':rho<=.001,'mixed_tail':(rho>.001)&(rho<.01),'uncontinued_NH':rho>=.01}.items():
        parts[name]={'internal_energy_N_mm':float(np.sum(W*weights*mask)*10**3),
            'signed_internal_macro_Fz_N':float(np.sum(P33*weights*mask)*10**2)}
    row=min(path,key=lambda x:abs(x['time']-ts));assert abs(row['time']-ts)<1e-12
    assert abs(sum(v['internal_energy_N_mm'] for v in parts.values())-row['energy_N_mm'])<1e-9
    assert abs(sum(v['signed_internal_macro_Fz_N'] for v in parts.values())-row['internal_macro_Fz_N'])<1e-9
    domain_rows.append({'field':str(statefile.relative_to(R)),'compression':aa,'time_s':ts,
        'total_internal_energy_N_mm':row['energy_N_mm'],'total_signed_internal_macro_Fz_N':row['internal_macro_Fz_N'],
        'domains':parts,'actual_J_min_geometric_core':float(np.linalg.det(Fs)[rho>=.5].min())})
result['saved_state_energy_force_domain_decomposition']=domain_rows
result['wall_seconds']=time.perf_counter()-started
(A/'endpoint_tangent_diagnosis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))

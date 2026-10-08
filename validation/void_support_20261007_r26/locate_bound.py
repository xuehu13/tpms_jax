"""Frozen endpoint local row reconstruction on CPU; no path or spectrum solve."""
import os
os.environ['JAX_PLATFORMS']='cpu'
from pathlib import Path
import sys,json,time
import numpy as np
R=Path('/home/xuehu/projects/tpms_jax');sys.path.insert(0,str(R))
import jax,jax.numpy as jnp,basix
from jax_fem.basis import get_elements
from hyperelastic_fem import make_density_hyperelastic_problem,objective_void_energy,void_nh_weight
start=time.perf_counter();D=Path(__file__).resolve().parent;baseline=len(sys.argv)>2 and sys.argv[2]=='original';O=(R/'validation/step_control_20261007_r18/controlled_forward') if baseline else D/'forward';support=1. if baseline else .1
statefile=Path(sys.argv[1]) if len(sys.argv)>1 else O/('field.npz' if (O/'field.npz').exists() else 'last_valid_field.npz')
state=np.load(statefile);N=int(state['N']);q=state['q'];t=float(state['time'])
observations=json.loads((O/'accepted_path.json').read_text());which=min(range(len(observations)),key=lambda i:abs(observations[i]['time']-t));row=observations[which]
bound=json.loads((O/'stability_path.json').read_text())[which]
assert abs(t-bound['time'])<1e-12 and abs(t-row['time'])<1e-12
cache=np.load(R/'validation/geometry_transfer_20261006_r15/gauss_field.npz')
rho=cache['rho'];weights=cache['JxW'];scale=1e-4+.9999*rho
p=make_density_hyperelastic_problem(1,rho_quad=.5,periodic_axes=(0,1,2),element_degree=2,
 quadrature_order=4,material_model='objective_void')
grads=np.asarray(p.shape_grads[0])*N;shapes=np.asarray(p.fe.shape_vals)
assert np.max(np.abs(np.asarray(p.physical_quad_points[0])/N-cache['physical_quad_points'][0]))<5e-12
assert np.max(np.abs(np.asarray(p.fe.JxW[0])/N**3/weights[0]-1))<5e-12
family,cell,_,_,degree,order=get_elements('HEX27')
local=np.rint(2*basix.create_element(family,cell,degree).points[order]).astype(int)
origins=2*np.indices((N,)*3).reshape(3,-1).T
indices=origins[:,None,:]+local[None,:,:];periodic=indices%(2*N)
classes=(periodic[...,0]*(2*N)+periodic[...,1])*(2*N)+periodic[...,2]
diag=np.einsum('qn,cq,cq->cn',shapes**2,weights,scale)
total=np.sum(weights*scale,axis=1);cm=diag*(total/diag.sum(axis=1))[:,None]*1e-7
mass=np.zeros((2*N)**3);np.add.at(mass,classes.ravel(),cm.ravel())
inv=1/np.sqrt(mass);inv[0]=0.
target=bound['max_periodic_class'];component=bound['max_component']
selected=np.flatnonzero(np.any(classes==target,axis=1));ids=classes[selected];s=scale[selected]
H=np.zeros((3,3));H[2,2]=-row['compression']
F=np.eye(3)+H+np.einsum('cni,qnj->cqij',q[ids],grads)
r=(s-1e-4)/.9999;J=np.linalg.det(F)
def energy(X,S):
 phi=(S-1e-4)/.9999
 return (support+(1-support)*void_nh_weight(phi))*objective_void_energy(X,phi,1e-4)
A=np.asarray(jax.jit(jax.vmap(jax.hessian(energy,argnums=0)))(jnp.asarray(F.reshape(-1,3,3)),jnp.asarray(s.ravel()))).reshape((*s.shape,3,3,3,3))
Kq=np.einsum('cqijkl,qaj,qbl,cq->cqaibk',A,grads,grads,weights[selected])
def target_row(K):
 rows=np.sum(np.abs(K)*inv[ids][:,None,None,:,None],axis=(-2,-1))*inv[ids][:,:,None]
 return float(sum(rows[c,a,component] for c in range(len(selected)) for a in np.flatnonzero(ids[c]==target)))
computed=target_row(Kq.sum(axis=1));rel=abs(computed/bound['R_s_minus2']-1)
assert rel<1e-10
groups={}
for name,mask in [('deep_phi_le_0p001',r<=.001),('mixed_0p001_lt_phi_lt_0p01',(r>.001)&(r<.01)),('required_NH_phi_ge_0p01',r>=.01)]:
 groups[name]={'points':int(mask.sum()),'occupancy_min_max':[float(r[mask].min()),float(r[mask].max())] if mask.any() else None,'max_F_frobenius':float(np.linalg.norm(F,axis=(-2,-1))[mask].max()) if mask.any() else None,'max_F_singular_value':float(np.linalg.svd(F[mask],compute_uv=False).max()) if mask.any() else None,'actual_J_min':float(J[mask].min()) if mask.any() else None,
  'negative_J_points':int(np.sum(mask&(J<=0))),
  'separate_absolute_row_bound_s_minus2':target_row(np.sum(Kq*mask[:,:,None,None,None,None],axis=1))}
W=np.asarray(jax.jit(jax.vmap(energy))(jnp.asarray(F.reshape(-1,3,3)),jnp.asarray(s.ravel()))).reshape(s.shape)
for name,mask in [('deep_phi_le_0p001',r<=.001),('mixed_0p001_lt_phi_lt_0p01',(r>.001)&(r<.01)),('required_NH_phi_ge_0p01',r>=.01)]:
 groups[name]['local_stored_energy_N_mm']=float(np.sum((W*weights[selected])[mask])*1000)
idx=np.unravel_index(target,(2*N,)*3)
report={'support_energy_factor':support,'incident_periodic_mass_min_max':[float(mass[ids].min()),float(mass[ids].max())],'scope':'one accepted frozen endpoint; local conservative row diagnostic, not buckling eigenmode or unique peak cause',
 'compression':row['compression'],'state_file':str(statefile),'target_periodic_class':target,'component':component,
 'reference_class_position_mm':(10*np.asarray(idx)/(2*N)).tolist(),
 'incident_cells':selected.tolist(),'original_bound_R_s_minus2':bound['R_s_minus2'],
 'reconstructed_row_R_s_minus2':computed,'reconstruction_relative_error':rel,'partitions':groups,
 'partition_caution':'Absolute values are taken after integrating each group; separate bounds are not additive percentages of the assembled total.',
 'analysis_seconds':time.perf_counter()-start}
(D/('bound_at_checkpoint10_original.json' if baseline else ('bound_at_checkpoint10.json' if len(sys.argv)>1 else 'bound_location.json'))).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

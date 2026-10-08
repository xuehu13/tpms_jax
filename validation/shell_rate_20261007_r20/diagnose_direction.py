"""One fixed observed-jump direction at retained states; shared material, no solve."""
from pathlib import Path
import os,sys,json,time,hashlib
os.environ['JAX_PLATFORMS']='cpu'
R=Path('/home/xuehu/projects/tpms_jax');sys.path.insert(0,str(R))
import numpy as np
import jax
import jax.numpy as jnp
import basix
from jax_fem.basis import get_elements
from hyperelastic_fem import make_density_hyperelastic_problem
jax.config.update('jax_enable_x64',True)
D=R/'validation/shell_rate_20261007_r20/directional_diagnosis'
S=R/'validation/step_control_20261007_r18/controlled_forward'
protocol=json.loads((D/'protocol.json').read_text());start=time.perf_counter();N=32;levels=65
p=make_density_hyperelastic_problem(1,rho_quad=1.,eta=1e-4,periodic_axes=(0,1,2),element_degree=2,quadrature_order=4,material_model='objective_void')
grads=np.asarray(p.fe.shape_grads[0])*N
family,cell,_,_,degree,order=get_elements('HEX27')
local=np.rint(2*basix.create_element(family,cell,degree).points[order]).astype(int)
origins=2*np.indices((N,)*3).reshape(3,-1).T;indices=origins[:,None,:]+local[None,:,:]
nodes=(indices[...,0]*levels+indices[...,1])*levels+indices[...,2]
old=R/'validation/geometry_transfer_20261006_r15'
with np.load(old/'hrz_mass.npz') as f:ids=f['class_ids']
with np.load(old/'gauss_field.npz') as f:rho=f['rho'];weights=f['JxW'];points=f['physical_quad_points']
assert np.max(np.abs(origins[:,None,:]/(2*N)+np.asarray(p.physical_quad_points)[0][None,:,:]/N-points))<5e-14
cid=ids[nodes];scale=1e-4+(1-1e-4)*rho
with np.load(S/'last_valid_field.npz') as f:dq=f['q']
with np.load(S/'accepted_a0.1600.npz') as f:dq=dq-f['q']
assert np.max(np.abs(dq[0]))<1e-15 and np.isfinite(dq).all()
deltaF=np.einsum('cni,qnj->cqij',dq[cid],grads)
def one(F,sc,dF):
    _,dP=jax.jvp(lambda x:p.material_stress(x,sc),(F,),(dF,))
    return jnp.sum(dF*dP)
evaluate=jax.jit(jax.vmap(one));rows=[]
for name in protocol['states']:
    with np.load(S/name) as f:q=f['q'];t=float(f['time'])
    ss=np.clip(t/.004,0,1);a=float(.2*(10*ss**3-15*ss**4+6*ss**5))
    F=np.eye(3)+np.diag([0.,0.,-a])+np.einsum('cni,qnj->cqij',q[cid],grads)
    J=np.linalg.det(F);assert np.all(J[rho>=.01]>0) and np.isfinite(F).all()
    out=[]
    for first in range(0,len(F),1024):
        out.append(np.asarray(evaluate(jnp.asarray(F[first:first+1024].reshape(-1,3,3)),
            jnp.asarray(scale[first:first+1024].ravel()),jnp.asarray(deltaF[first:first+1024].reshape(-1,3,3)))).reshape(-1,27))
    c=np.concatenate(out);assert np.isfinite(c).all()
    masks={'deep_void':rho<=.001,'mixed_tail':(rho>.001)&(rho<.01),
           'main_interface':(rho>=.01)&(rho<.5),'geometric_core':rho>=.5}
    parts={k:float(np.sum(c*weights*m)*1000) for k,m in masks.items()}
    total=float(np.sum(c*weights)*1000);assert abs(sum(parts.values())-total)<1e-8*max(abs(total),1.)
    row={'state':name,'compression':a,'directional_second_variation_N_mm':total,'parts_N_mm':parts,
         'uncontinued_NH_part_N_mm':parts['main_interface']+parts['geometric_core'],
         'minimum_NH_J':float(J[rho>=.01].min())}
    rows.append(row);print(json.dumps(row),flush=True)
result={'protocol':protocol,'rows':rows,'wall_seconds':time.perf_counter()-start,
    'direction_max_nodal_increment_mm':float(np.linalg.norm(dq,axis=1).max()*10),
    'same_direction_at_all_states':True,'actual_unclipped_F_J':True,'new_time_advance':False,
    'design_AD':False,'not_full_spectrum_or_static_critical_point':True,
    'material_source_sha256':hashlib.sha256((R/'hyperelastic_fem.py').read_bytes()).hexdigest()}
(D/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({'seconds':result['wall_seconds'],'states':len(rows)}),flush=True)

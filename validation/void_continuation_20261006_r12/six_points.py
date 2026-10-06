"""Independent local derivative checks at the six frozen failure points."""
from pathlib import Path
import hashlib,json,sys
import basix,numpy as np
import jax,jax.numpy as jnp
R=Path('/home/xuehu/projects/tpms_jax');O=R/'validation/void_continuation_20261006_r12'
sys.path.insert(0,str(R))
from jax_fem.basis import get_elements
from hyperelastic_fem import objective_void_energy,objective_void_first_piola,void_nh_cutoff
old=R/'validation/objective_path_20261006_r11'
points=json.loads((old/'domain_probe/invalid_points.json').read_text())['points']
with np.load(old/'T0p004/field.npz') as f:q=f['q']
family,cell,_,_,degree,order=get_elements('HEX27');el=basix.create_element(family,cell,degree)
local=np.rint(2*el.points[order]).astype(int)
F=[];rho=[]
for p in points:
    index=p['cell_id'];origin=np.array([index//32**2,(index//32)%32,index%32])
    grid=(2*origin[None,:]+local)%64;ids=(grid[:,0]*64+grid[:,1])*64+grid[:,2]
    xi=np.array(p['reference_xyz_mm'])[None,:]*32/10-origin
    grad=el.tabulate(1,xi).take(order,axis=2)[1:4,0,:,0].T*32
    f=np.diag([1.,1.,.8])+np.einsum('ni,nj->ij',q[ids],grad)
    assert abs(np.linalg.det(f)-p['actual_J'])<1e-12
    F.append(f);rho.append(p['phi'])
F=jnp.asarray(F);rho=jnp.asarray(rho)
W=jax.jit(jax.vmap(objective_void_energy))(F,rho)
P=jax.jit(jax.vmap(objective_void_first_piola))(F,rho)
A=jax.jit(jax.vmap(jax.hessian(objective_void_energy,argnums=0)))(F,rho)
dr=jax.jit(jax.vmap(jax.grad(objective_void_energy,argnums=1)))(F,rho)
assert all(np.isfinite(np.asarray(x)).all() for x in [W,P,A,dr])
D=jnp.array([[.02,.03,.01],[.04,-.01,.02],[-.02,.01,.03]])
entries=[]
for i,p in enumerate(points):
    f=F[i];phi=rho[i];eps=1e-7
    ad=float(jnp.sum(P[i]*D));fd=float((objective_void_energy(f+eps*D,phi)-objective_void_energy(f-eps*D,phi))/(2*eps))
    adr=float(dr[i]);fdr=float((objective_void_energy(f,phi+eps)-objective_void_energy(f,phi-eps))/(2*eps))
    adA=np.asarray(jax.jvp(lambda x:objective_void_first_piola(x,phi),(f,),(D,))[1])
    fdA=np.asarray((objective_void_first_piola(f+eps*D,phi)-objective_void_first_piola(f-eps*D,phi))/(2*eps))
    estate=abs(ad-fd)/max(abs(ad),1e-12);erho=abs(adr-fdr)/max(abs(adr),1e-12)
    etangent=float(np.linalg.norm(adA-fdA)/max(np.linalg.norm(adA),1e-12))
    assert estate<1e-5 and erho<1e-5 and etangent<1e-5
    rotE=0.;rotP=0.
    for angle in [60,90]:
        t=np.deg2rad(angle);Q=jnp.array([[np.cos(t),-np.sin(t),0],[np.sin(t),np.cos(t),0],[0,0,1.]])
        rotE=max(rotE,abs(float(objective_void_energy(Q@f,phi)-W[i]))/max(abs(float(W[i])),1e-12))
        rotP=max(rotP,float(np.max(abs(np.asarray(objective_void_first_piola(Q@f,phi)-Q@P[i])))))
    assert rotE<1e-11 and rotP<1e-10
    entries.append({'cell_id':p['cell_id'],'phi':float(phi),'actual_J':p['actual_J'],
        'cutoff_J':float(void_nh_cutoff(phi)),'candidate_W_MPa':float(W[i]),
        'state_derivative_relative_error':estate,'occupancy_derivative_relative_error':erho,
        'stress_tangent_relative_error':etangent,'rotation_energy_relative_residual':rotE,
        'rotation_stress_absolute_residual_MPa':rotP})
record={'scope':'Six actual old failed sample locations, fixed F; local AD/FD, not trajectory gradient',
    'passed':True,'entries':entries,'maximum_state_derivative_relative_error':max(p['state_derivative_relative_error'] for p in entries),
    'maximum_occupancy_derivative_relative_error':max(p['occupancy_derivative_relative_error'] for p in entries),
    'maximum_tangent_relative_error':max(p['stress_tangent_relative_error'] for p in entries),
    'input_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [old/'T0p004/field.npz',old/'domain_probe/invalid_points.json',R/'hyperelastic_fem.py']}}
target=O/'six_points.json';assert not target.exists();target.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))

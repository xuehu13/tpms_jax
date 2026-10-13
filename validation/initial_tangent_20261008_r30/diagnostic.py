"""One bounded N32 initial linear equilibrium experiment, not a production solver."""
from pathlib import Path
import sys, os, time, json, hashlib
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
R=Path('/home/xuehu/projects/tpms_jax'); D=Path(__file__).resolve().parent
sys.path.insert(0,str(R))
import numpy as np
import jax, jax.numpy as jnp
from hyperelastic_fem import make_density_hyperelastic_problem, MU, KAPPA
from scripts.thin_target_explicit import ExplicitXYZ
jax.config.update('jax_enable_x64',True)
L=10.; A=1e-4; ETA=1e-4; MAX_ITER=6000; SOLVE_SECONDS=900
def write(name,v): (D/name).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def freeze():
    for name,h in json.loads((D/'frozen_before.json').read_text()).items():
        assert sha(R/name)==h,name
freeze(); start=time.perf_counter()
cache=np.load(R/'validation/geometry_transfer_20261006_r15/gauss_field.npz')
rho=cache['rho']; qp=cache['physical_quad_points']
p=make_density_hyperelastic_problem(32,eta=ETA,rho_quad=rho,periodic_axes=(0,1,2),
    element_degree=2,quadrature_order=4,material_model='objective_void')
assert np.max(abs(np.asarray(p.physical_quad_points)-qp))<1e-14
ex=ExplicitXYZ(p,L=L); ids=jnp.asarray(np.asarray(p.class_ids)[np.asarray(p.fe.cells)],dtype=jnp.int32)
g=ex.kernel_geometry[1]; w=ex.kernel_geometry[2][0]; s=ex.scale
lam=KAPPA-2*MU/3; eye=jnp.eye(3); H=jnp.zeros((3,3)).at[2,2].set(-A)
q0=jnp.zeros((ex.nc,3))
def project(x): return x-jnp.mean(x,axis=0,keepdims=True)
def strain_stress(q,h):
    grad=h+jnp.einsum('cni,qnj->cqij',q[ids],g)
    eps=.5*(grad+jnp.swapaxes(grad,-1,-2))
    stress=s[:,:,None,None]*(lam*jnp.trace(eps,axis1=-2,axis2=-1)[:,:,None,None]*eye+2*MU*eps)
    return grad,eps,stress
@jax.jit
def force(q,h):
    stress=strain_stress(q,h)[2]
    fc=jnp.einsum('cqij,qnj,q->cni',stress,g,w)
    return jnp.zeros_like(q).at[ids.ravel()].add(fc.reshape(-1,3))
@jax.jit
def op(q): return project(force(q,jnp.zeros((3,3))))
@jax.jit
def diagonal():
    local=MU*jnp.sum(g*g,axis=-1)[:,:,None]+(lam+MU)*g*g
    dc=jnp.einsum('cq,qni,q->cni',s,local,w)
    return jnp.zeros_like(q0).at[ids.ravel()].add(dc.reshape(-1,3))
diag=diagonal(); diag.block_until_ready(); assert float(diag.min())>0
b=-project(force(q0,H)); b.block_until_ready(); bn=float(jnp.linalg.norm(b))
# Validate against the current shared constitutive derivative and actual whole-mesh residual JVP.
scales=jnp.asarray(ETA+(1-ETA)*np.array([0,.0005,.001,.005,.01,.5,1.]))
tangent=jax.jit(jax.vmap(jax.jacfwd(p.material_stress,argnums=0)))
C=lam*jnp.einsum('ij,kl->ijkl',eye,eye)+MU*(jnp.einsum('ik,jl->ijkl',eye,eye)+jnp.einsum('il,jk->ijkl',eye,eye))
ce=float(jnp.max(abs(tangent(jnp.broadcast_to(eye,(7,3,3)),scales)-scales[:,None,None,None,None]*C)))
checks={'constitutive_initial_tangent_error_le_1e-11':ce<1e-11}
errors=[]
for qd,hd in [(q0,-A),(project(jax.random.normal(jax.random.key(30),q0.shape,dtype=jnp.float64))*1e-5,0.)]:
    exact=ex.reduce(jax.jvp(ex.force,(q0,jnp.array(0.)),(qd,jnp.array(hd)))[1])
    hh=jnp.zeros((3,3)).at[2,2].set(hd)
    approx=force(qd,hh)
    errors.append(float(jnp.linalg.norm(approx-exact)/jnp.linalg.norm(exact)))
checks['whole_mesh_shared_force_JVP_le_5e-10']=max(errors)<5e-10
write('operator_validation.json',{'checks':checks,'constitutive_max_absolute_error':ce,'whole_mesh_relative_errors':errors})
assert all(checks.values()),checks
prep=time.perf_counter()-start
print(json.dumps({'stage':'prepared','seconds':prep,'classes':ex.nc,'device':str(jax.devices()[0]),'JVP_errors':errors}),flush=True)
@jax.jit
def chunk(state):
    def one(st,_):
        x,r,z,v,rz,err,k=st
        def advance(st):
            x,r,z,v,rz,err,k=st; kv=op(v); denom=jnp.vdot(v,kv)
            alpha=rz/denom; x=project(x+alpha*v); r=project(r-alpha*kv)
            z=project(r/diag); rz2=jnp.vdot(r,z); v=project(z+(rz2/rz)*v)
            err=jnp.linalg.norm(r)/bn
            err=jnp.where((denom>0)&jnp.isfinite(err),err,jnp.nan)
            return x,r,z,v,rz2,err,k+1
        return jax.lax.cond((err>1e-10)&jnp.isfinite(err)&(k<MAX_ITER),advance,lambda st:st,st),None
    return jax.lax.scan(one,state,None,length=50)[0]
r=b; z=project(r/diag); state=(q0,r,z,z,jnp.vdot(r,z),jnp.array(1.),jnp.array(0))
solve_start=time.perf_counter(); history=[]; stop='iteration_limit'
while int(state[-1])<MAX_ITER:
    if time.perf_counter()-solve_start>SOLVE_SECONDS: stop='time_budget'; break
    state=chunk(state); state[0].block_until_ready(); k=int(state[-1]); err=float(state[-2])
    row={'iteration':k,'recurrence_relative_residual':err,'seconds':time.perf_counter()-solve_start}
    if k%250==0 or err<=1e-10 or not np.isfinite(err):
        row['true_relative_residual']=float(jnp.linalg.norm(op(state[0])-b)/bn)
    history.append(row); print(json.dumps(row),flush=True)
    if not np.isfinite(err): stop='PCG_breakdown'; break
    if err<=1e-10: stop='recurrence_tolerance'; break
solve_seconds=time.perf_counter()-solve_start
q=state[0]-state[0][ex.pin]; final_res=force(q,H)
free_res=final_res.at[ex.pin].set(0.)
relative=float(jnp.linalg.norm(free_res)/bn)
@jax.jit
def metrics(q):
    grad,eps,stress=strain_stress(q,H)
    U=.5*jnp.sum(stress*eps*w[None,:,None,None])*L**3
    F=jnp.sum(stress[:,:,2,2]*w[None,:])*L**2
    return U,F,jnp.min(jnp.linalg.det(eye+grad)),jnp.max(abs(eps))
U,F,Jmin,emax=[float(v) for v in metrics(q)]
macro_energy=.5*F*(-A*L); energy_error=abs(U/macro_energy-1)
checks.update({'true_free_relative_residual_le_1e-8':relative<=1e-8,
    'energy_work_relative_le_1e-6':energy_error<=1e-6,
    'finite_solution':bool(np.isfinite(np.asarray(q)).all()),
    'small_strain_range':emax<=.02 and Jmin>.98,
    'solve_budget':solve_seconds<=SOLVE_SECONDS,'restored_pin':float(jnp.max(abs(q[ex.pin])))==0.})
# Exact class mapping constructs periodic fluctuations; verify opposite original-node pairs too.
full=np.asarray(q)[np.asarray(p.class_ids)].reshape(65,65,65,3)
pgap=max(float(np.max(abs(np.take(full,0,axis=i)-np.take(full,-1,axis=i)))) for i in range(3))
checks['periodic_fluctuation_gap_le_1e-12']=pgap<=1e-12
result={'status':'ok' if all(checks.values()) else 'not_accepted','checks':checks,
    'case':'diverse_04','compression':A,'delta_mm':-A*L,'Fz_N':F,'stiffness_N_per_mm':F/(-A*L),
    'energy_N_mm':U,'macro_energy_N_mm':macro_energy,'energy_relative_error':energy_error,
    'true_free_relative_residual':relative,'periodic_gap_normalized':pgap,'minimum_linearized_state_J':Jmin,
    'max_abs_strain':emax,'stop':stop,'iterations':int(state[-1]),'preparation_seconds':prep,
    'solve_seconds':solve_seconds,'wall_seconds':time.perf_counter()-start,
    'gauge':'mean-zero periodic solve, then restore the original fixed-class translation',
    'mass_in_operator':False,'design_gradient_certified':False,'operator_JVP_errors':errors,
    'weighted_volume_mm3':float(jnp.sum(s*w[None,:])*L**3)}
np.savez_compressed(D/'linear_state.npz',q=np.asarray(q),class_ids=np.asarray(p.class_ids),H=np.asarray(H))
write('convergence.json',history); write('jax_result.json',result); freeze()
print(json.dumps(result,indent=2),flush=True)
if result['status']!='ok': sys.exit(2)

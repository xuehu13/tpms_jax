"""One narrow-width initial equilibrium at the existing mixed rule. Production unchanged."""
from pathlib import Path
import os,sys,json,time,hashlib
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
R=Path('/home/xuehu/projects/tpms_jax');D=Path(__file__).resolve().parent
sys.path.insert(0,str(R))
import numpy as np
import basix
from scipy.special import expit
import jax,jax.numpy as jnp
from jax_fem.basis import get_elements
from hyperelastic_fem import make_density_hyperelastic_problem,MU,KAPPA
from scripts.thin_target_explicit import ExplicitXYZ
from surface_distance import PeriodicSurfaceDistance
jax.config.update('jax_enable_x64',True)
L=10.;N=32;A=1e-4;ETA=1e-4;ELL=.025/(2*np.log(9));MAX_ITER=6000;BUDGET=900.
Q=R/'validation/local_quadrature_20261008_r33'
def write(n,v):(D/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def freeze():
    for n,h in json.loads((D/'frozen_before.json').read_text()).items():assert sha(R/n)==h,n
start=time.perf_counter();freeze();cfg=json.loads((D/'protocol.json').read_text())
assert cfg['total_budget_seconds']==BUDGET and cfg['maximum_iterations']==MAX_ITER
assert cfg['dense_points_per_axis']==12 and cfg['interface_10_90_mm']==.025
assert json.loads((Q/'result.json').read_text())['status']=='ok'
sel=np.load(Q/'selected_cells.npy');assert sha(Q/'selected_cells.npy')==cfg['selected_cells_sha256']
c=np.load(R/'validation/geometry_transfer_20261006_r15/gauss_field.npz')
old=np.load(R/'validation/interface_width_20261008_r37/width_0p025/linear_state.npz')
wr=json.loads((R/'validation/interface_width_20261008_r37/result.json').read_text())
wcase=next(x for x in wr['cases'] if x['interface_10_90_mm']==.025)
wlocal=next(x for x in wr['local_integration'] if x['width_mm']==.025)
assert wcase['status']=='ok' and wlocal['stable_existing_levels']
rho_old=expit((.25-c['distance']*L)/ELL)
p=make_density_hyperelastic_problem(N,eta=ETA,rho_quad=c['rho'],periodic_axes=(0,1,2),
    element_degree=2,quadrature_order=4,material_model='objective_void')
ex=ExplicitXYZ(p,L=L)
assert np.array_equal(np.asarray(p.class_ids),old['class_ids'])
assert np.max(abs(np.asarray(p.physical_quad_points)-c['physical_quad_points']))<1e-14
ids=jnp.asarray(np.asarray(p.class_ids)[np.asarray(p.fe.cells)],dtype=jnp.int32)
g=ex.kernel_geometry[1];w=ex.kernel_geometry[2][0];s=jnp.asarray(ETA+(1-ETA)*rho_old)
lam=KAPPA-2*MU/3;eye=jnp.eye(3);H=jnp.zeros((3,3)).at[2,2].set(-A)
q0=jnp.zeros((ex.nc,3));patch_ids=ids[sel]
family,cell,_,_,degree,order=get_elements('HEX27');el=basix.create_element(family,cell,degree)
local=el.points[order]/N
origin=np.asarray(p.physical_quad_points)[sel,0,:]-np.asarray(p.physical_quad_points)[0,0,:]
# Cell zero has origin zero; this identity is checked against connectivity coordinates.
origin_check=np.asarray(p.fe.points)[np.asarray(p.fe.cells)[sel]].min(axis=1)
assert np.max(abs(origin-origin_check))<1e-14
surface=PeriodicSurfaceDistance(c['surface_vertices'],c['surface_triangles'])
z,wt=np.polynomial.legendre.leggauss(12);z=(z+1)/2;wt/=2
ref=np.array([[a,b,c] for a in z for b in z for c in z])
weights=np.array([a*b*c for a in wt for b in wt for c in wt])/N**3
gd=el.tabulate(1,ref)[1:,:,:,:][:,:,:,0].transpose(1,2,0)[:,order,:]*N
g_old=np.asarray(g);w_old=np.asarray(w)
def element_matrix(gg,coeff):
    v=gg.reshape(len(gg),81)
    gram=np.einsum('cq,qi,qk->cik',coeff,v,v,optimize=True).reshape(-1,27,3,27,3)
    bb=np.einsum('cnjmj->cnm',gram)
    out=lam*gram+MU*gram.transpose(0,1,4,3,2)
    out+=MU*bb[:, :,None,:,None]*np.eye(3)[None,None,:,None,:]
    return out.reshape(-1,81,81)
delta=np.empty((len(sel),81,81));max_asym=0.;last=time.perf_counter()
for pos in range(0,len(sel),32):
    if time.perf_counter()-start>BUDGET:raise TimeoutError('matrix preparation exceeded frozen budget')
    which=sel[pos:pos+32];points=origin[pos:pos+len(which),None,:]+ref[None,:,:]/N
    rho=expit((.25-surface.query(points)*L)/ELL)
    kd=element_matrix(gd,(ETA+(1-ETA)*rho)*weights)
    ko=element_matrix(g_old,(ETA+(1-ETA)*rho_old[which])*w_old)
    delta[pos:pos+len(which)]=kd-ko
    max_asym=max(max_asym,float(np.max(abs(kd-kd.swapaxes(-1,-2)))/np.max(abs(kd))))
    if time.perf_counter()-last>10:
        print(json.dumps({'stage':'matrix_preparation','completed_cells':pos+len(which),'seconds':time.perf_counter()-start}),flush=True)
        last=time.perf_counter()
oldq=np.asarray(old['q'])[np.asarray(p.class_ids)[np.asarray(p.fe.cells)[sel]]]
aff=local@np.asarray(H).T;uold=(oldq+aff).reshape(len(sel),81)
du=.5*np.einsum('ci,cij,cj->',uold,delta,uold)*L**3
expected_delta=wlocal['dense12_energy_N_mm']-wlocal['original_selected_energy_N_mm']
reconstruction_error=float(abs(du/expected_delta-1))
trans=np.tile(np.eye(3),(27,1));translation_error=float(np.max(abs(delta@trans))/np.max(abs(delta)))
checks={'matrix_symmetry_le_1e-12':max_asym<=1e-12,'constant_translation_le_1e-11':translation_error<=1e-11,
        'r37_width025_fixed_state_energy_reproduced_le_1e-9':reconstruction_error<=1e-9}
write('operator_validation.json',{'checks':checks,'matrix_symmetry_relative':max_asym,
 'translation_relative':translation_error,'r33_delta_energy_N_mm':expected_delta,
 'matrix_delta_energy_N_mm':float(du),'relative_energy_error':reconstruction_error})
assert all(checks.values()),checks
delta=jnp.asarray(delta);aff=jnp.asarray(aff);h_basis=jnp.zeros((27,3)).at[:,2].set(jnp.asarray(local[:,2]))
def project(x):return x-jnp.mean(x,axis=0,keepdims=True)
def strain_stress(q,h):
    grad=h+jnp.einsum('cni,qnj->cqij',q[ids],g)
    eps=.5*(grad+jnp.swapaxes(grad,-1,-2))
    stress=s[:,:,None,None]*(lam*jnp.trace(eps,axis1=-2,axis2=-1)[:,:,None,None]*eye+2*MU*eps)
    return grad,eps,stress
@jax.jit
def original_force(q,h):
    fc=jnp.einsum('cqij,qnj,q->cni',strain_stress(q,h)[2],g,w)
    return jnp.zeros_like(q).at[ids.ravel()].add(fc.reshape(-1,3))
@jax.jit
def correction(q,h):
    u=(q[patch_ids]+jnp.asarray(local)@h.T).reshape(len(sel),81)
    fc=jnp.einsum('cij,cj->ci',delta,u).reshape(len(sel),27,3)
    return jnp.zeros_like(q).at[patch_ids.ravel()].add(fc.reshape(-1,3))
@jax.jit
def force(q,h):return original_force(q,h)+correction(q,h)
@jax.jit
def op(q):return project(force(q,jnp.zeros((3,3))))
@jax.jit
def diagonal():
    aa=MU*jnp.sum(g*g,axis=-1)[:,:,None]+(lam+MU)*g*g
    dc=jnp.einsum('cq,qni,q->cni',s,aa,w)
    base=jnp.zeros_like(q0).at[ids.ravel()].add(dc.reshape(-1,3))
    dd=jnp.diagonal(delta,axis1=-2,axis2=-1).reshape(len(sel),27,3)
    return base.at[patch_ids.ravel()].add(dd.reshape(-1,3))
diag=diagonal();diag.block_until_ready();assert float(diag.min())>0
b=-project(force(q0,H));b.block_until_ready();bn=float(jnp.linalg.norm(b))
before=jnp.asarray(old['q']);before=project(before)
@jax.jit
def metrics(q):
    grad,eps,stress=strain_stress(q,H)
    ub=.5*jnp.sum(stress*eps*w[None,:,None,None])*L**3
    fb=jnp.sum(stress[:,:,2,2]*w[None,:])*L**2
    u=(q[patch_ids]+aff).reshape(len(sel),81)
    df=jnp.einsum('cij,cj->ci',delta,u).reshape(len(sel),27,3)
    uu=.5*jnp.sum(u.reshape(len(sel),27,3)*df)*L**3
    ff=jnp.sum(df*h_basis)*L**2
    return ub+uu,fb+ff,jnp.min(jnp.linalg.det(eye+grad)),jnp.max(abs(eps))
fixed_u=float(metrics(before)[0]);expected_u=wcase['U_N_mm']+expected_delta
checks['fixed_original_state_total_energy_le_1e-10']=abs(fixed_u/expected_u-1)<=1e-10
assert checks['fixed_original_state_total_energy_le_1e-10']
@jax.jit
def chunk(state):
    def one(st,_):
        def advance(st):
            x,r,z,v,rz,err,k=st;kv=op(v);denom=jnp.vdot(v,kv)
            alpha=rz/denom;x=project(x+alpha*v);r=project(r-alpha*kv)
            z=project(r/diag);rz2=jnp.vdot(r,z);v=project(z+(rz2/rz)*v)
            err=jnp.linalg.norm(r)/bn;err=jnp.where((denom>0)&jnp.isfinite(err),err,jnp.nan)
            return x,r,z,v,rz2,err,k+1
        return jax.lax.cond((st[-2]>1e-10)&jnp.isfinite(st[-2])&(st[-1]<MAX_ITER),advance,lambda st:st,st),None
    return jax.lax.scan(one,state,None,length=50)[0]
r=b-op(before);z=project(r/diag)
state=(before,r,z,z,jnp.vdot(r,z),jnp.linalg.norm(r)/bn,jnp.array(0));history=[];stop='iteration_limit'
prep=time.perf_counter()-start;solve_start=time.perf_counter()
print(json.dumps({'stage':'prepared','seconds':prep,'classes':ex.nc,'selected_cells':len(sel),
 'initial_relative_residual':float(state[-2]),'fixed_old_state_energy_N_mm':fixed_u}),flush=True)
while int(state[-1])<MAX_ITER:
    if time.perf_counter()-start>BUDGET:stop='total_time_budget';break
    state=chunk(state);state[0].block_until_ready();k=int(state[-1]);err=float(state[-2])
    row={'iteration':k,'recurrence_relative_residual':err,'seconds':time.perf_counter()-start}
    if k%250==0 or err<=1e-10 or not np.isfinite(err):
        row['true_relative_residual']=float(jnp.linalg.norm(op(state[0])-b)/bn)
        print(json.dumps(row),flush=True)
    history.append(row)
    if not np.isfinite(err):stop='PCG_breakdown';break
    if err<=1e-10:stop='recurrence_tolerance';break
q=state[0]-state[0][ex.pin];res=force(q,H).at[ex.pin].set(0.);relative=float(jnp.linalg.norm(res)/bn)
U,F,Jmin,emax=[float(v) for v in metrics(q)]
work=.5*F*(-A*L);enerr=abs(U/work-1)
full=np.asarray(q)[np.asarray(p.class_ids)].reshape(65,65,65,3)
pgap=max(float(np.max(abs(np.take(full,0,axis=i)-np.take(full,-1,axis=i)))) for i in range(3))
checks.update({'true_free_relative_residual_le_1e-8':relative<=1e-8,'energy_work_relative_le_1e-6':enerr<=1e-6,
 'periodic_gap_le_1e-12':pgap<=1e-12,'finite_solution':bool(np.isfinite(np.asarray(q)).all()),
 'small_strain_range':emax<=.02 and Jmin>.98,'total_budget':time.perf_counter()-start<=BUDGET,
 'restored_pin':float(jnp.max(abs(q[ex.pin])))==0.,'energy_below_fixed_state':U<=fixed_u*(1+1e-10)})
old_result={'stiffness_N_per_mm':wcase['K0_N_per_mm']}
base=json.loads((R/'validation/local_reequilibrium_20261008_r34/result.json').read_text())
assert base['status']=='ok' and base['selected_cells']==len(sel)
shell_k=cfg['shell_stiffness_N_per_mm'];K=F/(-A*L)
out={'status':'ok' if all(checks.values()) else 'not_accepted','checks':checks,
 'case':'diverse_04 N32 initial static; interface .025, existing selected-12/rest-27 rule',
 'interface_10_90_mm':.025,'baseline_width_mm':.05,
 'relative_to_same_rule_width005':K/base['stiffness_N_per_mm']-1,
 'same_rule_width005_stiffness_N_per_mm':base['stiffness_N_per_mm'],
 'same_rule_width005_relative_to_shell':base['relative_to_shell'],
 'compression':A,'selected_cells':len(sel),'original_rule_points':27,'selected_rule_points':1728,
 'Fz_N':F,'stiffness_N_per_mm':K,'energy_N_mm':U,'macro_energy_N_mm':work,
 'energy_work_relative_error':enerr,'true_free_relative_residual':relative,'periodic_gap':pgap,
 'minimum_linearized_state_J':Jmin,'max_abs_strain':emax,'stop':stop,'iterations':int(state[-1]),
 'own_width_original27_stiffness_N_per_mm':old_result['stiffness_N_per_mm'],'shell_stiffness_N_per_mm':shell_k,
 'relative_to_own_width_original27':K/old_result['stiffness_N_per_mm']-1,'relative_to_shell':K/shell_k-1,
 'own_width_original27_relative_to_shell':old_result['stiffness_N_per_mm']/shell_k-1,
 'fixed_original_state_energy_N_mm':fixed_u,'reequilibration_release_N_mm':fixed_u-U,
 'preparation_seconds':prep,'solve_seconds':time.perf_counter()-solve_start,'wall_seconds':time.perf_counter()-start,
 'production_changed':False,'mass_or_damping_used':False,'new_Abaqus_job':False,
 'full_domain_dense_rule':False,'complete20pct_or_design_AD_certified':False}
np.savez_compressed(D/'linear_state.npz',q=np.asarray(q),class_ids=np.asarray(p.class_ids),H=np.asarray(H))
write('convergence.json',history);write('result.json',out);freeze();print(json.dumps(out,indent=2),flush=True)
if out['status']!='ok':sys.exit(2)

"""One flat membrane check reusing r27 mesh and the current shared material tangent."""
from pathlib import Path
import os,sys,json,time,hashlib,importlib.util
os.environ.setdefault('JAX_PLATFORMS','cpu')
R=Path('/home/xuehu/projects/tpms_jax');D=Path(__file__).resolve().parent;sys.path.insert(0,str(R))
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import jax,jax.numpy as jnp
spec=importlib.util.spec_from_file_location('r27_reuse',R/'validation/plate_bending_20261008_r27/diagnostic.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
def write(name,v):(D/name).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def freeze():
 for name,h in json.loads((D/'frozen_before.json').read_text()).items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h,name
freeze();start=time.perf_counter();p,_=old.mesh_problem(1.25)
cells=np.asarray(p.fe.cells);scale=np.asarray(p.stiffness_scale);w=np.asarray(p.fe.JxW);g=np.asarray(p.shape_grads)
tan=jax.jit(jax.vmap(jax.jacfwd(p.material_stress,argnums=0)))
A=np.asarray(tan(jnp.broadcast_to(jnp.eye(3),(scale.size,3,3)),jnp.asarray(scale.ravel()))).reshape(*scale.shape,3,3,3,3)
lam=old.E*old.NU/((1+old.NU)*(1-2*old.NU));mu=old.E/(2*(1+old.NU));eye=np.eye(3)
C=lam*np.einsum('ij,kl->ijkl',eye,eye)+mu*(np.einsum('ik,jl->ijkl',eye,eye)+np.einsum('il,jk->ijkl',eye,eye))
tangent_error=float(np.max(abs(A-scale[:,:,None,None,None,None]*C)));assert tangent_error<1e-11
Kc=np.einsum('cqijkl,cqaj,cqbl,cq->caibk',A,g,g,w,optimize=True).reshape(len(cells),81,81)*old.L
dofs=(3*cells[:,:,None]+np.arange(3)).reshape(len(cells),81)
K=sp.coo_matrix((Kc.ravel(),(np.repeat(dofs,81,axis=1).ravel(),np.tile(dofs,(1,81)).ravel())),shape=(len(p.fe.points)*3,)*2).tocsr()
K.sum_duplicates();K.eliminate_zeros();points=np.asarray(p.fe.points)*old.L
left=np.flatnonzero(np.isclose(points[:,0],0));right=np.flatnonzero(np.isclose(points[:,0],old.L))
zpin=int(np.argmin(np.linalg.norm(points-[0,0,1.25],axis=1)))
fixed=np.unique(np.concatenate([3*left,3*right,3*np.arange(len(points))+1,[3*zpin+2]]))
free=np.setdiff1d(np.arange(K.shape[0]),fixed);u=np.zeros(K.shape[0]);a=1e-4;delta=a*old.L;u[3*right]=delta
rhs=-(K@u)[free];solve=time.perf_counter();u[free]=spsolve(K[free][:,free].tocsc(),rhs);seconds=time.perf_counter()-solve
res=K@u;relative=float(np.linalg.norm(res[free])/np.linalg.norm(rhs));F=float(res[3*right].sum());U=float(.5*u@res)
volume=float(np.sum(scale*w)*old.L**3);Ksample=old.E/(1-old.NU**2)*volume/old.L**2
section=old.continuous_section(1.25);Kdense=old.E/(1-old.NU**2)*section['area_mm2']/old.L
Kbinary=old.E/(1-old.NU**2)*old.B*old.T/old.L
grad=np.einsum('cni,cqnj->cqij',u.reshape(-1,3)[cells]/old.L,g,optimize=True)
eps=.5*(grad+grad.swapaxes(-1,-2));expected=-old.NU/(1-old.NU)*a
ezzerr=float(np.max(abs(eps[:,:,2,2]-expected))/abs(expected))
checks={'constitutive_initial_tangent_error_le_1e-11':tangent_error<1e-11,
 'true_free_relative_residual_le_1e-8':relative<=1e-8,'reaction_energy_work_le_1e-8':abs(U/(.5*F*delta)-1)<1e-8,
 'sampled_plane_stress_stiffness_le_1e-8':abs((F/delta)/Ksample-1)<1e-8,
 'thickness_Poisson_contraction_relative_le_1e-6':ezzerr<=1e-6,
 'finite':bool(np.isfinite(u).all()),'solve_budget':seconds<=900}
result={'status':'ok' if all(checks.values()) else 'not_accepted','checks':checks,'case':'single flat membrane, centered phase_layer',
 'geometry_mm':{'length':old.L,'width':old.B,'background_height':old.H,'thickness':old.T,'midplane_z':1.25},
 'conditions':'ux=0 and 0.001 mm on x end faces; uy=0 everywhere (in-plane lateral strain held zero); uz free except one translation pin. This is a patch diagnostic, not TPMS XYZ boundary substitution.',
 'K_solved_N_per_mm':F/delta,'K_actual_Gauss_analytic_N_per_mm':Ksample,
 'K_dense_same_field_N_per_mm':Kdense,'K_binary_plate_N_per_mm':Kbinary,
 'solved_over_sampled_analytic_minus_one':(F/delta)/Ksample-1,
 'solved_over_binary_plate_minus_one':(F/delta)/Kbinary-1,
 'expected_thickness_strain':expected,'thickness_strain_relative_error':ezzerr,
 'true_free_relative_residual':relative,'energy_N_mm':U,'force_N':F,
 'weighted_volume_mm3':volume,'preparation_seconds':solve-start,'solve_seconds':seconds,
 'wall_seconds':time.perf_counter()-start,'full_TPMS_normal_relaxation_certified':False,
 'causal_reason_for_TPMS_5pct_bias_identified':False}
np.savez_compressed(D/'membrane_state.npz',u_mm=u.reshape(-1,3));write('result.json',result);freeze();print(json.dumps(result,indent=2))
if result['status']!='ok':sys.exit(2)

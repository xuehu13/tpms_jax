"""One original-HRZ generalized Ritz diagnostic; reuse stiffness, no trajectory."""
from pathlib import Path
import os,sys,json,ast,time,gc
os.environ['JAX_PLATFORMS']='cuda,cpu';os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['OMP_NUM_THREADS']='4'
R=Path('/home/xuehu/projects/tpms_jax');sys.path.insert(0,str(R))
import numpy as np
import jax
import jax.numpy as jnp
from scipy import sparse
from scipy.sparse.linalg import eigsh,splu,LinearOperator,ArpackNoConvergence
from hyperelastic_fem import make_density_hyperelastic_problem
jax.config.update('jax_enable_x64',True)
D=R/'validation/critical_mode_20261007_r21';O=D/'physical_metric';start=time.perf_counter()
protocol=json.loads((O/'protocol.json').read_text());old=R/'validation/geometry_transfer_20261006_r15'
# Reuse only pure coarse interpolation helpers and fixed basis setup, not the original global diagnostic loop.
t=(D/'diagnose.py').read_text();cut=t.index('spaces={};checks=[]')
t=t[:cut];t=t.replace("assert not (D/'results/result.json').exists() and not (D/'results/partial_result.json').exists()",'None')
ns={};exec(compile(t,str(D/'diagnose.py'),'exec'),ns)
P=ns['prolongation'](8);P.eliminate_zeros();mass=ns['mass'];rho=ns['rho'];weights=ns['weights'];cid=ns['cid'];grads=ns['grads'];scale=ns['scale'];interpolate=ns['interpolate']
assert np.max(np.abs(P.sum(axis=1).A.ravel()-1))<1e-12
Ms=(P.T@P.multiply(mass[:,None])).tocsr();Ms=(Ms+Ms.T)*.5;Ms=Ms[1:,1:]
assert np.all(Ms.diagonal()>0)
rng=np.random.default_rng(2121)
for _ in range(3):
 x=rng.normal(size=Ms.shape[0]);assert float(x@(Ms@x))>0
M=sparse.kron(Ms,sparse.eye(3),format='csr');lu=splu(Ms.tocsc())
Minv=LinearOperator(M.shape,matvec=lambda x:lu.solve(np.asarray(x).reshape(-1,3)).ravel(),dtype=np.float64)
x=rng.normal(size=M.shape[0]);inv_error=float(np.linalg.norm(M@Minv.matvec(x)-x)/np.linalg.norm(x));assert inv_error<1e-9
sparse.save_npz(O/'restricted_HRZ_mass.npz',M)
def emit(x):print(json.dumps(x),flush=True)
emit({'phase':'mass','seconds':time.perf_counter()-start,'inverse_residual':inv_error,'dofs':M.shape[0]})
p=ns['p'];stress_jvp=jax.jit(jax.vmap(lambda f,s,df:jax.jvp(lambda y:p.material_stress(y,s),(f,),(df,))[1]))
cuda_cid=jnp.asarray(cid);cuda_grads=jnp.asarray(grads)
form=jax.jit(lambda q,a:jnp.eye(3)+jnp.diag(jnp.array([0.,0.,-a]))+jnp.einsum('cni,qnj->cqij',q[cuda_cid],cuda_grads))
rows=[];S=R/'validation/step_control_20261007_r18/controlled_forward'
for name in protocol['states']:
 K=sparse.load_npz(D/'results'/f'{name[:-4]}_N8_K.npz');beg=time.perf_counter();converged=True;failure=None
 try:vals,vec=eigsh(K,M=M,Minv=Minv,k=6,which='SA',ncv=60,tol=1e-8,maxiter=400,v0=rng.normal(size=K.shape[0]))
 except ArpackNoConvergence as e:vals=e.eigenvalues;vec=e.eigenvectors;converged=False;failure=str(e)
 order=np.argsort(vals);vals=vals[order];vec=vec[:,order]
 with np.load(S/name) as f:q=f['q'];t=float(f['time'])
 s=np.clip(t/.004,0,1);a=float(.2*(10*s**3-15*s**4+6*s**5))
 residuals=[float(np.linalg.norm(K@vec[:,i]-vals[i]*(M@vec[:,i]))/max(np.linalg.norm(K@vec[:,i]),abs(vals[i])*np.linalg.norm(M@vec[:,i]),1e-30)) for i in range(len(vals))]
 orth=float(np.max(np.abs(vec.T@(M@vec)-np.eye(len(vals))))) if len(vals) else None
 row={'state':name,'compression':a,'converged':converged,'failure':failure,'eigenvalues_s_minus2':vals.tolist(),'generalized_relative_residuals':residuals,'mass_orthogonality_error':orth,'eigen_seconds':time.perf_counter()-beg,'modes':[]}
 np.savez_compressed(O/f'{name[:-4]}_modes.npz',eigenvalues=vals,vectors=vec)
 if len(vals):
  qc=np.zeros((4096,3));qc[1:]=vec[:,0].reshape(-1,3);mode=P@qc
  surface_mode,_=interpolate(mode);factor=1/float(np.linalg.norm(surface_mode,axis=1).max());mode*=factor;surface_mode*=factor
  F=np.asarray(form(jnp.asarray(q),jnp.asarray(a)));df=np.asarray(jax.jit(lambda u:jnp.einsum('cni,qnj->cqij',u[cuda_cid],cuda_grads))(jnp.asarray(mode)))
  out=[]
  for first in range(0,len(cid),1024):
   fs=jnp.asarray(F[first:first+1024].reshape(-1,3,3));ds=jnp.asarray(df[first:first+1024].reshape(-1,3,3))
   dp=np.asarray(stress_jvp(fs,jnp.asarray(scale[first:first+1024].ravel()),ds)).reshape(-1,27,3,3)
   out.append(np.sum(df[first:first+1024]*dp,axis=(-2,-1)))
  density=np.concatenate(out)
  parts={key:float(np.sum(density*weights*mask)*1000) for key,mask in {'deep_void':rho<=.001,'mixed_tail':(rho>.001)&(rho<.01),'main_interface':(rho>=.01)&(rho<.5),'geometric_core':rho>=.5}.items()}
  total=sum(parts.values());direct=float((vec[:,0]@(K@vec[:,0]))*factor**2*1000);err=abs(total-direct)/max(abs(total),abs(direct),1e-12);assert err<1e-6
  # Displacement participation is evaluated at all original Gauss points.
  delta=np.einsum('qn,cni->cqi',np.asarray(p.fe.shape_vals),mode[cid]);d2=np.sum(delta**2,axis=-1)
  part=float(np.sum(d2*weights*rho)/np.sum(d2*weights))
  record={'rank':0,'curvature_for_1mm_max_surface_mode_N_mm':total,'parts_N_mm':parts,'Ritz_direct_relative_error':err,'occupancy_weighted_displacement_fraction':part,'surface_RMS_mm':float(np.sqrt(np.sum(ns['aw']*np.sum(surface_mode**2,axis=1))/ns['aw'].sum())),'full_background_max_mode_mm':float(np.linalg.norm(mode,axis=1).max()*10)}
  row['modes'].append(record)
  np.savez_compressed(O/f'{name[:-4]}_selected.npz',mode=mode,surface_mode_mm=surface_mode,surface=ns['surface'],triangles=ns['triangles'])
 rows.append(row);(O/'partial_result.json').write_text(json.dumps({'protocol':protocol,'rows':rows,'mass_inverse_residual':inv_error,'wall_seconds':time.perf_counter()-start},indent=2,allow_nan=False))
 emit({'phase':'state','state':name,'seconds':time.perf_counter()-beg,'converged':converged,'eigenvalues':vals.tolist(),'mode':row['modes']})
(O/'result.json').write_text(json.dumps({'protocol':protocol,'rows':rows,'mass_inverse_residual':inv_error,'wall_seconds':time.perf_counter()-start,'no_new_forward_no_design_AD':True,'not_full_space_or_static_critical_certification':True},indent=2,allow_nan=False))
emit({'phase':'complete','seconds':time.perf_counter()-start})

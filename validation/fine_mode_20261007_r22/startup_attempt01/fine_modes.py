"""One saved-state matrix-free fine mechanical diagnostic; no time integration."""
from pathlib import Path
import os,sys,json,time,gc
os.environ['JAX_PLATFORMS']='cuda,cpu';os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['OMP_NUM_THREADS']='4'
R=Path('/home/xuehu/projects/tpms_jax');sys.path.insert(0,str(R))
import numpy as np,jax,jax.numpy as jnp
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import lobpcg
jax.config.update('jax_enable_x64',True)
D=R/'validation/fine_mode_20261007_r22';old=R/'validation/critical_mode_20261007_r21'
protocol=json.loads((D/'protocol.json').read_text());started=time.perf_counter();history=[];calls=0
def write(name,x):(D/name).write_text(json.dumps(x,indent=2,allow_nan=False))
def emit(x):print(json.dumps(x),flush=True)
def guard():
 if time.perf_counter()-started>protocol['budget_seconds']-20:raise TimeoutError('Single diagnostic budget; no automatic retry')
source=(old/'diagnose.py').read_text();source=source[:source.index('spaces={};checks=[]')]
source=source.replace("assert not (D/'results/result.json').exists() and not (D/'results/partial_result.json').exists()",'None')
ns={};exec(compile(source,str(old/'diagnose.py'),'exec'),ns)
p=ns['p'];cid=ns['cid'];grads=ns['grads'];weights=ns['weights'];rho=ns['rho'];scale=ns['scale'];mass=ns['mass'];interp=ns['interpolate']
assert len(mass)==64**3 and np.all(mass>0)
with np.load(R/protocol['state']) as f:q=f['q'];t=float(f['time'])
s=np.clip(t/.004,0,1);a=float(.2*(10*s**3-15*s**4+6*s**5))
cj=jnp.asarray(cid);gj=jnp.asarray(grads);wj=jnp.asarray(weights)
form=jax.jit(lambda q:jnp.eye(3)+jnp.diag(jnp.array([0.,0.,-a]))+jnp.einsum('cni,qnj->cqij',q[cj],gj))
F=np.asarray(form(jnp.asarray(q)));J=np.linalg.det(F)
assert np.isfinite(F).all() and np.all(J[rho>=.01]>0)
tangent=jax.jit(jax.vmap(lambda f,s,d:jax.jvp(lambda y:p.material_stress(y,s),(f,),(d,))[1]))
H=np.empty((len(cid),27,3,3,3,3))
for k in range(3):
 for l in range(3):
  unit=np.zeros((3,3));unit[k,l]=1.
  for first in range(0,len(cid),1024):
   guard();last=min(first+1024,len(cid));fs=jnp.asarray(F[first:last].reshape(-1,3,3))
   H[first:last,:,:,:,k,l]=np.asarray(tangent(fs,jnp.asarray(scale[first:last].ravel()),jnp.broadcast_to(jnp.asarray(unit),fs.shape))).reshape(last-first,27,3,3)
  emit({'phase':'material_column','column':[k,l],'seconds':time.perf_counter()-started})
H=H.reshape(len(cid),27,9,9);assert np.isfinite(H).all()
picks=np.linspace(0,rho.size-1,16,dtype=int)
ref=np.asarray(jax.jit(jax.vmap(jax.jacfwd(p.material_stress,argnums=0)))(jnp.asarray(F.reshape(-1,3,3)[picks]),jnp.asarray(scale.ravel()[picks]))).reshape(-1,9,9)
herr=float(np.max(np.abs(H.reshape(-1,9,9)[picks]-ref))/max(np.abs(ref).max(),1.));sym=float(np.linalg.norm(H-H.swapaxes(-1,-2))/np.linalg.norm(H))
assert herr<1e-11 and sym<1e-11
# Positive row-bound diagonal only preconditions iteration; the operator keeps signed H.
bound=np.max(np.sum(np.abs(H),axis=-1),axis=-1)
diagonal=np.zeros(len(mass));local=np.einsum('cq,qn->cn',weights*bound,np.sum(grads**2,axis=-1))
np.add.at(diagonal,cid.ravel(),local.ravel());assert np.all(diagonal>0)
sm=np.repeat(np.sqrt(mass[1:]),3);major=np.repeat(diagonal[1:]/mass[1:],3)
hj=jnp.asarray(H);del H,ref,bound,local;gc.collect()
@jax.jit
def Kaction(v):
 df=jnp.einsum('cnib,qnj->cqijb',v[cj],gj)
 dp=jnp.einsum('cqrs,cqsb->cqrb',hj,df.reshape(len(cid),27,9,-1)).reshape(df.shape)
 nodal=jnp.einsum('cqijb,qnj,cq->cnib',dp,gj,wj)
 return jnp.zeros(v.shape).at[cj.ravel()].add(nodal.reshape(-1,3,v.shape[-1]))
def apply(x):
 global calls
 guard();x=np.asarray(x);x=x[:,None] if x.ndim==1 else x
 v=np.zeros((len(mass),3,x.shape[1]));v[1:]=(x/sm[:,None]).reshape(-1,3,x.shape[1])
 y=np.asarray(Kaction(jnp.asarray(v)))[1:].reshape(len(sm),-1)/sm[:,None]
 assert np.isfinite(y).all();calls+=1
 if calls%10==0:emit({'phase':'operator','calls':calls,'columns':x.shape[1],'seconds':time.perf_counter()-started})
 return y
P=ns['prolongation'](8)
with np.load(old/'physical_metric/accepted_a0.1200_modes.npz') as f:vc=f['vectors'][:,:4];initial_eigen=f['eigenvalues'][:4]
qc=np.zeros((4096,3,4));qc[1:]=vc.reshape(-1,3,4)
u=(P@qc.reshape(4096,12)).reshape(len(mass),3,4);X=u[1:].reshape(len(sm),4)*sm[:,None]
AX=apply(X);gram=X.T@X;projected=X.T@AX
Mr=sparse.load_npz(old/'physical_metric/restricted_HRZ_mass.npz');Kr=sparse.load_npz(old/'results/accepted_a0.1200_N8_K.npz')
oldgram=vc.T@(Mr@vc);oldprojected=vc.T@(Kr@vc)
gerr=float(np.linalg.norm(gram-oldgram)/np.linalg.norm(oldgram));kerr=float(np.linalg.norm(projected-oldprojected)/np.linalg.norm(oldprojected))
assert gerr<1e-11 and kerr<1e-9
write('operator_checks.json',{'same_original_mass_relative_error':gerr,'projected_action_vs_original_N8_K_relative_error':kerr,'H_JVP_vs_jacfwd_relative_error':herr,'material_H_asymmetry':sym,'free_dofs':len(sm),'initial_eigenvalues_s_minus2':initial_eigen.tolist(),'original_positive_J_min':float(J[rho>=.01].min()),'preconditioner_min_max':[float(major.min()),float(major.max())],'preconditioner_does_not_change_operator':True})
gv,gq=eigh(gram);X=X@(gq/np.sqrt(gv)[None,:])@gq.T
tol=1e-3*float(np.min(np.abs(initial_eigen)))
np.savez_compressed(D/'initial_block.npz',mass_whitened_vectors=X,original_initial_eigenvalues=initial_eigen)
emit({'phase':'fine_refinement','absolute_solver_tol':tol,'seconds':time.perf_counter()-started})
eigen,V,lh,rh=lobpcg(apply,X,M=lambda x:x/major[:,None],largest=False,maxiter=200,tol=tol,verbosityLevel=1,retLambdaHistory=True,retResidualNormsHistory=True)
AV=apply(V);r=AV-V*eigen[None,:]
whitened=np.linalg.norm(r,axis=0)/np.maximum(np.linalg.norm(AV,axis=0),np.abs(eigen)*np.linalg.norm(V,axis=0))
fine=np.linalg.norm(sm[:,None]*r,axis=0)/np.maximum(np.linalg.norm(sm[:,None]*AV,axis=0),np.linalg.norm(sm[:,None]*V*eigen[None,:],axis=0))
original=np.zeros((len(mass),3,4));original[1:]=(V/sm[:,None]).reshape(-1,3,4)
mean=np.sum(original*mass[:,None,None],axis=0)/mass.sum();trans=mass.sum()*np.sum(mean**2,axis=0)/np.sum(original**2*mass[:,None,None],axis=(0,1))
np.savez_compressed(D/'refined_modes.npz',mass_whitened_vectors=V,original_periodic_modes=original,eigenvalues=eigen,mass_whitened_residual=whitened,original_fine_residual=fine,uniform_translation_inertia_fraction=trans)
write('iteration_history.json',{'eigenvalues':[np.asarray(x).tolist() for x in lh],'absolute_residual_norms':[np.asarray(x).tolist() for x in rh],'operator_calls':calls})
records=[]
for i in range(4):
 mode=original[:,:,i];sw,_=interp(mode);factor=1/np.linalg.norm(sw,axis=1).max();mode=mode*factor;sw=sw*factor
 df=np.einsum('cni,qnj->cqij',mode[cid],grads);parts={key:0. for key in ['deep_void','mixed_tail','main_interface','geometric_core']}
 for first in range(0,len(cid),1024):
  guard();last=min(first+1024,len(cid));dp=np.asarray(tangent(jnp.asarray(F[first:last].reshape(-1,3,3)),jnp.asarray(scale[first:last].ravel()),jnp.asarray(df[first:last].reshape(-1,3,3)))).reshape(last-first,27,3,3)
  density=np.sum(df[first:last]*dp,axis=(-2,-1))*weights[first:last]*1000;rr=rho[first:last]
  for key,mask in {'deep_void':rr<=.001,'mixed_tail':(rr>.001)&(rr<.01),'main_interface':(rr>=.01)&(rr<.5),'geometric_core':rr>=.5}.items():parts[key]+=float(density[mask].sum())
 curv=sum(parts.values());expected=float(V[:,i]@AV[:,i])*factor**2*1000;err=abs(curv-expected)/max(abs(curv),abs(expected),1e-12);assert err<1e-6
 delta=np.einsum('qn,cni->cqi',np.asarray(p.fe.shape_vals),mode[cid]);d2=np.sum(delta**2,axis=-1)
 record={'rank_lowest_block':i,'eigenvalue_s_minus2':float(eigen[i]),'mass_whitened_relative_residual':float(whitened[i]),'original_fine_relative_residual':float(fine[i]),'uniform_translation_inertia_fraction':float(trans[i]),'mode_converged':bool(whitened[i]<=1e-3 and fine[i]<=1e-3),'curvature_for_1mm_max_surface_mode_N_mm':curv,'parts_N_mm':parts,'direct_curvature_relative_error':err,'occupancy_weighted_displacement_fraction':float(np.sum(d2*weights*rho)/np.sum(d2*weights))}
 records.append(record);np.savez_compressed(D/f'mode{i}_surface.npz',surface_mode_mm=sw,surface=ns['surface'],triangles=ns['triangles'])
write('result.json',{'compression':a,'rows':records,'operator_calls':calls,'wall_seconds':time.perf_counter()-started,'solver_history_entries':len(lh),'same_original_pin_and_domains':True,'new_time_advance':False,'new_design_AD':False,'not_a_static_or_global_stability_certification':True})
emit({'phase':'complete','rows':records,'seconds':time.perf_counter()-started})

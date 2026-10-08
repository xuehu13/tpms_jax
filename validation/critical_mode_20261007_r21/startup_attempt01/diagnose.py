"""Restricted periodic mechanical eigendiagnostic, never a time integrator."""
from pathlib import Path
import os, sys, json, time, ast, gc
os.environ['JAX_PLATFORMS']='cpu'
os.environ['OPENBLAS_NUM_THREADS']='4'
R=Path('/home/xuehu/projects/tpms_jax');sys.path.insert(0,str(R))
import numpy as np
import jax
import jax.numpy as jnp
import basix
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh, ArpackNoConvergence
from threadpoolctl import threadpool_limits
from jax_fem.basis import get_elements
from hyperelastic_fem import make_density_hyperelastic_problem
jax.config.update('jax_enable_x64',True)
threadpool_limits(limits=4)
D=R/'validation/critical_mode_20261007_r21';D.joinpath('results').mkdir(exist_ok=False)
protocol=json.loads((D/'protocol.json').read_text());started=time.perf_counter()
S=R/'validation/step_control_20261007_r18/controlled_forward'
old=R/'validation/geometry_transfer_20261006_r15'
def emit(value):print(json.dumps(value),flush=True)
def write(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')
def guard():
 if time.perf_counter()-started>1500:raise TimeoutError('Recorded 25min diagnostic budget')
p=make_density_hyperelastic_problem(1,rho_quad=1.,eta=1e-4,periodic_axes=(0,1,2),element_degree=2,quadrature_order=4,material_model='objective_void')
N=32;levels=65;grads=np.asarray(p.fe.shape_grads[0])*N
family,cell,_,_,degree,order=get_elements('HEX27')
local=np.rint(2*basix.create_element(family,cell,degree).points[order]).astype(int)
origins=2*np.indices((N,)*3).reshape(3,-1).T
indices=origins[:,None,:]+local[None,:,:]
nodes=(indices[...,0]*levels+indices[...,1])*levels+indices[...,2]
with np.load(old/'hrz_mass.npz') as f:ids=f['class_ids'];mass=f['periodic_class_mass_normalized']
with np.load(old/'gauss_field.npz') as f:rho=f['rho'];weights=f['JxW'];points=f['physical_quad_points']
cid=ids[nodes];scale=1e-4+(1-1e-4)*rho
fine_coordinates=np.indices((64,)*3).reshape(3,-1).T/64
assert np.array_equal(ids,np.ravel_multi_index((np.indices((65,)*3).reshape(3,-1).T%64).T,(64,)*3))
assert np.max(np.abs(origins[:,None,:]/64+np.asarray(p.physical_quad_points)[0][None,:,:]/N-points))<5e-14
with np.load(old/'surface_geometry.npz') as f:surface=f['points'];triangles=f['triangles']
area=np.linalg.norm(np.cross(surface[triangles[:,1]]-surface[triangles[:,0]],surface[triangles[:,2]]-surface[triangles[:,0]]),axis=1)/2
aw=np.zeros(len(surface));np.add.at(aw,triangles.ravel(),np.repeat(area/3,3))
source=R/'validation/mechanism_20261007_r17/analyze_localization.py'
tree=ast.parse(source.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='interpolate')
ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),ns);interpolate=ns['interpolate']
def shape(x):return np.stack((2*x*x-3*x+1,4*x-4*x*x,2*x*x-x),axis=-1)
def dshape(x):return np.stack((4*x-3,4-8*x,4*x-1),axis=-1)
lex=np.indices((3,)*3).reshape(3,-1).T
def coarse_basis(x,nc):
 L=shape(x);DL=dshape(x);B=np.prod(np.stack([L[:,j,lex[:,j]] for j in range(3)],axis=-1),axis=-1)
 G=np.stack([DL[:,j,lex[:,j]]*np.prod(np.stack([L[:,k,lex[:,k]] for k in range(3) if k!=j],axis=-1),axis=-1)*nc for j in range(3)],axis=-1)
 return B,G
def prolongation(nc):
 c=np.minimum(np.floor(fine_coordinates*nc).astype(int),nc-1);x=fine_coordinates*nc-c;B,_=coarse_basis(x,nc)
 xyz=(2*c[:,None,:]+lex[None,:,:])%(2*nc)
 columns=np.ravel_multi_index(xyz.reshape(-1,3).T,(2*nc,)*3).reshape(-1,27)
 P=sparse.coo_matrix((B.ravel(),(np.repeat(np.arange(len(fine_coordinates)),27),columns.ravel())),shape=(64**3,(2*nc)**3)).tocsr()
 return P
spaces={};checks=[]
rng=np.random.default_rng(2107)
for nc in [4,8]:
 P=prolongation(nc);qtest=rng.normal(size=((2*nc)**3,3));qtest-=qtest[0]
 coarsecell=np.floor(points[:,0,:]*nc).astype(int);coarsecell=np.minimum(coarsecell,nc-1)
 groups=np.ravel_multi_index(coarsecell.T,(nc,)*3)
 cells_xyz=np.indices((nc,)*3).reshape(3,-1).T
 coarsecid=np.ravel_multi_index(((2*cells_xyz[:,None,:]+lex[None,:,:])%(2*nc)).reshape(-1,3).T,(2*nc,)*3).reshape(-1,27)
 sample=np.linspace(0,len(cid)-1,128,dtype=int)
 _,G=coarse_basis((points[sample]*nc-coarsecell[sample,None,:]).reshape(-1,3),nc)
 gf=np.einsum('cni,qnj->cqij',(P@qtest)[cid[sample]],grads)
 gc_=np.einsum('cni,cqnj->cqij',qtest[coarsecid[groups[sample]]],G.reshape(len(sample),27,27,3))
 err=float(np.max(np.abs(gf-gc_)));assert err<2e-12
 checks.append({'Ncoarse':nc,'prolongation_gradient_max_error':err,'P_partition_unity_error':float(np.max(np.abs(P.sum(axis=1).A.ravel()-1)))})
 spaces[nc]=(P,coarsecell,groups,coarsecid)
write(D/'basis_checks.json',checks);emit({'phase':'basis_checked','checks':checks})
tangent=jax.jit(jax.vmap(jax.jacfwd(p.material_stress,argnums=0)))
rows=[]
for name in protocol['states']:
 guard();st=time.perf_counter()
 with np.load(S/name) as f:q=f['q'];t=float(f['time'])
 ss=np.clip(t/.004,0,1);a=float(.2*(10*ss**3-15*ss**4+6*ss**5))
 F=np.eye(3)+np.diag([0.,0.,-a])+np.einsum('cni,qnj->cqij',q[cid],grads)
 J=np.linalg.det(F);assert np.isfinite(F).all() and np.all(J[rho>=.01]>0)
 H=np.empty((len(cid),27,3,3,3,3))
 for first in range(0,len(cid),1024):
  guard();last=min(first+1024,len(cid))
  H[first:last]=np.asarray(tangent(jnp.asarray(F[first:last].reshape(-1,3,3)),jnp.asarray(scale[first:last].ravel()))).reshape(last-first,27,3,3,3,3)
 assert np.isfinite(H).all()
 emit({'phase':'tangents','state':name,'seconds':time.perf_counter()-st})
 for nc in [4,8]:
  guard();bt=time.perf_counter();P,ccell,groups,coarsecid=spaces[nc];element_blocks=[]
  for c in range(nc**3):
   guard();take=np.flatnonzero(groups==c)
   _,G=coarse_basis((points[take]*nc-ccell[take,None,:]).reshape(-1,3),nc)
   hc=H[take].reshape(-1,3,3,3,3);w=weights[take].ravel()
   # Contract the original full-domain material Hessian, never a new constitutive law.
   kg=np.einsum('qaj,qijkl,qbl,q->aibk',G,hc,G,w,optimize=True).reshape(81,81)
   element_blocks.append(kg)
  blocks=np.stack(element_blocks);dofs=(coarsecid[:,:,None]*3+np.arange(3)).reshape(-1,81)
  K=sparse.coo_matrix((blocks.ravel(),(np.broadcast_to(dofs[:,:,None],blocks.shape).ravel(),np.broadcast_to(dofs[:,None,:],blocks.shape).ravel())),shape=(3*(2*nc)**3,)*2).tocsr()
  asym=float(sparse.linalg.norm(K-K.T)/sparse.linalg.norm(K));assert asym<1e-11
  K=(K+K.T)*.5;free=K[3:,3:];sparse.save_npz(D/'results'/f'{name[:-4]}_N{nc}_K.npz',free)
  emit({'phase':'assembled','state':name,'Ncoarse':nc,'seconds':time.perf_counter()-bt,'nnz':free.nnz})
  et=time.perf_counter();converged=True;failure=None
  if nc==4:vals,vec=eigh(free.toarray(),subset_by_index=(0,5),driver='evr')
  else:
   try:vals,vec=eigsh(free,k=6,which='SA',ncv=60,tol=1e-8,maxiter=400,v0=rng.normal(size=free.shape[0]))
   except ArpackNoConvergence as e:vals=e.eigenvalues;vec=e.eigenvectors;converged=False;failure=str(e)
  order_=np.argsort(vals);vals=vals[order_];vec=vec[:,order_]
  residuals=[float(np.linalg.norm(free@vec[:,i]-vals[i]*vec[:,i])/max(np.linalg.norm(free@vec[:,i]),abs(vals[i]),1e-30)) for i in range(len(vals))]
  np.savez_compressed(D/'results'/f'{name[:-4]}_N{nc}_modes.npz',eigenvalues=vals,vectors=vec)
  row={'state':name,'compression':a,'Ncoarse':nc,'free_dofs':free.shape[0],'basis_eigenvalues':vals.tolist(),'eigenpair_relative_residuals':residuals,'converged':converged,'failure':failure,'relative_asymmetry':asym,'assembly_seconds':et-bt,'eigen_seconds':time.perf_counter()-et,'minimum_NH_J':float(J[rho>=.01].min()),'modes':[]}
  for i in range(len(vals)):
   qc=np.zeros((free.shape[0]+3,));qc[3:]=vec[:,i];qc=qc.reshape(-1,3)
   mode=P@qc;mode-=np.sum(mode*mass[:,None],axis=0)/mass.sum()
   surface_mode=interpolate(surface,mode[ids],32,2)
   surface_mode-=np.sum(surface_mode*aw[:,None],axis=0)/aw.sum()
   maximum=float(np.linalg.norm(surface_mode,axis=1).max())
   assert maximum>0
   factor=1/(10*maximum);mode*=factor;surface_mode*=factor*10
   df=np.einsum('cni,qnj->cqij',mode[cid],grads)
   density=np.einsum('cqij,cqijkl,cqkl->cq',df,H,df,optimize=True)
   parts={key:float(np.sum(density*weights*mask)*1000) for key,mask in {'deep_void':rho<=.001,'mixed_tail':(rho>.001)&(rho<.01),'main_interface':(rho>=.01)&(rho<.5),'geometric_core':rho>=.5}.items()}
   total=sum(parts.values());matrix=float(vals[i]*factor**2*1000)
   relative=abs(total-matrix)/max(abs(total),abs(matrix),1e-12);assert relative<1e-6
   # Fine-space mechanical residual: Ritz stationarity need not imply fine-space stationarity.
   dp=np.einsum('cqijkl,cqkl->cqij',H,df,optimize=True)
   elemental=np.einsum('cqij,qnj,cq->cni',dp,grads,weights,optimize=True)
   full_grad=np.zeros((len(mass),3));np.add.at(full_grad,cid.ravel(),elemental.reshape(-1,3))
   fineq=mode-mode[0];finegrad=full_grad[1:]
   projected=(P.T@full_grad)[1:];qcpin=qc[1:]*factor
   projected_res=float(np.linalg.norm(projected-vals[i]*qcpin)/max(np.linalg.norm(projected),abs(vals[i])*np.linalg.norm(qcpin),1e-30))
   # Norm-ratio describes force components outside this coarse stationarity subspace, not a certified full eigen residual.
   gp=np.zeros_like(qc);gp[1:]=projected
   denom=float(np.sum(fineq[1:]**2));rq=float(np.sum(fineq[1:]*finegrad)/denom)
   fine_res=float(np.linalg.norm(finegrad-rq*fineq[1:])/max(np.linalg.norm(finegrad),abs(rq)*np.linalg.norm(fineq[1:]),1e-30))
   sample_u=interpolate(surface,(fineq)[ids],32,2);sample_u-=np.sum(sample_u*aw[:,None],axis=0)/aw.sum()
   delta=np.einsum('qn,cni->cqi',np.asarray(p.fe.shape_vals),mode[cid])
   dis2=np.sum(delta**2,axis=-1)
   material_part=float(np.sum(dis2*weights*rho)/np.sum(dis2*weights))
   record={'rank':i,'curvature_for_1mm_max_surface_mode_N_mm':total,'parts_N_mm':parts,'Ritz_direct_relative_error':relative,'projected_residual':projected_res,'fine_Euclidean_eigen_residual':fine_res,'occupancy_weighted_displacement_fraction':material_part,'surface_RMS_mm':float(np.sqrt(np.sum(aw*np.sum(surface_mode**2,axis=1))/aw.sum())),'full_background_max_mode_mm':float(np.linalg.norm(mode,axis=1).max()*10)}
   row['modes'].append(record)
   if i==0:np.savez_compressed(D/'results'/f'{name[:-4]}_N{nc}_selected.npz',mode=fineq,surface_mode_mm=surface_mode,surface=surface,triangles=triangles,area_weights=aw)
  row['wall_seconds']=time.perf_counter()-bt;rows.append(row)
  write(D/'results/partial_result.json',{'protocol':protocol,'rows':rows,'elapsed_seconds':time.perf_counter()-started})
  emit({'phase':'eigen_done','state':name,'Ncoarse':nc,'seconds':row['wall_seconds'],'values':vals.tolist(),'converged':converged})
  del free,K,blocks;gc.collect()
 del H,F;gc.collect()
write(D/'results/result.json',{'protocol':protocol,'rows':rows,'wall_seconds':time.perf_counter()-started,'new_time_advance':False,'new_design_AD':False,'new_Abaqus_jobs':0,'no_full_space_or_static_stability_certification':True})
emit({'phase':'complete','seconds':time.perf_counter()-started,'rows':len(rows)})

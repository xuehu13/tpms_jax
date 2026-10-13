"""Frozen interface-width-only initial static diagnostic. Production unchanged."""
from pathlib import Path
import sys,os,time,json,hashlib
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
R=Path('/home/xuehu/projects/tpms_jax');D=Path(__file__).resolve().parent;sys.path.insert(0,str(R))
import numpy as np,jax,jax.numpy as jnp,basix
from scipy.special import expit
from hyperelastic_fem import make_density_hyperelastic_problem,MU,KAPPA
from scripts.thin_target_explicit import ExplicitXYZ
from surface_distance import PeriodicSurfaceDistance
from jax_fem.basis import get_elements
jax.config.update('jax_enable_x64',True)
L=10.;N=32;A=1e-4;ETA=1e-4
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,v):(D/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def freeze():
 for n,h in json.loads((D/'frozen_before.json').read_text()).items():assert sha(R/n)==h,n
def main():
 start=time.perf_counter();freeze();cfg=json.loads((D/'protocol.json').read_text())
 c=np.load(R/'validation/geometry_transfer_20261006_r15/gauss_field.npz')
 old=np.load(R/'validation/initial_tangent_20261008_r30/linear_state.npz')
 oldr=json.loads((R/'validation/initial_tangent_20261008_r30/jax_result.json').read_text())
 shell=cfg['shell_stiffness_N_per_mm'];selected=np.load(R/'validation/local_quadrature_20261008_r33/selected_cells.npy')
 p=make_density_hyperelastic_problem(N,eta=ETA,rho_quad=c['rho'],periodic_axes=(0,1,2),
  element_degree=2,quadrature_order=4,material_model='objective_void')
 ex=ExplicitXYZ(p,L=L);ids=jnp.asarray(np.asarray(p.class_ids)[np.asarray(p.fe.cells)],dtype=jnp.int32)
 assert np.array_equal(np.asarray(p.class_ids),old['class_ids'])
 assert np.max(abs(np.asarray(p.physical_quad_points)-c['physical_quad_points']))<1e-14
 g=ex.kernel_geometry[1];w=ex.kernel_geometry[2][0];lam=KAPPA-2*MU/3;eye=jnp.eye(3)
 H=jnp.zeros((3,3)).at[2,2].set(-A);qzero=jnp.zeros((ex.nc,3));init=jnp.asarray(old['q'])
 def project(x):return x-jnp.mean(x,axis=0,keepdims=True)
 def strain(q,h):
  grad=h+jnp.einsum('cni,qnj->cqij',q[ids],g)
  return grad,.5*(grad+grad.swapaxes(-1,-2))
 @jax.jit
 def force(q,h,s):
  ep=strain(q,h)[1]
  stress=s[:,:,None,None]*(lam*jnp.trace(ep,axis1=-2,axis2=-1)[:,:,None,None]*eye+2*MU*ep)
  fc=jnp.einsum('cqij,qnj,q->cni',stress,g,w)
  return jnp.zeros_like(q).at[ids.ravel()].add(fc.reshape(-1,3))
 @jax.jit
 def op(q,s):return project(force(q,jnp.zeros((3,3)),s))
 @jax.jit
 def diagonal(s):
  loc=MU*jnp.sum(g*g,axis=-1)[:,:,None]+(lam+MU)*g*g
  dc=jnp.einsum('cq,qni,q->cni',s,loc,w)
  return jnp.zeros_like(qzero).at[ids.ravel()].add(dc.reshape(-1,3))
 @jax.jit
 def metrics(q,s):
  grad,ep=strain(q,H);stress=s[:,:,None,None]*(lam*jnp.trace(ep,axis1=-2,axis2=-1)[:,:,None,None]*eye+2*MU*ep)
  return (.5*jnp.sum(stress*ep*w[None,:,None,None])*L**3,
   jnp.sum(stress[:,:,2,2]*w[None,:])*L**2,jnp.min(jnp.linalg.det(eye+grad)),jnp.max(abs(ep)))
 @jax.jit
 def chunk(state,s,diag,bn):
  def one(st,_):
   def step(st):
    x,r,z,v,rz,err,k=st;kv=op(v,s);den=jnp.vdot(v,kv);alpha=rz/den
    x=project(x+alpha*v);r=project(r-alpha*kv);z=project(r/diag);rz2=jnp.vdot(r,z)
    v=project(z+(rz2/rz)*v);err=jnp.linalg.norm(r)/bn
    err=jnp.where((den>0)&jnp.isfinite(err),err,jnp.nan)
    return x,r,z,v,rz2,err,k+1
   return jax.lax.cond((st[-2]>1e-10)&jnp.isfinite(st[-2])&(st[-1]<6000),step,lambda st:st,st),None
  return jax.lax.scan(one,state,None,length=50)[0]
 baseline_phi=expit((.25-c['distance']*L)/(.05/(2*np.log(9))))
 reproduction=float(np.max(abs(baseline_phi-c['rho'])))
 base_s=jnp.asarray(ETA+(1-ETA)*baseline_phi)
 u0,f0,_,_=[float(v) for v in metrics(init,base_s)]
 assert reproduction<1e-12 and abs(u0/oldr['energy_N_mm']-1)<1e-10
 assert abs(f0/oldr['Fz_N']-1)<1e-10
 prep=time.perf_counter()-start
 results={};states={.05:np.asarray(init)}
 for width in cfg['new_widths_mm']:
  tick=time.perf_counter();name='width_'+str(width).replace('.','p');out=D/name;out.mkdir(exist_ok=False)
  rho=expit((.25-c['distance']*L)/(width/(2*np.log(9))))
  scale=jnp.asarray(ETA+(1-ETA)*rho);diag=diagonal(scale);assert float(diag.min())>0
  b=-project(force(qzero,H,scale));bn=float(jnp.linalg.norm(b));x=project(init)
  r=b-op(x,scale);z=project(r/diag)
  state=(x,r,z,z,jnp.vdot(r,z),jnp.linalg.norm(r)/bn,jnp.array(0));history=[];stop='iteration_limit'
  while int(state[-1])<6000:
   if time.perf_counter()-start>cfg['total_budget_seconds']:stop='time_budget';break
   state=chunk(state,scale,diag,bn);state[0].block_until_ready();k=int(state[-1]);err=float(state[-2])
   history.append({'iteration':k,'recurrence_residual':err,'seconds':time.perf_counter()-tick})
   if k%500==0:print(json.dumps({'stage':name,**history[-1]}),flush=True)
   if not np.isfinite(err):stop='breakdown';break
   if err<=1e-10:stop='recurrence_tolerance';break
  q=state[0]-state[0][ex.pin];true=float(jnp.linalg.norm(force(q,H,scale).at[ex.pin].set(0.))/bn)
  U,F,Jmin,emax=[float(v) for v in metrics(q,scale)];work=.5*F*(-A*L)
  full=np.asarray(q)[np.asarray(p.class_ids)].reshape(65,65,65,3)
  gap=max(float(np.max(abs(np.take(full,0,axis=i)-np.take(full,-1,axis=i)))) for i in range(3))
  checks={'true_free_residual':true<=1e-8,'energy_work':abs(U/work-1)<=1e-6,'periodic_gap':gap<=1e-12,
   'finite':bool(np.isfinite(np.asarray(q)).all()),'small_strain':Jmin>.98 and emax<.02,
   'budget':time.perf_counter()-start<=cfg['total_budget_seconds']}
  K=F/(-A*L)
  row={'status':'ok' if all(checks.values()) else 'not_accepted','checks':checks,'interface_10_90_mm':width,
   'inverse_length_per_mm':2*np.log(9)/width,'K0_N_per_mm':K,'Fz_N':F,'U_N_mm':U,
   'relative_to_shell':K/shell-1,'relative_to_original':K/oldr['stiffness_N_per_mm']-1,
   'true_free_residual':true,'energy_work_error':abs(U/work-1),'periodic_gap':gap,
   'minimum_linearized_J':Jmin,'max_abs_strain':emax,'iterations':int(state[-1]),'stop':stop,
   'occupancy_volume_mm3':float(np.sum(rho*c['JxW'])*L**3),
   'weighted_volume_mm3':float(np.sum((ETA+(1-ETA)*rho)*c['JxW'])*L**3),'solve_and_metrics_seconds':time.perf_counter()-tick}
  (out/'result.json').write_text(json.dumps(row,indent=2));(out/'convergence.json').write_text(json.dumps(history,indent=2))
  np.savez_compressed(out/'linear_state.npz',q=np.asarray(q),class_ids=np.asarray(p.class_ids),H=np.asarray(H))
  results[width]=row;states[width]=np.asarray(q);print(json.dumps(row),flush=True)
  if row['status']!='ok':raise RuntimeError('Width state invalid; no additional iterations')
 # Original width/state is a frozen reuse, not a new solve.
 results[.05]={'status':'reused_valid_r30','interface_10_90_mm':.05,'inverse_length_per_mm':2*np.log(9)/.05,
  'K0_N_per_mm':oldr['stiffness_N_per_mm'],'U_N_mm':oldr['energy_N_mm'],'relative_to_shell':oldr['stiffness_N_per_mm']/shell-1,
  'relative_to_original':0.,'occupancy_volume_mm3':float(np.sum(baseline_phi*c['JxW'])*L**3),
  'weighted_volume_mm3':oldr['weighted_volume_mm3']}
 family,cell,_,_,degree,order=get_elements('HEX27');el=basix.create_element(family,cell,degree)
 surface=PeriodicSurfaceDistance(c['surface_vertices'],c['surface_triangles'])
 origins=np.asarray(p.fe.points)[np.asarray(p.fe.cells)[selected]].min(axis=1)
 patchids=np.asarray(ids)[selected];post={};poststart=time.perf_counter()
 for level in [8,12]:
  z,wg=np.polynomial.legendre.leggauss(level);z=(z+1)/2;wg/=2
  ref=np.array([[a,b,c] for a in z for b in z for c in z])
  wt=np.array([a*b*c for a in wg for b in wg for c in wg])*(L/N)**3
  gd=el.tabulate(1,ref)[1:,:,:,:][:,:,:,0].transpose(1,2,0)[:,order,:]*N
  totals={width:np.zeros(2) for width in [.025,.05,.1]}
  for pos in range(0,len(selected),32):
   if time.perf_counter()-start>cfg['total_budget_seconds']:raise TimeoutError('Frozen total budget exceeded')
   n=min(32,len(selected)-pos);distance=surface.query(origins[pos:pos+n,None,:]+ref[None,:,:]/N)*L
   for width in totals:
    uc=states[width][patchids[pos:pos+n]]
    grad=np.asarray(H)+np.einsum('cni,qnj->cqij',uc,gd,optimize=True);ep=.5*(grad+grad.swapaxes(-1,-2))
    unit=.5*lam*np.trace(ep,axis1=-2,axis2=-1)**2+MU*np.sum(ep*ep,axis=(-1,-2))
    phi=expit((.25-distance)/(width/(2*np.log(9))))
    totals[width]+=np.array([np.sum((ETA+(1-ETA)*phi)*unit*wt),np.sum(phi*wt)])
  post[level]=totals
 local=[]
 for width in [.025,.05,.1]:
  q=states[width];grad=np.asarray(H)+np.einsum('cni,qnj->cqij',q[patchids],np.asarray(g),optimize=True)
  ep=.5*(grad+grad.swapaxes(-1,-2));unit=.5*lam*np.trace(ep,axis1=-2,axis2=-1)**2+MU*np.sum(ep*ep,axis=(-1,-2))
  phi=expit((.25-c['distance'][selected]*L)/(width/(2*np.log(9))))
  orig=float(np.sum((ETA+(1-ETA)*phi)*unit*c['JxW'][selected])*L**3)
  a,b=post[8][width],post[12][width]
  local.append({'width_mm':width,'original_selected_energy_N_mm':orig,'dense8_energy_N_mm':float(a[0]),
   'dense12_energy_N_mm':float(b[0]),'dense12_relative_to_original_selected':float(b[0]/orig-1),
   'dense8_12_relative_energy':float(a[0]/b[0]-1),'dense8_12_relative_volume':float(a[1]/b[1]-1),
   'dense12_relative_change_scaled_by_own_full_energy':float((b[0]-orig)/results[width]['U_N_mm']),
   'stable_existing_levels':bool(max(abs(a/b-1))<=.01)})
 result={'status':'initial_width_diagnostic_complete','case':'diverse_04 N32 original 27-point static',
  'baseline_reproduction_max_occupancy_error':reproduction,'baseline_reused_r30':True,
  'cases':[results[x] for x in [.025,.05,.1]],'local_integration':local,'selected_cells':len(selected),
  'selected_region_outside_not_dense_checked':True,'dense_checks_are_fixed_state_not_new_global_K':True,
  'width_is_one_changed_factor':True,'production_changed':False,'new_Abaqus_job':False,'new_design_AD':False,
  'preparation_seconds':prep,'local_postprocess_seconds':time.perf_counter()-poststart,'wall_seconds':time.perf_counter()-start}
 write('result.json',result);freeze();print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()

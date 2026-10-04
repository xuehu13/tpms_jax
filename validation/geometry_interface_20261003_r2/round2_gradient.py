from pathlib import Path
import json,sys,time,resource,subprocess,hashlib,threading
R=Path('/home/xuehu/projects/tpms_jax');sys.path.insert(0,str(R));O=R/'validation/geometry_interface_20261003_r2';S=O/'step3';S.mkdir(exist_ok=True)
import numpy as np
import jax;import jax.numpy as jnp
jax.config.update('jax_log_compiles',True)
from voxel_field import ImplicitVoxelDesign,sample_implicit
from design_fem import GyroidDesign
N=int(sys.argv[1]);small=N==8
choice=json.loads((O/'step1/summary.json').read_text())['selected_M']
route='implicit' if choice else 'analytic';M=choice
D=S/f'N{N}';D.mkdir(exist_ok=False)
def rss():return int(Path('/proc/self/status').read_text().split('VmRSS:')[1].split()[0])/1024
def gpu():
 try:return subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip()
 except Exception:return 'unavailable'
start=time.perf_counter()
gpu_samples=[];stop_monitor=threading.Event()
def monitor():
 while not stop_monitor.is_set():
  value=gpu()
  try:gpu_samples.append({'seconds':time.perf_counter()-start,'used_MiB':int(value.split(',')[0]),'total_MiB':int(value.split(',')[1])})
  except (ValueError,IndexError):pass
  stop_monitor.wait(1.)
threading.Thread(target=monitor,daemon=True).start()
if choice:
 g=sample_implicit(M);theta=jnp.concatenate([g.ravel(),jnp.array([.541062])])
 x=(jnp.arange(M,dtype=jnp.float64)+.5)/M;xyz=jnp.stack(jnp.meshgrid(x,x,x,indexing='ij'),axis=-1)
 d1=jnp.zeros_like(theta).at[-1].set(1.)
 spatial=g*(1+.4*jnp.cos(2*jnp.pi*xyz[...,0]));d2=jnp.concatenate([spatial.ravel()/jnp.linalg.norm(spatial),jnp.zeros(1)])
else:
 theta=jnp.array([.541062,0.,0.,0.]);d1=jnp.array([1.,0.,0.,0.]);d2=jnp.array([0.,1.,0.,0.]);g=None
np.savez(D/'inputs.npz',theta=np.asarray(theta),direction_c=np.asarray(d1),direction_spatial=np.asarray(d2))
(D/'plan.json').write_text(json.dumps({'N':N,'M':M,'route':route,'directions':['unit c','unit L2 g*(1+.4*cos(2pi*x))' if choice else 'cos(2pi*x) width'],'steps':[.001,.0005],'fixed':True,'beta':40,'eta':1e-4,'max_forwards':9 if small else 1,'max_adjoint':1,'normalization':'unit discrete L2','input_sha256':hashlib.sha256((D/'inputs.npz').read_bytes()).hexdigest()},indent=2)+'\n')
kwargs={}
if not small:
 from petsc4py import PETSc
 for key,val in {'ksp_rtol':1e-11,'ksp_atol':1e-13,'ksp_max_it':5000,'ksp_error_if_not_converged':True}.items():PETSc.Options()[key]=val
 solver={'petsc_solver':{'ksp_type':'cg','pc_type':'gamg'},'tol':1e-12,'rel_tol':1e-10}
 kwargs={'solver_options':solver,'adjoint_options':{'petsc_solver':{'ksp_type':'cg','pc_type':'gamg'}}}
build=time.perf_counter();design=ImplicitVoxelDesign(N,M,**kwargs) if choice else GyroidDesign(N,beta=40,emin_ratio=1e-4,**kwargs);build_seconds=time.perf_counter()-build
memory={'built_rss_MiB':rss(),'gpu_built':gpu()}
t=time.perf_counter();(values,w),pullback=jax.vjp(design._outputs,theta);values.block_until_ready();forward_seconds=time.perf_counter()-t
memory.update(forward_rss_MiB=rss(),forward_peak_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,gpu_forward=gpu())
t=time.perf_counter();gradient=pullback((jnp.array([1.,0.,0.]),jnp.zeros_like(w)))[0];gradient.block_until_ready();adjoint_seconds=time.perf_counter()-t
memory.update(adjoint_rss_MiB=rss(),adjoint_peak_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,gpu_adjoint=gpu())
np.save(D/'gradient.npy',np.asarray(gradient))
check=design.forward(theta,precomputed=(values,w));check.pop('theta');assert check['status']=='ok',check
K=float(values[0]);assert abs(K-abs(check['Fz_top'])/.01)/K<=1e-6
envelope=jax.grad(lambda t:design.responses(t,jax.lax.stop_gradient(w))[0])(theta)
np.save(D/'envelope_gradient.npy',np.asarray(envelope));envelope_error=float(jnp.linalg.norm(gradient-envelope)/jnp.maximum(jnp.linalg.norm(envelope),1e-14))
assert envelope_error<=1e-6 and np.isfinite(np.asarray(gradient)).all()
rows=[];checks=[{'case':'baseline',**check}];forward_count=1
if small:
 for i,direction in enumerate([d1,d2]):
  ad=float(jnp.dot(gradient,direction))
  for h in [.001,.0005]:
   kval={};durations={}
   for sign in [-1,1]:
    perturbed=theta+sign*h*direction;design.validate_theta(perturbed)
    t=time.perf_counter();v,z=design._outputs(perturbed);v.block_until_ready();durations[str(sign)]=time.perf_counter()-t;forward_count+=1
    ch=design.forward(perturbed,precomputed=(v,z));ch.pop('theta');assert ch['status']=='ok',ch
    kval[sign]=float(v[0]);checks.append({'case':f'd{i}_h{h}_{sign}',**ch})
   fd=(kval[1]-kval[-1])/(2*h);err=abs(fd-ad);near=abs(ad)<=1e-8*max(abs(K),1);rel=None if near else err/abs(ad)
   row={'direction':i,'h':h,'AD':ad,'FD':fd,'abs_error':err,'relative_error':rel,'near_zero':near,'passed':err<=1e-7*max(abs(K),1) if near else rel<=1e-3,'reused_compile_forward_seconds':durations};rows.append(row);print(json.dumps(row),flush=True)
   (D/'differences.json').write_text(json.dumps(rows,indent=2)+'\n');(D/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
stop_monitor.set();(D/'gpu_samples.json').write_text(json.dumps(gpu_samples,indent=2)+'\n');memory['sampled_device_peak_MiB']=max((s['used_MiB'] for s in gpu_samples),default=None)
summary={'N':N,'M':M,'route':route,'K':K,'Vf':float(values[1]),'dK_dc':float(gradient[-1] if choice else gradient[0]),'spatial_direction_AD':float(jnp.dot(gradient,d2)),'build_seconds':build_seconds,'forward_seconds_including_first_trace_compile':forward_seconds,'adjoint_seconds_including_first_trace_compile':adjoint_seconds,'memory':memory,'envelope_relative_L2_error':envelope_error,'forward_count':forward_count,'adjoint_count':1,'total_seconds':time.perf_counter()-start,'baseline':check,'directional_passed':all(r['passed'] for r in rows),'source_sha256':{n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in ['voxel_field.py','geometry.py','design_fem.py','density_fem.py','pbc.py','fem.py']}}
(D/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
ledger=json.loads((O/'execution.json').read_text());ledger.append({'stage':3,'case':f'N{N}','kind':'small_forward_adjoint' if small else 'cost_forward_adjoint','seconds':summary['total_seconds'],'forward_count':forward_count,'adjoint_count':1,'exit_code':0});(O/'execution.json').write_text(json.dumps(ledger,indent=2)+'\n')
if not summary['directional_passed']:raise SystemExit('Derivative criterion failed; no further cost runs')

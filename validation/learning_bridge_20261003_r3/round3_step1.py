from pathlib import Path
import os, sys, time, json, resource, subprocess, threading, signal, logging
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
R=Path('/home/xuehu/projects/tpms_jax');sys.path.insert(0,str(R))
O=R/'validation/learning_bridge_20261003_r3';S=O/'step1'
import numpy as np
import jax, jax.numpy as jnp
from voxel_field import ImplicitVoxelDesign
from petsc4py import PETSc
signal.alarm(1200)
for key,val in {'ksp_rtol':1e-11,'ksp_atol':1e-13,'ksp_max_it':5000,'ksp_error_if_not_converged':True}.items():PETSc.Options()[key]=val
solver={'petsc_solver':{'ksp_type':'cg','pc_type':'gamg'},'tol':1e-12,'rel_tol':1e-10}
N=int(sys.argv[1]);case=sys.argv[2];target=S/f'N{N}_{case}'
target.mkdir(exist_ok=False)
from jax_fem import logger
handler=logging.FileHandler(target/'solver.log')
handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
logger.addHandler(handler)
previous=R/'validation/geometry_interface_20261003_r2/step3/N32'
data=np.load(previous/'inputs.npz')
theta=data['theta'].copy();direction=None;h=None
if case != 'baseline':
 _,d,hs,sgn=case.split('_');direction=data['direction_c' if d=='c' else 'direction_spatial'];h=float(hs);theta+=int(sgn)*h*direction
np.save(target/'input.npy',theta)
start=time.perf_counter();samples=[];done=threading.Event()
def abort(signum, frame):
 summary={'N':N,'case':case,'passed':False,'reason':'resource_margin_or_timeout_stop','signal':signum,'seconds':time.perf_counter()-start,'forward_count_budget_charge':1,'general_adjoint_count':0}
 (target/'aborted.json').write_text(json.dumps(summary,indent=2)+'\n')
 (target/'resource_samples.json').write_text(json.dumps(samples,indent=2)+'\n')
 ledger=json.loads((O/'execution.json').read_text());ledger.append({'step':1,**summary});(O/'execution.json').write_text(json.dumps(ledger,indent=2)+'\n')
 raise SystemExit('Resource margin/timeout reached; stop third round')
signal.signal(signal.SIGTERM,abort);signal.signal(signal.SIGALRM,abort)
def rss():return int(Path('/proc/self/status').read_text().split('VmRSS:')[1].split()[0])/1024
def available():return int(Path('/proc/meminfo').read_text().split('MemAvailable:')[1].split()[0])/1024
def monitor():
 while not done.is_set():
  row={'seconds':time.perf_counter()-start,'host_rss_MiB':rss(),'host_available_MiB':available()}
  try:
   gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip().split(',')
   row.update(device_used_MiB=int(gpu[0]),device_total_MiB=int(gpu[1]))
  except Exception:pass
  samples.append(row);done.wait(.5)
  if N==64 and (row['host_available_MiB']<1024 or ('device_total_MiB' in row and row['device_total_MiB']-row['device_used_MiB']<512)):
   os.kill(os.getpid(),signal.SIGTERM);return
threading.Thread(target=monitor,daemon=True).start()
start_available=available();phases={}
design=ImplicitVoxelDesign(N,64,solver_options=solver)
if N==64:
 assert json.loads((S/'N64_resource_precheck.json').read_text())['allowed']
 assert available()>=1024
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip().split(',')
 assert int(gpu[1])-int(gpu[0])>=512
phases['built_rss_MiB']=rss();t=time.perf_counter()
if case=='baseline':
 values,jac,w=design.stationary_outputs(jnp.asarray(theta));values.block_until_ready()
 np.save(target/'jacobian.npy',np.asarray(jac))
else:
 all_values,w=design._outputs(jnp.asarray(theta));all_values.block_until_ready();values=all_values[:2]
phases['solve_and_gradient_seconds' if case=='baseline' else 'solve_seconds']=time.perf_counter()-t
phases['post_value_rss_MiB']=rss();phases['post_value_peak_MiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
all_values=design.responses(jnp.asarray(theta),w)
row=design.forward(theta,precomputed=(all_values,w));row.pop('theta')
pts=np.asarray(design.problem.fe.points);node=np.asarray(w);per=0.
for ax in (0,1):
 other=[i for i in (0,1,2) if i!=ax]
 low=np.flatnonzero(np.isclose(pts[:,ax],0.,atol=1e-10));high=np.flatnonzero(np.isclose(pts[:,ax],1.,atol=1e-10))
 low=low[np.lexsort((pts[low,other[1]],pts[low,other[0]]))];high=high[np.lexsort((pts[high,other[1]],pts[high,other[0]]))]
 assert np.allclose(pts[low][:,other],pts[high][:,other],atol=1e-12,rtol=0)
 per=max(per,float(np.max(np.abs(node[low]-node[high]))))
loaded=np.isclose(pts[:,2],0.,atol=1e-10)|np.isclose(pts[:,2],1.,atol=1e-10)
row['checks'].update(periodic=per<=1e-10,loaded_face=float(np.abs(node[loaded,2]).max())<=1e-10,K_reaction=abs(float(values[0])-abs(row['Fz_top'])/.01)/float(values[0])<=1e-6)
row['status']='ok' if all(row['checks'].values()) else 'check_failed'
summary={'N':N,'case':case,'K':float(values[0]),'Vf':float(values[1]),'baseline_checks':row,'phases':phases,'forward_count':1,'general_adjoint_count':0,'start_host_available_MiB':start_available}
if N==32:
 old=json.loads((previous/'summary.json').read_text());oldgrad=np.load(previous/'gradient.npy');new=np.asarray(jac)[0]
 err=float(np.linalg.norm(new-oldgrad)/np.linalg.norm(oldgrad));response=max(abs(summary[k]-old[k])/abs(old[k]) for k in ['K','Vf'])
 summary.update(old_adjoint_relative_L2=err,old_response_relative=response,passed=row['status']=='ok' and err<=1e-6 and response<=1e-8)
else:summary['passed']=row['status']=='ok'
done.set();summary.update(total_seconds=time.perf_counter()-start,peak_host_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,sampled_host_available_min_MiB=min(s['host_available_MiB'] for s in samples),sampled_device_peak_MiB=max((s.get('device_used_MiB',0) for s in samples),default=0))
(target/'resource_samples.json').write_text(json.dumps(samples,indent=2)+'\n');(target/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
ledger=json.loads((O/'execution.json').read_text());ledger.append({'step':1,'case':f'N{N}_{case}','forward_count':1,'general_adjoint_count':0,'seconds':summary['total_seconds'],'passed':summary['passed']});(O/'execution.json').write_text(json.dumps(ledger,indent=2)+'\n')
print(json.dumps(summary),flush=True)
if not summary['passed']:raise SystemExit('Step1 criterion failed')

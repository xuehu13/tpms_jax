from pathlib import Path
import os,sys,json,time,resource,threading,subprocess,signal,logging
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
R=Path('/home/xuehu/projects/tpms_jax');sys.path.insert(0,str(R))
O=R/'validation/learning_bridge_20261003_r3';S=O/'step2'
import numpy as np
import jax.numpy as jnp
from design_fem import GyroidDesign
from jax_fem import logger
from petsc4py import PETSc
label=sys.argv[1];N=int(sys.argv[2]);q=next(a['theta'] for a in json.loads((S/'anchors.json').read_text()) if a['label']==label)
D=S/f'{label}_N{N}';D.mkdir(exist_ok=False)
handler=logging.FileHandler(D/'solver.log');handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'));logger.addHandler(handler)
np.save(D/'theta.npy',np.array(q));start=time.perf_counter();samples=[];done=threading.Event()
def abort(signum,frame):
 row={'step':2,'case':f'{label}_N{N}','passed':False,'reason':'resource_or_timeout_stop','forward_count':1,'seconds':time.perf_counter()-start}
 (D/'aborted.json').write_text(json.dumps(row,indent=2)+'\n');(D/'resource_samples.json').write_text(json.dumps(samples,indent=2)+'\n')
 ledger=json.loads((O/'execution.json').read_text());ledger.append(row);(O/'execution.json').write_text(json.dumps(ledger,indent=2)+'\n');raise SystemExit('Third round stopped at resource/time margin')
signal.signal(signal.SIGTERM,abort);signal.signal(signal.SIGALRM,abort);signal.alarm(1200)
def monitor():
 while not done.is_set():
  row={'seconds':time.perf_counter()-start,'host_available_MiB':int(Path('/proc/meminfo').read_text().split('MemAvailable:')[1].split()[0])/1024}
  try:
   g=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip().split(',');row.update(device_used_MiB=int(g[0]),device_total_MiB=int(g[1]))
  except Exception:pass
  samples.append(row)
  if row['host_available_MiB']<1024 or ('device_total_MiB' in row and row['device_total_MiB']-row['device_used_MiB']<512):os.kill(os.getpid(),signal.SIGTERM);return
  done.wait(.5)
threading.Thread(target=monitor,daemon=True).start()
for k,v in {'ksp_rtol':1e-11,'ksp_atol':1e-13,'ksp_max_it':5000,'ksp_error_if_not_converged':True}.items():PETSc.Options()[k]=v
model=GyroidDesign(N,beta=40,emin_ratio=1e-4,solver_options={'petsc_solver':{'ksp_type':'cg','pc_type':'gamg'},'tol':1e-12,'rel_tol':1e-10})
values,w=model._outputs(jnp.array(q));values.block_until_ready();row=model.forward(q,precomputed=(values,w))
pts=np.asarray(model.problem.fe.points);node=np.asarray(w);per=0.
for axis in (0,1):
 other=[i for i in (0,1,2) if i!=axis];low=np.flatnonzero(np.isclose(pts[:,axis],0.,atol=1e-10));high=np.flatnonzero(np.isclose(pts[:,axis],1.,atol=1e-10))
 low=low[np.lexsort((pts[low,other[1]],pts[low,other[0]]))];high=high[np.lexsort((pts[high,other[1]],pts[high,other[0]]))]
 assert np.allclose(pts[low][:,other],pts[high][:,other],atol=1e-12,rtol=0);per=max(per,float(np.abs(node[low]-node[high]).max()))
loaded=np.isclose(pts[:,2],0.,atol=1e-10)|np.isclose(pts[:,2],1.,atol=1e-10)
row['checks'].update(periodic=per<=1e-10,flat_loaded_face=float(abs(node[loaded,2]).max())<=1e-10,K_reaction=abs(float(values[0])-abs(row['Fz_top'])/.01)/float(values[0])<=1e-6)
done.set();summary={'label':label,'N':N,'theta':q,'K':float(values[0]),'Vf':float(values[1]),'passed':all(row['checks'].values()),'checks':row,'seconds':time.perf_counter()-start,'forward_count':1,'general_adjoint_count':0,'peak_host_rss_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'sampled_min_available_host_MiB':min(r['host_available_MiB'] for r in samples),'sampled_device_peak_MiB':max(r.get('device_used_MiB',0) for r in samples)}
(D/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(D/'resource_samples.json').write_text(json.dumps(samples,indent=2)+'\n')
ledger=json.loads((O/'execution.json').read_text());ledger.append({'step':2,'case':f'{label}_N{N}','forward_count':1,'general_adjoint_count':0,'seconds':summary['seconds'],'passed':summary['passed']});(O/'execution.json').write_text(json.dumps(ledger,indent=2)+'\n')
print(json.dumps(summary),flush=True)
if not summary['passed']:raise SystemExit('Physical checks failed')

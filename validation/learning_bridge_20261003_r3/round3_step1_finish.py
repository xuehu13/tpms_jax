from pathlib import Path
import json,sys,subprocess,time,hashlib
R=Path('/home/xuehu/projects/tpms_jax');O=R/'validation/learning_bridge_20261003_r3';S=O/'step1'
import numpy as np
previous=R/'validation/geometry_interface_20261003_r2/step3/N32'
data=np.load(previous/'inputs.npz');jac=np.load(S/'N64_baseline/jacobian.npy');baseline=json.loads((S/'N64_baseline/summary.json').read_text())
assert baseline['passed'] and np.isfinite(jac).all()
(S/'direction_plan.json').write_text(json.dumps({'source_inputs':str(previous/'inputs.npz'),'sha256':hashlib.sha256((previous/'inputs.npz').read_bytes()).hexdigest(),'directions':['c','spatial'],'steps':[.001,.0005],'max_remaining_forward':8,'nonzero_relative_tolerance':.001,'near_zero_AD':1e-8,'near_zero_absolute_tolerance_scale':1e-7,'locked_before_FD':True},indent=2)+'\n')
results=[]
for name in ['c','spatial']:
 direction=data['direction_'+name]
 ad=jac@direction
 for h in [.001,.0005]:
  responses={}
  for sign in [-1,1]:
   case=f'fd_{name}_{h}_{sign}'
   ledger=json.loads((O/'execution.json').read_text())
   assert sum(x.get('forward_count',x.get('forward_count_budget_charge',0)) for x in ledger)<10
   assert sum(x['seconds'] for x in ledger)<7200
   with (S/f'N64_{case}.console.txt').open('x') as stream:
    proc=subprocess.run([sys.executable,str(O/'round3_step1.py'),'64',case],cwd=R,stdout=stream,stderr=subprocess.STDOUT,timeout=1210)
   assert proc.returncode==0,case
   row=json.loads((S/f'N64_{case}/summary.json').read_text());assert row['passed']
   responses[sign]=np.array([row['K'],row['Vf']])
  fd=(responses[1]-responses[-1])/(2*h)
  for i,response in enumerate(['K','Vf']):
   scale=max(abs(baseline[response]),1);err=abs(fd[i]-ad[i]);near=abs(ad[i])<=1e-8*scale;rel=None if near else float(err/abs(ad[i]))
   row={'direction':name,'h':h,'response':response,'AD':float(ad[i]),'FD':float(fd[i]),'abs_error':float(err),'relative_error':rel,'near_zero':bool(near),'passed':bool(err<=1e-7*scale if near else rel<=1e-3)}
   results.append(row);print(json.dumps(row),flush=True)
  (S/'differences.json').write_text(json.dumps(results,indent=2)+'\n')
  assert all(r['passed'] for r in results),'Directional derivative failed; stop third round'
cases=[json.loads(p.read_text()) for p in S.glob('N*/summary.json')]
summary={'passed':True,'medium_large_forward_count':sum(r['forward_count'] for r in cases),'general_adjoint_count':0,'total_seconds':sum(r['total_seconds'] for r in cases),'N64_K':baseline['K'],'N64_Vf':baseline['Vf'],'N64_dK_dc':float(jac[0,-1]),'N64_spatial_AD':float(jac[0]@data['direction_spatial']),'max_nonzero_relative_FD_error':max(r['relative_error'] for r in results if r['relative_error'] is not None),'peak_host_rss_MiB':max(r['peak_host_rss_MiB'] for r in cases),'sampled_min_host_available_MiB':min(r['sampled_host_available_min_MiB'] for r in cases),'sampled_peak_device_MiB':max(r['sampled_device_peak_MiB'] for r in cases),'cost_includes_new_process_build_compile_and_checks':True,'resource_samples_are_not_rigorous_instantaneous_bounds':True}
(S/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)

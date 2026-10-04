from pathlib import Path
import json,sys,subprocess,time,hashlib,os
R=Path('/home/xuehu/projects/tpms_jax');sys.path.insert(0,str(R))
O=R/'validation/geometry_interface_20261003_r2';S=O/'step1';S.mkdir(exist_ok=True)
import numpy as np
import jax; jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from voxel_field import sample_implicit,periodic_trilinear
from geometry import density,project_field
N=64;c=.541062;b=40.
offsets=(1+np.array([-1,1])/np.sqrt(3))/2
x=(np.arange(N)[:,None]+offsets).reshape(-1)/N
points=jnp.asarray(np.stack(np.meshgrid(x,x,x,indexing='ij'),axis=-1))
analytic=np.asarray(density(points,c,b));audits=[]
for M in (64,32):
 g=sample_implicit(M);mapped=np.asarray(project_field(periodic_trilinear(g,points),c,b))
 test=jnp.array([[0.,.23,.51],[-.01,.91,1.2],[.73,.36,.21]])
 per=float(jnp.max(jnp.abs(periodic_trilinear(g,test)-periodic_trilinear(g,test+jnp.array([1.,-1.,2.])))))
 row={'M':M,'N':N,'implicit_min':float(g.min()),'implicit_max':float(g.max()),'periodic_translation_error':per,'projection_min':float(mapped.min()),'projection_max':float(mapped.max()),'analytic_vf':float(analytic.mean()),'mapped_vf':float(mapped.mean()),'weighted_L1':float(np.abs(mapped-analytic).mean()),'max_field_error':float(np.abs(mapped-analytic).max()),'analytic_transition':float(np.mean((analytic>.05)&(analytic<.95))),'mapped_transition':float(np.mean((mapped>.05)&(mapped<.95)))}
 assert per<1e-12 and mapped.min()>=0 and mapped.max()<=1
 audits.append(row)
(S/'field_audit.json').write_text(json.dumps(audits,indent=2)+'\n')
del points,analytic,mapped,g
# Separate processes avoid retaining field-audit GPU allocations during large FEM.
print(json.dumps(audits),flush=True)
K0=1.9764840370021242;KA=1.952391117811203
rows=[]
for M in (64,32):
 cmd=[sys.executable,str(R/'scripts/capture_binary_projection_reference.py'),'--N','64','--beta','40','--emin-ratio','1e-4','--lateral','fixed','--solver','petsc','--c',str(c),'--voxel-M',str(M),'--field-kind','implicit','--out',str(S/f'M{M}_N64.json')]
 ledger=json.loads((O/'execution.json').read_text()) if (O/'execution.json').exists() else []
 if not (S/f'M{M}_N64.json').exists():
  start=time.perf_counter()
  with (S/f'M{M}_N64.console.txt').open('w') as f:
   result=subprocess.run(cmd,cwd=R,env={**os.environ,'XLA_PYTHON_CLIENT_PREALLOCATE':'false'},stdout=f,stderr=subprocess.STDOUT,timeout=1200)
  elapsed=time.perf_counter()-start
  ledger.append({'stage':1,'case':f'M{M}_N64','kind':'large_forward','command':cmd,'seconds':elapsed,'exit_code':result.returncode})
  (O/'execution.json').write_text(json.dumps(ledger,indent=2)+'\n')
  assert result.returncode==0
 else:elapsed=next(l['seconds'] for l in ledger if l['case']==f'M{M}_N64')
 r=json.loads((S/f'M{M}_N64.json').read_text());K=abs(r['Fz_top'])/.01
 d=abs(K-K0)/K0;e=abs(K-KA)/KA
 row={'M':M,'K':K,'mapping_relative':d,'binary_relative':e,'seconds':elapsed,'passed':r['status']=='ok' and d<=.01 and e<=.02}
 rows.append(row);print(json.dumps(row),flush=True)
 (S/'summary.json').write_text(json.dumps({'cases':rows,'passed_M':[r['M'] for r in rows if r['passed']],'selected_M':min([r['M'] for r in rows if r['passed']],default=None),'stopped':not row['passed'],'no_more_candidates':True},indent=2)+'\n')
 if not row['passed']:break

"""Check geometry fields and floating-point resolution at frozen saved state.

This is a local diagnostic, never a substitute for actual complete FD paths.
"""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import json,time
import numpy as np
from shape import ShapeDesignPath,B,jax,jnp
from run import ready,load
start=time.time();model=ShapeDesignPath();p=jnp.array([.5,.05,0.]);seed=jnp.array([0.,0.,1.])
fields,dfields=ready(jax.jvp(model.fields,(p,),(seed,)))
state=load(B/'G5_shape/AD/shape/final_state.npz');ds=load(B/'G5_shape/AD/shape/final_tangent.npz')
derivative=np.asarray(ready(jax.jvp(model.observe,(state,fields),(ds,dfields)))[1])
rows=[]
for h in [1e-7,5e-8]:
    fp=ready(model.fields(jnp.array([.5,.05,h])));fm=ready(model.fields(jnp.array([.5,.05,-h])))
    errors=[float(np.linalg.norm(np.asarray((x-y)/(2*h)-a).ravel())/max(np.linalg.norm(np.asarray(a).ravel()),1e-300))
            for x,y,a in zip(fp,fm,dfields)]
    sp=jax.tree_util.tree_map(lambda s,d:s+h*d,state,ds)
    sm=jax.tree_util.tree_map(lambda s,d:s-h*d,state,ds)
    vp=np.asarray(ready(model.observe(sp,fp)));vm=np.asarray(ready(model.observe(sm,fm)))
    numerical=(vp-vm)/(2*h)
    rows.append(dict(h=h,geometry_field_relative_L2=errors,
        affine_saved_state_observable_FD=numerical.tolist(),full_AD_at_saved_state=derivative.tolist(),
        relative_difference=(np.abs(numerical-derivative)/np.maximum(np.abs(derivative),1e-300)).tolist(),
        observable_response_difference=(vp-vm).tolist()))
result=dict(scope='Local saved-state resolution only, no new dynamic time steps',rows=rows,
    primary_geometry_scale_relative_tolerance=1e-5,local_observable_relative_tolerance=.01,
    passed=all(r['geometry_field_relative_L2'][0]<1e-5 and max(r['relative_difference'])<.01 for r in rows),
    original_response_gate_unchanged=True,wall_minutes=(time.time()-start)/60)
out=B/'checks/G_shape_refined_resolution.json';assert not out.exists()
out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2),flush=True)

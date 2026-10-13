"""One scalar-objective VJP with hierarchical recomputation and host checkpoints."""
from pathlib import Path
import argparse,json,math,os,time
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
from design import DesignPath,B,jax,jnp,np
from run import write,ready,valid,save

def functions(model):
    # Keep256-step checkpoints on the host; nested32-step rematerialization
    # avoids retaining hundreds of large element-kernel tapes on the8GB GPU.
    inner=model.block(32,remat=True)
    def outer(state,fields,dt,count):
        def one(s,k):
            return jax.checkpoint(inner)(s,fields,dt,jnp.clip(count-32*k,0,32)),None
        return jax.lax.scan(one,state,jnp.arange(8))[0]
    forward=jax.jit(outer)
    @jax.jit
    def backward(state,fields,dt,count,cotangent):
        _,pullback=jax.vjp(lambda s,f:outer(s,f,dt,count),state,fields)
        return pullback(cotangent)
    return outer,forward,backward

def benchmark():
    start=time.time();model=DesignPath();p=jnp.array([.5,.05])
    fields,dfields=ready(jax.jvp(model.fields,(p,),(jnp.array([1.,0.]),)))
    # Benchmark at the completed loaded state, where contact/work sensitivities
    # are substantial; startup work is too close to zero for a relative test.
    with np.load(B/'AD/thickness/final_state.npz') as z:
        state=tuple(jnp.asarray(z[f's{i}']) for i in range(8))
    with np.load(B/'AD/thickness/final_tangent.npz') as z:
        dstate=tuple(jnp.asarray(z[f's{i}']) for i in range(8))
    cfg=json.loads((B/'checks/benchmark.json').read_text());dt=cfg['fixed_dt_s'];steps=cfg['steps_to5pct']
    outer,forward,backward=functions(model)
    cot=tuple(jnp.zeros_like(x) for x in state[:-1])+(jnp.array(1.),)
    t=time.time();ready(forward(state,fields,dt,256));fc=time.time()-t
    t=time.time();ready(forward(state,fields,dt,256));fr=time.time()-t
    t=time.time();vjp=ready(backward(state,fields,dt,256,cot));bc=time.time()-t
    t=time.time();vjp=ready(backward(state,fields,dt,256,cot));br=time.time()-t
    tangent=ready(jax.jvp(lambda s,f:outer(s,f,dt,256),(state,fields),(dstate,dfields)))[1]
    duality=sum(float(jnp.vdot(x,y)) for x,y in zip(jax.tree_util.tree_leaves(vjp),jax.tree_util.tree_leaves((dstate,dfields))))
    expected=float(tangent[-1]);err=abs(duality-expected)/max(abs(expected),1e-20)
    result=dict(block_steps=256,forward256_s=fr,reverse256_s=br,
                compile_and_first_forward_s=fc,compile_and_first_reverse_s=bc,
                block_work_JVP=expected,block_VJP_contraction=duality,duality_relative_error=err,
                estimated_full_forward_reverse_minutes=(fr+br)*math.ceil(steps/256)/60,
                host_checkpoint_estimate_GiB=sum(x.size*x.dtype.itemsize for x in state)*math.ceil(steps/256)/1024**3,
                wall_s=time.time()-start,passed=err<1e-7,
                scope='256-step AD duality/runtime at the loaded5pct state; not full-path reverse certification')
    write(B/'checks/reverse_benchmark.json',result);print(json.dumps(result),flush=True)
    if not result['passed']:raise ValueError('Block JVP/VJP duality failed')

def execute():
    started=time.time();model=DesignPath()
    first=json.loads((B/'analysis/metrics.json').read_text())
    if not first['step1_pass']:raise ValueError('First full-path JVP/FD check has not passed')
    bench=json.loads((B/'checks/reverse_benchmark.json').read_text())
    if not bench['passed']:raise ValueError('Reverse benchmark not passed')
    # A separately declared step2 budget, chosen after benchmark and before VJP.
    budget=json.loads((B/'input/reverse_protocol.json').read_text())
    deadline=started+budget['wall_budget_seconds']
    write(B/'checks/reverse_execution_clock.json',dict(started_unix=started,deadline_unix=deadline,pid=os.getpid()))
    if bench['estimated_full_forward_reverse_minutes']*60>deadline-time.time()-300:
        raise TimeoutError('Reverse benchmark exceeds declared budget')
    out=B/'AD/reverse_work';out.mkdir(exist_ok=False)
    cfg=json.loads((B/'checks/benchmark.json').read_text());dt=cfg['fixed_dt_s'];steps=cfg['steps_to5pct']
    p=jnp.array([.5,.05]);fields=ready(model.fields(p));state=ready(model.initial(fields))
    outer,forward,backward=functions(model);checkpoints=[];counts=[];diag_rows=[];done=0
    def guard():
        if time.time()>deadline:raise TimeoutError('Reverse work budget exhausted')
    write(B/'status.json',dict(stage='step2_reverse_forward',active_jobs=[dict(pid=os.getpid(),task='reverse_work')],step1='passed',step2='running',step3='pending',step4='pending'))
    while done<steps:
        guard();checkpoints.append(jax.tree_util.tree_map(np.asarray,state));count=min(256,steps-done);counts.append(count)
        state=ready(forward(state,fields,dt,count));done+=count
        diag=np.asarray(model.diagnostics(state,fields))
        if not valid(diag):save(out/'invalid_state.npz',state);raise ValueError('Reverse forward actualJ/finite energy guard')
        diag_rows.append(dict(step=done,min_required_J=float(diag[2]),all_J_min=float(diag[1]),negative_void_J_points=int(diag[4])))
        if len(counts)%4==0 or done==steps:
            progress=dict(stage='checkpoint_forward',steps=done,total_steps=steps,wall_s=time.time()-started)
            write(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    values=np.asarray(model.observe(state,fields));save(out/'final_state.npz',state)
    base=json.loads((B/'AD/thickness/result.json').read_text())['terminal']
    equivalence=max(abs(values[0]-base['work_N_mm'])/max(abs(base['work_N_mm']),1e-20),
                    abs(values[1]-base['force_N'])/max(abs(base['force_N']),1e-20))
    if equivalence>1e-6:raise ValueError('Rematerialized forward differs from original fixed-step path')
    cot=tuple(jnp.zeros_like(x) for x in state[:-1])+(jnp.array(1.),)
    fields_cot=jax.tree_util.tree_map(jnp.zeros_like,fields)
    write(B/'status.json',dict(stage='step2_reverse_backward',active_jobs=[dict(pid=os.getpid(),task='reverse_work')],step1='passed',step2='running',step3='pending',step4='pending'))
    for k in range(len(counts)-1,-1,-1):
        guard();saved=jax.tree_util.tree_map(jnp.asarray,checkpoints[k])
        cot,g=ready(backward(saved,fields,dt,counts[k],cot))
        fields_cot=ready(jax.tree_util.tree_map(lambda x,y:x+y,fields_cot,g))
        if k%4==0:
            progress=dict(stage='backward',blocks_left=k,total_blocks=len(counts),wall_s=time.time()-started)
            write(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    _,init_pullback=jax.vjp(model.initial,fields)
    fields_cot=jax.tree_util.tree_map(lambda x,y:x+y,fields_cot,init_pullback(cot)[0])
    _,parameter_pullback=jax.vjp(model.fields,p);gradient=np.asarray(ready(parameter_pullback(fields_cot))[0])
    jvp=base['work_gradient_N'];relative=abs(gradient[0]-jvp)/max(abs(jvp),1e-20)
    result=dict(complete=True,objective='full_path_loading_work',parameters=['thickness_mm','interface_10_90_mm'],
                value_N_mm=float(values[0]),gradient_N=gradient.tolist(),
                reference_JVP_thickness_N=jvp,relative_VJP_JVP_thickness_difference=float(relative),
                forward_equivalence_relative=float(equivalence),passed=bool(np.isfinite(gradient).all() and relative<=.01),
                interface_gradient_status='VJP computed; independent interface FD not yet checked',
                wall_minutes=(time.time()-started)/60,steps=steps,dt_s=dt,
                checks=diag_rows,scope='Complete fixed-time5pct reverse work derivative; not terminal-force VJP or shape certification')
    write(out/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k!='checks'}),flush=True)
    write(B/'status.json',dict(stage='step2_reverse_analyzed',active_jobs=[],step1='passed',step2_reverse='passed' if result['passed'] else 'not_passed',step2_interface='pending',step3='pending',step4='pending',new_reverse_AD=bool(result['passed'])))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['benchmark','execute']);a=ap.parse_args()
    try:benchmark() if a.mode=='benchmark' else execute()
    except (Exception,KeyboardInterrupt) as exc:
        write(B/f'reverse_failure_{a.mode}.json',dict(type=type(exc).__name__,message=str(exc)))
        write(B/'status.json',dict(stage='step2_reverse_stopped',active_jobs=[],step1='passed',step2='not_passed',step3='gated_not_started',step4='gated_not_started',reason=str(exc)))
        raise

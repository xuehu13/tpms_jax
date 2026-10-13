"""One bounded coefficient batch, importing the frozen r53 execution helpers."""
import os,time,json,math,hashlib,signal,traceback
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
from coefficient_model import CoefficientPath,B,R,jax,jnp,np
from reverse import functions
from scipy.optimize import brentq
from pathlib import Path

def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)
def ready(x):return jax.block_until_ready(x)
def flat(x):return np.concatenate([np.asarray(v).ravel() for v in jax.tree_util.tree_leaves(x)])
def save(p,state):np.savez(p,**{f's{i}':np.asarray(x) for i,x in enumerate(state)})
def load(p):
    with np.load(p) as z:return tuple(jnp.asarray(z[f's{i}']) for i in range(8))
def check(ad,fd,fine,scale,goal_scale=1.):
    near=abs(fine)*scale/goal_scale<1e-6
    ae=abs(ad-fine)*scale/goal_scale if near else abs(ad-fine)/abs(fine)
    fe=abs(fd-fine)*scale/goal_scale if near else abs(fd-fine)/abs(fine)
    threshold=1e-5 if near else .01
    return dict(AD=float(ad),FD=float(fd),FD_half=float(fine),near_zero=bool(near),
                AD_FD_error=float(ae),FD_halving_error=float(fe),passed=bool(ae<=threshold and fe<=threshold))

def main():
    protocol=json.loads((B/'input/protocol.json').read_text());started=protocol['started_unix'];deadline=started+protocol['total_wall_budget_s']
    stopped=[]
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda s,f:stopped.append(s))
    def guard():
        if stopped:raise InterruptedError('Termination requested')
        if time.time()>deadline:raise TimeoutError('Frozen total wall budget exhausted')
    def status(stage,**kw):
        write(B/'status.json',dict(stage=stage,active_jobs=[dict(pid=os.getpid(),task=stage)],wall_s=time.time()-started,
            heartbeat='PAUSED',r53='unchanged_paused',**kw))
    status('geometry_checks')
    model=CoefficientPath();p=np.array(protocol['baseline']);scales=np.array(protocol['parameter_scales']);h=scales*protocol['relative_fd_step']
    base=model.geometry.evaluate(p);geomchecks=[];fields=ready(model.fields(jnp.asarray(p)))
    baseline_projection=base['audit']
    for k in range(3):
        guard();seed=np.eye(3)[k]
        _,df=ready(jax.jvp(model.fields,(jnp.asarray(p),),(jnp.asarray(seed),)))
        perturbed=[];partitions=[]
        for delta in [h[k],-h[k],h[k]/2,-h[k]/2]:
            q=p+delta*seed
            audit=model.partition_audit(q);partitions.append(audit)
            if audit['changed_nearest_rim_assignments']:raise ValueError('Declared FD domain changes nearest-rim partition')
            perturbed.append(ready(model.fields(jnp.asarray(q))))
        # Check each field separately; do not let large mass/stiffness hide contact interpolation.
        rows=[]
        d_leaves=jax.tree_util.tree_leaves(df);f_leaves=[jax.tree_util.tree_leaves(v) for v in perturbed]
        for name,ad,a,b,c,d in zip(['scale','node_mass','periodic_mass','macro_c','macro_schur','contact_area','contact_Q2'],d_leaves,*f_leaves):
            ad=np.asarray(ad);fd=(np.asarray(a)-np.asarray(b))/(2*h[k]);fine=(np.asarray(c)-np.asarray(d))/h[k]
            norm=np.linalg.norm(fine.ravel());near=norm*scales[k]<1e-10
            ae=np.linalg.norm((ad-fine).ravel())*scales[k] if near else np.linalg.norm((ad-fine).ravel())/norm
            fe=np.linalg.norm((fd-fine).ravel())*scales[k] if near else np.linalg.norm((fd-fine).ravel())/norm
            rows.append(dict(field=name,relative_AD_FD=float(ae),relative_FD_halving=float(fe),near_zero=bool(near),passed=bool(ae<=(1e-7 if near else .01) and fe<=(1e-7 if near else .01))))
        geomchecks.append(dict(coefficient=protocol['parameters'][k],fields=rows,partitions=partitions,passed=all(x['passed'] for x in rows)))
    _,radial=ready(jax.jvp(model.fields,(jnp.asarray(p),),(jnp.asarray(p),)))
    radial_norm=float(np.linalg.norm(flat(radial)))
    scaled=model.geometry.evaluate(p*1.0001)
    scale_distance_error=float(np.max(np.abs(scaled['distance']-base['distance'])))
    geom=dict(coefficients=geomchecks,projection=baseline_projection,common_scaling_field_tangent_l2=radial_norm,
        common_scaling_max_distance_error_over_L=scale_distance_error,passed=all(x['passed'] for x in geomchecks) and radial_norm<1e-7 and scale_distance_error<1e-10)
    write(B/'checks/geometry.json',geom)
    if not geom['passed']:raise ValueError('Equation geometry/contact-field derivative gate failed; no long run')
    status('short_duality_and_cost')
    outer,forward,backward=functions(model)
    tangent=jax.jit(lambda s,ds,f,df,dt,count:jax.jvp(lambda ss,ff:outer(ss,ff,dt,count),(s,f),(ds,df)))
    observe_jvp=jax.jit(lambda s,ds,f,df:jax.jvp(model._observe,(s,f),(ds,df)))
    seed=jnp.array([.3,-.7,.2]);_,dfields=ready(jax.jvp(model.fields,(jnp.asarray(p),),(seed,)))
    state,dstate=ready(jax.jvp(model.initial,(fields,),(dfields,)));dt=5.8e-8
    ready(forward(state,fields,dt,256));t=time.time();ready(forward(state,fields,dt,256));fr=time.time()-t
    ready(tangent(state,dstate,fields,dfields,dt,256));t=time.time();_,ds=ready(tangent(state,dstate,fields,dfields,dt,256));jr=time.time()-t
    rng=np.random.default_rng(57);cot=tuple(jnp.asarray(rng.normal(size=x.shape)) for x in state)
    ready(backward(state,fields,dt,256,cot));t=time.time();vs,vf=ready(backward(state,fields,dt,256,cot));br=time.time()-t
    expected=sum(float(jnp.vdot(x,y)) for x,y in zip(ds,cot))
    terms=[float(jnp.vdot(x,y)) for x,y in zip(jax.tree_util.tree_leaves((vs,vf)),jax.tree_util.tree_leaves((dstate,dfields)))]
    actual=sum(terms);err=abs(actual-expected)/max(abs(expected),sum(abs(x) for x in terms)*1e-10,1e-20)
    bench=dict(forward256_s=fr,JVP256_s=jr,VJP256_s=br,short_duality_relative=err,short_duality_passed=err<1e-7,
        scope='256-step all-chain mixed-coefficient duality and cost only; not full response certification',wall_s=time.time()-started)
    write(B/'checks/benchmark.json',bench)
    if not bench['short_duality_passed']:raise ValueError('Short JVP/VJP duality failed')
    choices=[]
    for target in protocol['candidate_compressions']:
        fraction=brentq(lambda x:.2*(10*x**3-15*x**4+6*x**5)-target,0,1,xtol=1e-14)
        end=fraction*.004;steps=math.ceil(end/dt);blocks=math.ceil(steps/256)
        estimate=(12*fr+jr+br)*blocks*1.2+600
        choices.append(dict(compression=target,end_time_s=end,steps=steps,dt_s=end/steps,estimated_remaining_s=estimate))
    eligible=[c for c in choices if c['estimated_remaining_s']<deadline-time.time()]
    if not eligible:raise TimeoutError('Neither declared small-compression endpoint fits frozen180min budget')
    execution=dict(eligible[0]);execution.update(choices=choices,fd_steps=h.tolist(),frozen_unix=time.time(),source_hashes={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in (B/'tools').glob('*.py')})
    write(B/'input/execution.json',execution);steps=execution['steps'];dt=execution['dt_s']
    # Representative full JVP also supplies the exact baseline checkpoints reused by VJP.
    status('full_JVP_c2',total_steps=steps)
    fields,dfields=ready(jax.jvp(model.fields,(jnp.asarray(p),),(jnp.array([0.,1.,0.]),)))
    state,dstate=ready(jax.jvp(model.initial,(fields,),(dfields,)))
    out=B/'AD/c2';out.mkdir();(out/'checkpoints').mkdir();counts=[];rows=[];done=0
    def observe(s,f,done,wall,grad=None):
        diag=np.asarray(model.diagnostics(s,f));values=np.asarray(model.observe(s,f))
        if not (np.isfinite(diag).all() and diag[2]>0 and diag[3]==0 and diag[5]>0):raise ValueError('Actual-J/finite-energy guard failed')
        row=dict(steps=done,time_s=float(s[-2]),compression=float(model.ex.motion(s[-2])),work=float(values[0]),force=float(values[1]),
            min_required_J=float(diag[2]),min_actual_J=float(diag[1]),KE=float(diag[6]),IE=float(diag[0]),wall_s=wall)
        if grad is not None:row['gradient']=np.asarray(grad).tolist()
        return row
    case_start=time.time()
    while done<steps:
        guard();count=min(256,steps-done);save(out/'checkpoints'/f'{len(counts):04d}.npz',state);counts.append(count)
        state,dstate=ready(tangent(state,dstate,fields,dfields,dt,count));done+=count
        _,grad=ready(observe_jvp(state,dstate,fields,dfields));row=observe(state,fields,done,time.time()-case_start,grad);rows.append(row)
        if len(rows)%4==0 or done==steps:write(out/'progress.json',row);print(json.dumps(dict(case='JVP_c2',**row)),flush=True)
    save(out/'final_state.npz',state);save(out/'final_tangent.npz',dstate)
    write(out/'result.json',dict(complete=True,terminal=rows[-1],counts=counts,path=rows,wall_s=time.time()-case_start))
    primal_terminal=state;grad_c2=np.asarray(grad);base_values=np.asarray(model.observe(state,fields));fd_results={}
    for k in [1,0,2]:
        for name,delta in [('plus_h',h[k]),('minus_h',-h[k]),('plus_half',h[k]/2),('minus_half',-h[k]/2)]:
            guard();label=protocol['parameters'][k]+'_'+name;status('FD_'+label,completed_FD=list(fd_results));cs=time.time()
            q=p+delta*np.eye(3)[k];ff=ready(model.fields(jnp.asarray(q)));s=ready(model.initial(ff));done=0;path=[];folder=B/'FD'/label;folder.mkdir()
            while done<steps:
                guard();count=min(256,steps-done);s=ready(forward(s,ff,dt,count));done+=count
                row=observe(s,ff,done,time.time()-cs);path.append(row)
                if len(path)%4==0 or done==steps:write(folder/'progress.json',row)
            save(folder/'final_state.npz',s);res=dict(complete=True,parameters=q.tolist(),terminal=path[-1],path=path,wall_s=time.time()-cs)
            write(folder/'result.json',res);fd_results[label]=res;print(json.dumps(dict(case=label,**path[-1])),flush=True)
        if k==1:
            def fd_value(k,objective):
                c=protocol['parameters'][k]
                a=fd_results[c+'_plus_h']['terminal'][objective];b=fd_results[c+'_minus_h']['terminal'][objective]
                aa=fd_results[c+'_plus_half']['terminal'][objective];bb=fd_results[c+'_minus_half']['terminal'][objective]
                return (a-b)/(2*h[k]),(aa-bb)/h[k]
            checks={obj:check(grad_c2[i],*fd_value(1,obj),scales[1]) for i,obj in enumerate(['work','force'])}
            write(B/'checks/full_JVP_c2.json',checks)
            if not checks['work']['passed']:raise ValueError('Representative full coefficient work JVP/FD gate failed; long VJP not executed')
    # Check FD self-consistency for each primary work derivative before long reverse.
    for k in range(3):
        c=protocol['parameters'][k];a=fd_results[c+'_plus_h']['terminal']['work'];b=fd_results[c+'_minus_h']['terminal']['work']
        aa=fd_results[c+'_plus_half']['terminal']['work'];bb=fd_results[c+'_minus_half']['terminal']['work']
        fd=(a-b)/(2*h[k]);fine=(aa-bb)/h[k]
        if not check(fine,fd,fine,scales[k])['passed']:raise ValueError('Primary work FD halves unstable for '+c+'; long VJP gated')
    status('full_work_VJP',baseline_checkpoints_reused=True)
    cot=tuple(jnp.zeros_like(x) for x in primal_terminal[:-1])+(jnp.array(1.),)
    fields_cot=jax.tree_util.tree_map(jnp.zeros_like,fields);reverse_start=time.time()
    for k in range(len(counts)-1,-1,-1):
        guard();saved=load(out/'checkpoints'/f'{k:04d}.npz')
        cot,g=ready(backward(saved,fields,dt,counts[k],cot));fields_cot=ready(jax.tree_util.tree_map(lambda x,y:x+y,fields_cot,g))
        if k%4==0:write(B/'AD/work_VJP_progress.json',dict(blocks_left=k,total_blocks=len(counts),wall_s=time.time()-reverse_start))
    _,back_init=jax.vjp(model.initial,fields);fields_cot=jax.tree_util.tree_map(lambda x,y:x+y,fields_cot,back_init(cot)[0])
    _,back_param=jax.vjp(model.fields,jnp.asarray(p));gradient=np.asarray(ready(back_param(fields_cot))[0]);checks=[]
    for k in range(3):
        c=protocol['parameters'][k];a=fd_results[c+'_plus_h']['terminal']['work'];b=fd_results[c+'_minus_h']['terminal']['work']
        aa=fd_results[c+'_plus_half']['terminal']['work'];bb=fd_results[c+'_minus_half']['terminal']['work']
        checks.append(dict(coefficient=c,**check(gradient[k],(a-b)/(2*h[k]),(aa-bb)/h[k],scales[k])))
    result=dict(complete=True,objective='full_path_loading_work',gradient=gradient.tolist(),gradient_units='N mm per dimensionless coefficient',
        baseline_work=float(base_values[0]),baseline_force=float(base_values[1]),checks=checks,
        c2_VJP_JVP_relative=float(abs(gradient[1]-grad_c2[0])/max(abs(grad_c2[0]),1e-30)),
        common_scaling_gradient_contraction=float(gradient@p),reverse_wall_s=time.time()-reverse_start,total_wall_s=time.time()-started,
        passed=all(x['passed'] for x in checks),compression=execution['compression'],steps=steps)
    write(B/'AD/work_VJP_result.json',result)
    write(B/'status.json',dict(stage='completed' if result['passed'] else 'completed_not_passed',active_jobs=[],result='AD/work_VJP_result.json',
        heartbeat='PAUSED',r53='unchanged_paused',new_training=False,new_Git=False))
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        write(B/'failure.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        write(B/'status.json',dict(stage='stopped',active_jobs=[],reason=str(exc),heartbeat='PAUSED',r53='unchanged_paused'))
        raise

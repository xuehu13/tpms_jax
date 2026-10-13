"""Bounded G design checks; one shared fixed-time AD/FD execution loop."""
from pathlib import Path
import argparse,hashlib,json,math,os,signal,time
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
from design import DesignPath,B,R,jax,jnp,np,estimate_initial_dt

def parameters(case):return case.get('parameters',case.get('params_mm'))
def make_model(job):
    cls=DesignPath
    if job and job.get('model_adapter')=='periodic_shape':
        from shape import ShapeDesignPath
        cls=ShapeDesignPath
    return cls(job.get('model_config','input/protocol.json'),job.get('design_geometry','input/design_geometry.npz')) if job else cls()

def write(p,v):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');temp.replace(p)
def ready(x):jax.block_until_ready(x);return x
def save(p,state):np.savez_compressed(p,**{f's{i}':np.asarray(x) for i,x in enumerate(state)})
def load(p):
    with np.load(p) as z:
        state=tuple(jnp.asarray(z[f's{i}']) for i in range(8))
    if not all(np.isfinite(np.asarray(x)).all() for x in state):raise ValueError('Nonfinite restart state/tangent')
    return state
def valid(diag):return bool(np.isfinite(diag).all() and diag[2]>0 and diag[3]==0 and diag[5]>0)

def stability_estimator(model):
    @jax.jit
    def apply(dq,da,q,a,t,fields):
        v=jax.jvp(lambda x,y:model._acceleration(x,y,t,fields),(q,a),(dq,da))[1]
        return -v[0],-v[1]
    def estimate(state,fields):
        rng=np.random.default_rng(53);dq=jnp.asarray(rng.normal(size=state[0].shape));da=jnp.asarray(rng.normal(size=2))
        lam=0.
        for _ in range(25):
            v=model.ex.full_velocity(dq,da);norm=jnp.sqrt(jnp.sum(fields[1][:,None]*v*v))
            dq,da=apply(dq/norm,da/norm,state[0],state[1],state[-2],fields)
            v=model.ex.full_velocity(dq,da);lam=float(jnp.sqrt(jnp.sum(fields[1][:,None]*v*v)))
        if not math.isfinite(lam) or lam<=0:raise ValueError('Invalid frequency estimate')
        return dict(allowed_dt_s=min(float(model.ex.dt_estimate)*.5,.6/math.sqrt(lam)),omega2=lam,
                    interpretation='25-iteration state power estimate with conservative safety; not a proven spectral bound')
    return estimate

def benchmark(spec=None):
    start=time.time();job=json.loads((B/spec).read_text()) if spec else None
    model=make_model(job)
    cfg=model.cfg;checks_dir=B/'checks'/job['id'] if job else B/'checks';checks_dir.mkdir(exist_ok=True)
    errors=model.baseline_checks();write(checks_dir/'baseline_equivalence.json',errors)
    if not errors['pass']:raise ValueError('Adapter does not reproduce original fields/acceleration')
    p=job.get('baseline_parameters',[.5,.05]) if job else [.5,.05]
    seed=job.get('benchmark_seed',[1.,0.]) if job else [1.,0.]
    fields,dfields=ready(jax.jvp(model.fields,(jnp.array(p),),(jnp.array(seed),)))
    state,dstate=ready(jax.jvp(model.initial,(fields,),(dfields,)))
    block=model.block();primal=jax.jit(block)
    tangent=jax.jit(lambda s,ds,f,df,dt,count:jax.jvp(lambda ss,ff:block(ss,ff,dt,count),(s,f),(ds,df)))
    dt=cfg['fixed_dt_max_s'];c=64
    t=time.time();ready(primal(state,fields,dt,c));pc=time.time()-t
    t=time.time();ready(primal(state,fields,dt,c));pr=time.time()-t
    t=time.time();ready(tangent(state,dstate,fields,dfields,dt,c));tc=time.time()-t
    t=time.time();ready(tangent(state,dstate,fields,dfields,dt,c));tr=time.time()-t
    initial_allowed,bound=estimate_initial_dt(model.ex)
    steps=math.ceil(cfg['end_time_s']/dt);actual_dt=cfg['end_time_s']/steps
    checks={'baseline':errors,'primal_compile_and_first64_s':pc,'primal64_s':pr,
            'JVP_compile_and_first64_s':tc,'JVP64_s':tr,'steps_to5pct':steps,'fixed_dt_s':actual_dt,
            'original_model_initial_allowed_dt_s':initial_allowed,'initial_estimate':bound,
            'estimated_one_primal_minutes':pr*steps/64/60,'estimated_one_JVP_minutes':tr*steps/64/60,
            'estimated_thickness_check_minutes':(4*pr+tr)*steps/64/60,
            'benchmark_wall_s':time.time()-start,'scope':'Benchmark includes no full design gradient or full response'}
    write(checks_dir/'benchmark.json',checks);print(json.dumps(checks),flush=True)
    status=json.loads((B/'status.json').read_text()) if job else dict(step1='pending',step2='pending',step3='pending',step4='pending')
    status.update(stage=(job['id'] if job else 'step1')+'_benchmark_complete',active_jobs=[]);write(B/'status.json',status)

def execute(spec=None):
    start=time.time();job=json.loads((B/spec).read_text()) if spec else None
    model=make_model(job);cfg=model.cfg
    job_id=job['id'] if job else 'first_thickness'
    previous=json.loads((B/'status.json').read_text())
    phase_history={k:v for k,v in previous.items() if k.startswith(('G','P')) and v in ('passed','not_passed')}
    if job:
        if not json.loads((B/'analysis/metrics.json').read_text())['step1_pass']:
            raise ValueError('Complete thickness derivative gate not passed')
        if job.get('requires_reverse_pass') and not json.loads((B/'AD/reverse_work/result.json').read_text())['passed']:
            raise ValueError('Complete reverse derivative gate not passed')
        for path,key in job.get('requires',[]):
            if not json.loads((B/path).read_text())[key]:raise ValueError(f'Gate not passed: {path}:{key}')
        if not all(hashlib.sha256((R/p).read_bytes()).hexdigest()==h for p,h in job.get('protected_sources',{}).items()):
            raise ValueError('Declared migration source changed')
    deadline=start+job['wall_budget_seconds'] if job else cfg['batch_started_unix']+cfg['budget']['first_thickness_total_seconds']
    bench=json.loads((B/job.get('benchmark_path','checks/benchmark.json')).read_text()) if job else json.loads((B/'checks/benchmark.json').read_text())
    if not bench['baseline']['pass']:raise ValueError('Model baseline gate not passed')
    dt=job.get('dt_s',bench['fixed_dt_s']) if job else bench['fixed_dt_s']
    steps=job.get('steps',bench['steps_to5pct']) if job else bench['steps_to5pct']
    cases=job['cases'] if job else [dict(label=label,params_mm=[t,.05],is_ad=ad,seed=[1.,0.]) for label,t,ad in
         [('AD/thickness',.5,True),('FD/t_plus_h',.5025,False),('FD/t_minus_h',.4975,False),
          ('FD/t_plus_half',.50125,False),('FD/t_minus_half',.49875,False)]]
    prediction=0.
    for case in cases:
        prefix=json.loads((B/case['initial_from']/'result.json').read_text()) if case.get('initial_from') else None
        if prefix:
            if not prefix['complete'] or parameters(prefix)!=parameters(case) or prefix['dt_s']!=dt or prefix['steps']>=steps:
                raise ValueError('Incompatible saved complete prefix')
            if case['is_ad'] and not (B/case['initial_from']/'final_tangent.npz').exists():raise ValueError('Missing full prefix tangent')
        remaining=steps-(prefix['steps'] if prefix else 0)
        prediction+=(bench['estimated_one_JVP_minutes'] if case['is_ad'] else bench['estimated_one_primal_minutes'])*remaining/bench['steps_to5pct']
    if prediction*60>deadline-time.time()-600:
        raise TimeoutError('Benchmark predicts this declared design check exceeds its budget')
    clock='execution_clock.json' if not job else f'execution_clock_{job_id}.json'
    write(B/'checks'/clock,dict(started_unix=start,deadline_unix=deadline,pid=os.getpid(),steps=steps,dt=dt,spec=spec))
    stop=[]
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda s,f:stop.append(s))
    block=model.block();primal=jax.jit(block)
    tangent=jax.jit(lambda s,ds,f,df,count:jax.jvp(lambda ss,ff:block(ss,ff,dt,count),(s,f),(ds,df)))
    observe_jvp=jax.jit(lambda s,ds,f,df:jax.jvp(model._observe,(s,f),(ds,df)))
    estimate=stability_estimator(model)
    results=[]
    for case in cases:
        label=case['label'];is_ad=case['is_ad']
        out=B/label;out.mkdir(parents=True,exist_ok=False);cs=time.time()
        params=jnp.array(parameters(case))
        if is_ad:
            fields,dfields=ready(jax.jvp(model.fields,(params,),(jnp.array(case['seed']),)))
            state,dstate=ready(jax.jvp(model.initial,(fields,),(dfields,)))
        else:fields=ready(model.fields(params));state=ready(model.initial(fields));dstate=None
        initial_bound=estimate(state,fields)
        if dt>initial_bound['allowed_dt_s']:raise ValueError('Fixed dt exceeds initial perturbed stability estimate')
        write(out/'initial_stability.json',initial_bound)
        rows=[];done=0;min_j=math.inf;min_required=math.inf;restart=None
        if case.get('initial_from'):
            source=B/case['initial_from'];prefix=json.loads((source/'result.json').read_text())
            state=load(source/'final_state.npz')
            if is_ad:dstate=load(source/'final_tangent.npz')
            if not math.isclose(float(state[-2]),prefix['steps']*dt,rel_tol=1e-10,abs_tol=1e-14):
                raise ValueError('Prefix time does not match fixed-step history')
            rows=prefix['path'].copy();done=prefix['steps']
            min_j=prefix['sampled_min_actual_J'];min_required=prefix['sampled_min_required_J']
            files=['result.json','final_state.npz']+(['final_tangent.npz'] if is_ad else [])
            restart=dict(source=case['initial_from'],steps=done,seed=case.get('seed'),
                         hashes={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in files},
                         interpretation='Reuses complete prefix state and its full parameter tangent, including accumulated work; not a frozen-state derivative')
            write(out/'restart.json',restart)
            bound=estimate(state,fields);write(out/'restart_stability.json',bound)
            if dt>bound['allowed_dt_s']:raise ValueError('Fixed dt exceeds restart stability estimate')
        print(json.dumps(dict(stage='case_started',case=label,parameters=parameters(case),steps=steps)),flush=True)
        status=dict(stage=f'{job_id}_running',active_jobs=[dict(pid=os.getpid(),case=label)],step1='passed' if job else 'running',step2='running' if job else 'pending',step2_reverse='passed' if job and job.get('requires_reverse_pass') else 'pending',new_reverse_AD=bool(job and job.get('requires_reverse_pass')),step3='pending',step4='pending',completed_cases=[x['case'] for x in results])
        status.update(phase_history)
        if job:status.update(job.get('status_fields',{}))
        write(B/'status.json',status)
        while done<steps:
            if stop:raise InterruptedError('Termination requested')
            if time.time()>deadline:raise TimeoutError('Declared design-check wall budget exhausted')
            milestones=job.get('milestones',[]) if job else []
            next_stop=min([m for m in milestones if m>done]+[steps])
            count=min(64,steps-done,next_stop-done)
            if is_ad:state,dstate=ready(tangent(state,dstate,fields,dfields,count))
            else:state=ready(primal(state,fields,dt,count))
            done+=count
            diag=np.asarray(model.diagnostics(state,fields));obs=np.asarray(model.observe(state,fields))
            if not valid(diag):
                save(out/'invalid_state.npz',state)
                write(out/'failure.json',dict(steps=done,diagnostics=diag.tolist(),reason='ActualJ/finite energy guard'))
                raise ValueError('Nonfinite or nonpositive actual J in required material')
            min_j=min(min_j,float(diag[1]));min_required=min(min_required,float(diag[2]))
            row=dict(time_s=float(state[-2]),compression=float(model.ex.motion(state[-2])),
                     work_N_mm=float(obs[0]),force_N=float(obs[1]),IE_N_mm=float(diag[0]),
                     all_J_min=float(diag[1]),required_positive_J_min=float(diag[2]),negative_void_J_points=int(diag[4]),
                     KE_N_mm=float(diag[6]),bottom_force_N=float(diag[8]),contact_energy_N_mm=float(diag[9]))
            if is_ad:
                _,grad=observe_jvp(state,dstate,fields,dfields)
                if not np.isfinite(np.asarray(grad)).all():
                    save(out/'invalid_tangent.npz',dstate)
                    raise ValueError('Nonfinite parameter tangent')
                if job and 'parameters' in case:
                    row.update(work_gradient_value=float(grad[0]),force_gradient_value=float(grad[1]))
                else:row.update(work_gradient_N=float(grad[0]),force_gradient_N_per_mm=float(grad[1]))
            rows.append(row)
            if len(rows)%8==0 or done==steps:
                write(out/'progress.json',dict(case=label,steps=done,total_steps=steps,wall_s=time.time()-cs,**row))
            if len(rows)%16==0:
                print(json.dumps(dict(case=label,steps=done,total_steps=steps,compression=row['compression'],wall_s=time.time()-cs)),flush=True)
        save(out/'final_state.npz',state)
        if is_ad:save(out/'final_tangent.npz',dstate)
        terminal_bound=estimate(state,fields);write(out/'terminal_stability.json',terminal_bound)
        if dt>terminal_bound['allowed_dt_s']:raise ValueError('Fixed dt exceeds terminal perturbed stability estimate')
        result=dict(case=label,seed=case.get('seed'),execution_spec=spec,complete=True,steps=done,dt_s=dt,restart=restart,
                    terminal=rows[-1],sampled_min_actual_J=min_j,sampled_min_required_J=min_required,
                    initial_stability=initial_bound,terminal_stability=terminal_bound,wall_s=time.time()-cs,path=rows)
        if 'parameters' in case:
            result.update(parameters=parameters(case),parameter_names=job['parameter_names'],parameter_units=job['parameter_units'],gradient_units=job['gradient_units'])
        else:result['params_mm']=parameters(case)
        write(out/'result.json',result);results.append(result)
        print(json.dumps({k:v for k,v in result.items() if k!='path'}),flush=True)
    status=dict(stage=f'{job_id}_paths_complete_analysis_pending',active_jobs=[],step1='passed' if job else 'paths_complete',step2='paths_complete' if job else 'pending',step3='pending',step4='pending',wall_s=time.time()-start)
    status.update(phase_history)
    if job:status.update(job.get('status_fields',{}))
    write(B/'status.json',status)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['benchmark','execute']);ap.add_argument('--spec');args=ap.parse_args()
    try:
        benchmark(args.spec) if args.mode=='benchmark' else execute(args.spec)
    except (Exception,KeyboardInterrupt) as exc:
        suffix=Path(args.spec).stem if args.spec else args.mode
        write(B/f'failure_{suffix}.json',dict(type=type(exc).__name__,message=str(exc)))
        status=json.loads((B/'status.json').read_text())
        status.update(stage='stopped_'+suffix,active_jobs=[],reason=str(exc),step3='gated_not_started',step4='gated_not_started')
        job=json.loads((B/args.spec).read_text()) if args.spec else {}
        status[job.get('plan_step','step2' if args.spec else 'step1')]='not_passed'
        write(B/'status.json',status)
        raise

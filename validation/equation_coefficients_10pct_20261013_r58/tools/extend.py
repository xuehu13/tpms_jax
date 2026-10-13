"""Bounded 10% continuation of the frozen r57 coefficient experiment."""
from pathlib import Path
import argparse,hashlib,json,math,os,shutil,signal,sys,time,traceback
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
R=Path('/home/xuehu/projects/tpms_jax')
S=R/'validation/equation_coefficients_20261012_r57'
B=R/'validation/equation_coefficients_10pct_20261013_r58'
W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
O=W/'output/r58_equation_coefficients_10pct_20261013'

def read(p):return json.loads(p.read_text())
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def prepare():
    import numpy as np
    assert not B.exists(),'Never overwrite an existing experiment'
    assert read(S/'status.json')['final_acceptance_passed']
    assert read(S/'analysis/metrics.json')['final_acceptance_passed']
    oldp=read(S/'input/protocol.json');oldex=read(S/'input/execution.json')
    original=read(S/'checks/source_guard_before.json')
    assert all(sha(R/p)==h for p,h in original.items()),'Protected original source changed'
    assert all(sha(S/'tools'/p)==h for p,h in oldex['source_hashes'].items()),'Frozen r57 code changed'
    assert oldp['baseline']==[-3.4,4.6,2.] and oldp['relative_fd_step']==1e-5
    for rel in ['input','tools','checks','AD','FD','analysis','history']:(B/rel).mkdir(parents=True,exist_ok=True)
    shutil.copy2(Path(__file__),B/'tools/extend.py')
    prefix=read(S/'AD/c2/result.json');dt=oldex['dt_s'];t0=prefix['terminal']['time_s'];end=.002
    full=int(math.floor((end-t0)/dt));tail=end-(t0+full*dt)
    extension=[];left=full
    while left:
        count=min(256,left);extension.append(dict(count=count,dt_s=dt));left-=count
    if tail>0:extension.append(dict(count=1,dt_s=tail))
    oldblocks=[dict(state_path=str(S/'AD/c2/checkpoints'/f'{i:04d}.npz'),count=n,dt_s=dt,source='r57_prefix') for i,n in enumerate(prefix['counts'])]
    assert sum(x['count'] for x in oldblocks)==oldex['steps']
    assert abs(t0-sum(x['count']*x['dt_s'] for x in oldblocks))<1e-12
    parameters=np.array(oldp['baseline']);h=np.array(oldp['parameter_scales'])*oldp['relative_fd_step']
    reused=[S/'AD/c2/result.json',S/'AD/c2/final_state.npz',S/'AD/c2/final_tangent.npz']+[Path(x['state_path']) for x in oldblocks]
    with np.load(S/'AD/c2/final_state.npz') as z:
        assert len(z.files)==8 and abs(float(z['s6'])-t0)<1e-15
        assert abs(float(z['s7'])-prefix['terminal']['work'])<1e-15
    for k,c in enumerate(oldp['parameters']):
        for name,delta in [('plus_h',h[k]),('minus_h',-h[k]),('plus_half',h[k]/2),('minus_half',-h[k]/2)]:
            folder=S/'FD'/(c+'_'+name);row=read(folder/'result.json')
            assert row['complete'] and row['terminal']['steps']==oldex['steps'] and np.array_equal(row['parameters'],parameters+delta*np.eye(3)[k])
            with np.load(folder/'final_state.npz') as z:
                assert len(z.files)==8 and abs(float(z['s6'])-t0)<1e-15
                assert abs(float(z['s7'])-row['terminal']['work'])<1e-15
            reused.extend([folder/'result.json',folder/'final_state.npz'])
    guard={str(R/p):v for p,v in original.items()}
    extra=list((S/'tools').glob('*.py'))+list((S/'input').glob('*'))+reused
    extra += [S/'status.json',S/'checks/geometry.json',S/'checks/benchmark.json',S/'AD/work_VJP_result.json',S/'analysis/metrics.json',
        R/'validation/platen_xy_gradients_20261011_r53/tools/run.py',R/'validation/platen_xy_G_20261010_r51/tools/platen_model.py',B/'tools/extend.py']
    for p in extra:
        if p.is_file():guard[str(p)]=sha(p)
    write(B/'checks/source_guard_before.json',guard)
    protocol=dict(authorization='2026-10-13 user: 都算一下压缩到10%的，0–0.5%压缩太小了',
        equation=oldp['equation'],parameters=oldp['parameters'],baseline=oldp['baseline'],parameter_scales=oldp['parameter_scales'],
        relative_fd_step=oldp['relative_fd_step'],thresholds=oldp['thresholds'],
        target_compression=.1,target_time_s=end,original_ramp='20% at4ms, unchanged; stop at2ms=10%',
        required_objective='complete0-to10pct loading-work VJP for all3 equation coefficients, each checked against two FD sizes; c2 complete JVP',
        auxiliary_objective='terminal force complete JVP c2; c1/c0 FD only',
        prefix_reuse='original r57 actual-geometry FD terminal states; c2 primal/tangent; original primal checkpoints; cumulative work and derivatives not reset',
        reverse_rule='reverse all extension then all original prefix blocks, accumulate field cotangents, one initial-state pullback, one parameter pullback',
        failure_policy='retain all partials; no tolerance relaxation, no FD ladder; attempt all12FD if dynamics valid; gate long VJP on c2work and work FD halving',
        total_wall_budget_s=28800,started_unix=time.time(),scope_excludes=['Abaqus','training','Git','heartbeat','other configurations','r53 resumption'])
    execution=dict(compression=.1,end_time_s=end,dt_s=dt,prefix_steps=oldex['steps'],prefix_time_s=t0,
        steps=oldex['steps']+sum(x['count'] for x in extension),extension_steps=sum(x['count'] for x in extension),
        prefix_blocks=oldblocks,extension_schedule=extension,blocks=oldblocks.copy(),
        expected_new_minutes=360.,estimated_upper_budget_minutes=480,source_prefix=str(S),frozen_unix=time.time())
    write(B/'input/protocol.json',protocol);write(B/'input/execution.json',execution)
    write(B/'status.json',dict(stage='prepared',active_jobs=[],target_compression=.1,heartbeat='PAUSED',r53='unchanged_paused',r57='unchanged_completed'))
    O.mkdir(exist_ok=True)
    for rel in ['input/protocol.json','input/execution.json','status.json']:
        dst=O/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(B/rel,dst)
    note='# r58：diverse_28三个方程系数的0–10%梯度验证\n\n独立10%批次已准备，尚无新通过结论。复用r57原0–0.5%的实际状态、切向和检查点；完整反向仍覆盖0–10%，不清零累计加载功。原4ms20%加载、步长、材料、网格和端缘模型均不变。\n\n预计新增约6小时，总预算8小时。主目标为三个系数完整加载功VJP对双档FD，c2另核对完整JVP和末端力。原门槛与失败保留；无heartbeat、Abaqus、训练或Git。\n\n正式WSL：'+str(B)+'；当前以status.json为准。完成后见analysis/README.md。\n'
    (B/'README.md').write_text(note);(O/'README.md').write_text(note)
    print(json.dumps(dict(batch=str(B),prefix_steps=oldex['steps'],extension_steps=execution['extension_steps'],full_steps=execution['steps'],tail_dt_s=tail,protected_files=len(guard))))

def execute():
    sys.path.insert(0,str(S/'tools'))
    from coefficient_model import CoefficientPath,jax,jnp,np
    from reverse import functions
    from execute import ready,save,load,check
    protocol=read(B/'input/protocol.json');execution=read(B/'input/execution.json');guard_hash=read(B/'checks/source_guard_before.json')
    started=protocol['started_unix'];deadline=started+protocol['total_wall_budget_s'];signals=[]
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda s,f:signals.append(s))
    def guard():
        if signals:raise InterruptedError('Termination requested')
        if time.time()>deadline:raise TimeoutError('Frozen8h total new wall budget exhausted')
    def status(stage,**kw):write(B/'status.json',dict(stage=stage,active_jobs=[dict(pid=os.getpid(),task=stage)],wall_s=time.time()-started,heartbeat='PAUSED',r53='unchanged_paused',r57='unchanged_completed',**kw))
    changed=[n for n,h in guard_hash.items() if sha(Path(n))!=h]
    if changed:raise ValueError('Frozen source/prefix changed: '+str(changed))
    status('continuation_preflight');model=CoefficientPath();p=np.array(protocol['baseline']);scales=np.array(protocol['parameter_scales']);h=scales*protocol['relative_fd_step']
    fields,dfields=ready(jax.jvp(model.fields,(jnp.asarray(p),),(jnp.array([0.,1.,0.]),)))
    state=load(S/'AD/c2/final_state.npz');dstate=load(S/'AD/c2/final_tangent.npz')
    outer,forward,backward=functions(model)
    tangent=jax.jit(lambda s,ds,f,df,dt,count:jax.jvp(lambda ss,ff:outer(ss,ff,dt,count),(s,f),(ds,df)))
    obs_jvp=jax.jit(lambda s,ds,f,df:jax.jvp(model._observe,(s,f),(ds,df)))
    value,derivative=ready(obs_jvp(state,dstate,fields,dfields));old=read(S/'AD/c2/result.json')['terminal']
    expected=np.array([old['work'],old['force']]);expected_d=np.array(old['gradient'])
    equivalence=float(np.max(np.abs(np.asarray(value)-expected)/np.maximum(np.abs(expected),1e-20)))
    tangent_equivalence=float(np.max(np.abs(np.asarray(derivative)-expected_d)/np.maximum(np.abs(expected_d),1e-20)))
    write(B/'checks/prefix_equivalence.json',dict(response_relative=equivalence,JVP_relative=tangent_equivalence,passed=bool(max(equivalence,tangent_equivalence)<1e-8)))
    if max(equivalence,tangent_equivalence)>=1e-8:raise ValueError('Recreated coefficient model differs from frozen prefix')
    # Recheck the finite extension branch without counting it as a full path.
    seed=jnp.array([.3,-.7,.2]);_,df=ready(jax.jvp(model.fields,(jnp.asarray(p),),(seed,)))
    rng=np.random.default_rng(58);ds=tuple(jnp.asarray(rng.normal(size=x.shape))*1e-6 for x in state)
    dt=execution['dt_s'];ready(forward(state,fields,dt,256));t=time.time();ready(forward(state,fields,dt,256));fr=time.time()-t
    _,dy=ready(tangent(state,ds,fields,df,dt,256));t=time.time();_,dy=ready(tangent(state,ds,fields,df,dt,256));jr=time.time()-t
    cot=tuple(jnp.asarray(rng.normal(size=x.shape)) for x in state)
    ready(backward(state,fields,dt,256,cot));t=time.time();vs,vf=ready(backward(state,fields,dt,256,cot));br=time.time()-t
    terms=[float(jnp.vdot(a,b)) for a,b in zip(jax.tree_util.tree_leaves((vs,vf)),jax.tree_util.tree_leaves((ds,df)))];expected=sum(float(jnp.vdot(a,b)) for a,b in zip(dy,cot))
    dual=abs(sum(terms)-expected)/max(abs(expected),sum(abs(x) for x in terms)*1e-10,1e-20)
    extblocks=len(execution['extension_schedule']);allblocks=extblocks+len(execution['prefix_blocks'])
    estimate=(12*fr+jr)*extblocks+br*allblocks+600
    bench=dict(forward256_s=fr,JVP256_s=jr,VJP256_s=br,short_duality_relative=dual,passed=bool(dual<1e-7),estimated_remaining_s=estimate,
        scope='256-step duality at reused0.5pct state; not complete gradient validation')
    write(B/'checks/benchmark.json',bench)
    if not bench['passed']:raise ValueError('Continuation short duality failed')
    if estimate>deadline-time.time():raise TimeoutError('Measured cost exceeds frozen8h budget; no endpoint reduction')
    def observe(s,f,done,wall,gradient=None):
        diag=np.asarray(model.diagnostics(s,f));value=np.asarray(model.observe(s,f))
        row=dict(steps=done,time_s=float(s[-2]),compression=float(model.ex.motion(s[-2])),work=float(value[0]),force=float(value[1]),
            min_required_J=float(diag[2]),min_actual_J=float(diag[1]),KE=float(diag[6]),IE=float(diag[0]),wall_s=wall)
        if gradient is not None:row['gradient']=np.asarray(gradient).tolist()
        if not (np.isfinite(diag).all() and np.isfinite(value).all() and diag[2]>0 and diag[3]==0 and diag[5]>0):
            save(B/'checks/invalid_state.npz',s);write(B/'checks/invalid_state.json',row);raise ValueError('Actual-J/finite-energy guard failed')
        return row
    out=B/'AD/c2';out.mkdir();(out/'checkpoints').mkdir();base=read(S/'AD/c2/result.json');rows=base['path'].copy();counts=base['counts'].copy();done=execution['prefix_steps'];cs=time.time()
    status('full_JVP_c2_extension',prefix_reused_steps=done,total_steps=execution['steps'])
    for i,block in enumerate(execution['extension_schedule']):
        guard();file=out/'checkpoints'/f'{i:04d}.npz';save(file,state)
        execution['blocks'].append(dict(state_path=str(file),count=block['count'],dt_s=block['dt_s'],source='r58_extension'))
        write(B/'input/execution.json',execution)
        state,dstate=ready(tangent(state,dstate,fields,dfields,block['dt_s'],block['count']));done+=block['count'];counts.append(block['count'])
        _,g=ready(obs_jvp(state,dstate,fields,dfields));row=observe(state,fields,done,time.time()-cs,g);rows.append(row)
        if i%4==0 or done==execution['steps']:write(out/'progress.json',row);print(json.dumps(dict(case='JVP_c2',**row)),flush=True)
    save(out/'final_state.npz',state);save(out/'final_tangent.npz',dstate)
    jres=dict(complete=True,terminal=rows[-1],counts=counts,path=rows,wall_s=time.time()-cs,extension_wall_s=time.time()-cs,prefix_source=str(S/'AD/c2/result.json'),prefix_reused=True)
    write(out/'result.json',jres);grad_c2=np.asarray(g);base_values=np.asarray(model.observe(state,fields));primal_terminal=state;fd_results={}
    for k in [1,0,2]:
        for name,delta in [('plus_h',h[k]),('minus_h',-h[k]),('plus_half',h[k]/2),('minus_half',-h[k]/2)]:
            guard();label=protocol['parameters'][k]+'_'+name;folder=B/'FD'/label;folder.mkdir();source=S/'FD'/label;old=read(source/'result.json')
            q=p+delta*np.eye(3)[k];assert np.array_equal(q,old['parameters']);ff=ready(model.fields(jnp.asarray(q)));s=load(source/'final_state.npz')
            before=np.asarray(ready(model.observe(s,ff)));expected=np.array([old['terminal']['work'],old['terminal']['force']])
            if np.max(np.abs(before-expected)/np.maximum(np.abs(expected),1e-20))>=1e-8:raise ValueError('Recreated FD prefix differs for '+label)
            done=execution['prefix_steps'];path=old['path'].copy();cs=time.time();status('FD_'+label,completed_FD=list(fd_results),total_steps=execution['steps'])
            for i,block in enumerate(execution['extension_schedule']):
                guard();s=ready(forward(s,ff,block['dt_s'],block['count']));done+=block['count'];row=observe(s,ff,done,time.time()-cs);path.append(row)
                if i%4==0 or done==execution['steps']:write(folder/'progress.json',row)
            save(folder/'final_state.npz',s);res=dict(complete=True,parameters=q.tolist(),terminal=path[-1],path=path,wall_s=time.time()-cs,prefix_source=str(source/'result.json'),prefix_reused=True)
            write(folder/'result.json',res);fd_results[label]=res;print(json.dumps(dict(case=label,**path[-1])),flush=True)
        if k==1:
            a,b,c,d=[fd_results['c2_'+n]['terminal'] for n in ['plus_h','minus_h','plus_half','minus_half']]
            checks={obj:check(grad_c2[i],(a[obj]-b[obj])/(2*h[1]),(c[obj]-d[obj])/h[1],scales[1]) for i,obj in enumerate(['work','force'])}
            write(B/'checks/full_JVP_c2.json',checks)
    stable=[]
    for k,c in enumerate(protocol['parameters']):
        a,b,aa,bb=[fd_results[c+'_'+n]['terminal']['work'] for n in ['plus_h','minus_h','plus_half','minus_half']]
        fine=(aa-bb)/h[k];stable.append(dict(coefficient=c,**check(fine,(a-b)/(2*h[k]),fine,scales[k])))
    write(B/'checks/work_FD_halving.json',stable)
    if not read(B/'checks/full_JVP_c2.json')['work']['passed'] or not all(x['passed'] for x in stable):raise ValueError('Full10pct work JVP/FD or FD-halving gate failed; long VJP not started')
    assert len(execution['blocks'])==len(counts) and sum(counts)==execution['steps']
    status('full_work_VJP',includes_original_prefix_reverse=True,total_blocks=len(counts));reverse_start=time.time()
    cot=tuple(jnp.zeros_like(x) for x in primal_terminal[:-1])+(jnp.array(1.),);fcot=jax.tree_util.tree_map(jnp.zeros_like,fields)
    for i in range(len(execution['blocks'])-1,-1,-1):
        guard();block=execution['blocks'][i];saved=load(Path(block['state_path']))
        cot,g=ready(backward(saved,fields,block['dt_s'],block['count'],cot));fcot=ready(jax.tree_util.tree_map(lambda x,y:x+y,fcot,g))
        if i%4==0:write(B/'AD/work_VJP_progress.json',dict(blocks_left=i,total_blocks=len(counts),wall_s=time.time()-reverse_start,block_source=block['source']))
    _,initback=jax.vjp(model.initial,fields);fcot=jax.tree_util.tree_map(lambda x,y:x+y,fcot,initback(cot)[0])
    _,pback=jax.vjp(model.fields,jnp.asarray(p));gradient=np.asarray(ready(pback(fcot))[0]);checks=[]
    for k,c in enumerate(protocol['parameters']):
        a,b,aa,bb=[fd_results[c+'_'+n]['terminal']['work'] for n in ['plus_h','minus_h','plus_half','minus_half']]
        checks.append(dict(coefficient=c,**check(gradient[k],(a-b)/(2*h[k]),(aa-bb)/h[k],scales[k])))
    dualcheck=check(gradient[1],grad_c2[0],grad_c2[0],scales[1])
    result=dict(complete=True,objective='full_path_loading_work',gradient=gradient.tolist(),gradient_units='N mm per dimensionless coefficient',baseline_work=float(base_values[0]),baseline_force=float(base_values[1]),checks=checks,
        c2_VJP_JVP_relative=float(abs(gradient[1]-grad_c2[0])/max(abs(grad_c2[0]),1e-30)),c2_VJP_JVP_check=dualcheck,
        common_scaling_gradient_contraction=float(gradient@p),reverse_wall_s=time.time()-reverse_start,total_wall_s=time.time()-started,
        passed=bool(all(x['passed'] for x in checks) and dualcheck['passed']),compression=.1,steps=execution['steps'],includes_original_prefix_reverse=True)
    write(B/'AD/work_VJP_result.json',result)
    write(B/'status.json',dict(stage='completed' if result['passed'] else 'completed_not_passed',active_jobs=[],result='AD/work_VJP_result.json',heartbeat='PAUSED',r53='unchanged_paused',r57='unchanged_completed'))
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','run']);args=ap.parse_args()
    try:prepare() if args.mode=='prepare' else execute()
    except BaseException as exc:
        if B.exists():
            write(B/'failure.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
            write(B/'status.json',dict(stage='stopped',active_jobs=[],reason=str(exc),heartbeat='PAUSED',r53='unchanged_paused',r57='unchanged_completed'))
        raise

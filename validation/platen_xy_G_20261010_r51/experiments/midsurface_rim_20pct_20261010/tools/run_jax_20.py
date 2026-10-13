"""One authorized rimline20 path; reuse r51 mechanics, preserve old sources."""
from pathlib import Path
import json,math,os,signal,sys,time
START=time.time()
E=Path(__file__).resolve().parents[1]
D=E.parents[1]
sys.path.insert(0,str(D/'tools'))
from run_jax import build,estimate_initial_dt,jax,jnp,np

def write(p,x):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)

def save_state(out,state,path,name='last_valid_state.npz'):
    p=out/name;tmp=p.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f,q=np.asarray(state[0]),macro=np.asarray(state[1]),vq=np.asarray(state[2]),va=np.asarray(state[3]),
                            aq=np.asarray(state[4]),aa=np.asarray(state[5]),time=float(state[-1]))
    os.replace(tmp,p)
    write(out/'path_latest.json',path)

def main():
    cfg=json.loads((E/'input/protocol.json').read_text());clock=json.loads((E/'checks/execution_clock.json').read_text())
    deadline=min(START+3300,clock['started_unix']+5400-300)
    if deadline<=time.time()+60:raise TimeoutError('Insufficient remaining pair budget')
    out=E/'jax';out.mkdir(exist_ok=False)
    write(out/'pid.json',{'pid':os.getpid(),'driver':str(Path(__file__).resolve()),'started_unix':START,'deadline_unix':deadline})
    stop=[]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda s,f:stop.append(s))
    ex=build(cfg);dt,bound=estimate_initial_dt(ex)
    end=cfg['load_time_s']+cfg['hold_time_s'];dt=end/math.ceil(end/dt);initial_dt=dt
    state=ex.initial_state();advance=ex.advance_block()
    diag={**cfg,'initial_dt_s':dt,'initial_stability_estimate':bound,'deadline_unix':deadline,
          'projected_inertia_schur':np.asarray(ex.schur).tolist(),'mass_conservation_error':ex.mass_conservation_error,
          'devices':[str(x) for x in jax.devices()],'sample_validation':'actual energy/J at each accepted64-step output block, not a proof at every step'}
    write(out/'input.json',diag)
    warm=advance(state,dt,1);jax.block_until_ready(warm)
    path=[ex.observe_free(state)];steps=0;reject=[];checks=[];next_cp=0;targets=[.05,.10,.15,.20]
    rng=np.random.default_rng(51)
    @jax.jit
    def tangent(dq,da,q,a,t):
        v=jax.jvp(lambda x,y:ex._acceleration_free(x,y,t),(q,a),(dq,da))[1]
        return -v[0],-v[1]
    def current_estimate(s):
        dq=jnp.asarray(rng.normal(size=s[0].shape));da=jnp.asarray(rng.normal(size=(2,)));lam=0.
        for _ in range(25):
            vf=ex.full_velocity(dq,da);norm=jnp.sqrt(jnp.sum(ex.nodem[:,None]*vf**2))
            dq,da=tangent(dq/norm,da/norm,s[0],s[1],s[-1])
            vf=ex.full_velocity(dq,da);lam=float(jnp.sqrt(jnp.sum(ex.nodem[:,None]*vf**2)))
        if not math.isfinite(lam) or lam<=0:raise ValueError('Invalid current tangent frequency estimate')
        return min(float(ex.dt_estimate)*.5,.6/math.sqrt(lam)),lam
    last_save=last_print=time.time();save_state(out,state,path)
    print(json.dumps({'stage':'running','initial_dt_s':dt,'estimated_steps':math.ceil(end/dt),'preparation_s':time.time()-START}),flush=True)
    status='complete';failure=None
    try:
        while float(state[-1])<end-1e-12:
            if stop:raise InterruptedError('Termination requested')
            if time.time()>deadline:raise TimeoutError('JAX or overall wall budget reached')
            count=min(64,max(1,math.ceil((end-float(state[-1]))/dt-1e-9)))
            local=min(dt,(end-float(state[-1]))/count)
            trial=advance(state,local,count);jax.block_until_ready(trial)
            try:row=ex.observe_free(trial)
            except ValueError as exc:
                reject.append({'time':float(state[-1]),'dt_s':dt,'reason':str(exc)})
                write(out/'rejections.json',reject)
                if dt<=initial_dt/16:raise
                dt*=.5;continue
            state=trial;steps+=count;row['dt_s']=local;path.append(row)
            row['rim_contact_penetration_mm']=row['cap_penetration_phi_ge_0p01_mm']
            progress={'steps':steps,'wall_seconds':time.time()-START,'remaining_budget_seconds':deadline-time.time(),'dt_s':dt,'last':row}
            write(out/'progress.json',progress)
            if next_cp<len(targets) and row['compression']>=targets[next_cp]-1e-12:
                target=targets[next_cp];save_state(out,state,path,f'state_{round(target*100):02d}pct.npz')
                check_start=time.time()
                allowed,lam=current_estimate(state)
                check={'target_compression':target,'actual_compression':row['compression'],'time':row['time'],
                       'allowed_dt_s':allowed,'old_dt_s':dt,'omega2_estimate':lam,'not_a_proven_global_bound':True,
                       'wall_seconds':time.time()-check_start}
                checks.append(check);write(out/'stability_checks.json',checks)
                if allowed<initial_dt/16:raise ValueError('Current estimated stable step below original reduction floor')
                dt=min(dt,allowed);next_cp+=1
            now=time.time()
            if now-last_save>45:save_state(out,state,path);last_save=now
            if now-last_print>30:print(json.dumps(progress),flush=True);last_print=now
    except (Exception,KeyboardInterrupt) as exc:
        status=type(exc).__name__;failure={'type':status,'message':str(exc)};write(out/'failure.json',failure)
    save_state(out,state,path)
    complete=abs(float(state[-1])-end)<1e-10
    result={'status':status,'complete':complete,'full20':complete and path[-1]['compression']>=.20-1e-12,
            'new_design_AD':False,'steps':steps,'wall_seconds':time.time()-START,'initial_dt_s':initial_dt,'terminal_dt_s':dt,
            'rejections':reject,'current_stability_checks':checks,'failure':failure,'path':path,
            'scope':cfg['stage'],'sample_validation':diag['sample_validation']}
    write(out/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='path'},indent=2),flush=True)
    if not complete:raise SystemExit(1)

if __name__=='__main__':
    try:main()
    except (Exception,KeyboardInterrupt) as exc:
        out=E/'jax'
        if out.exists():write(out/'startup_or_driver_failure.json',{'type':type(exc).__name__,'message':str(exc),'wall_seconds':time.time()-START})
        raise

"""Bounded P/D experiment loop. Material, force, mass and Verlet are r51 imports."""
from pathlib import Path
import argparse, importlib.util, json, math, signal, sys, time
START=time.time()
B=Path(__file__).resolve().parents[1];R=B.parents[1];G=R/'validation/platen_xy_G_20261010_r51'
sys.path.insert(0,str(G/'tools'))
spec=importlib.util.spec_from_file_location('r51_rim20_recording',G/'experiments/midsurface_rim_20pct_20261010/tools/run_jax_20.py')
record=importlib.util.module_from_spec(spec);spec.loader.exec_module(record)
build,estimate_initial_dt,jax,jnp,np=record.build,record.estimate_initial_dt,record.jax,record.jnp,record.np
write,save_state=record.write,record.save_state

def main():
    ap=argparse.ArgumentParser();ap.add_argument('kind',choices=['P','D']);args=ap.parse_args()
    E=B/args.kind;cfg=json.loads((E/'input/protocol.json').read_text());clock=json.loads((E/'checks/execution_clock.json').read_text())
    deadline=min(START+3900,clock['started_unix']+6000-300)
    if deadline<=time.time()+60:raise TimeoutError('Insufficient remaining pair budget')
    out=E/'jax';out.mkdir(exist_ok=False);write(out/'pid.json',{'pid':__import__('os').getpid(),'started_unix':START,'deadline_unix':deadline})
    stop=[]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda s,f:stop.append(s))
    ex=build(cfg);dt,bound=estimate_initial_dt(ex)
    end=cfg['load_time_s']+cfg['hold_time_s'];dt=end/math.ceil(end/dt);initial_dt=dt
    state=ex.initial_state();advance=ex.advance_block()
    diag={**cfg,'initial_dt_s':dt,'initial_stability_estimate':bound,'deadline_unix':deadline,
          'projected_inertia_schur':np.asarray(ex.schur).tolist(),'mass_conservation_error':ex.mass_conservation_error,
          'devices':[str(x) for x in jax.devices()],'sample_validation':'actual energy/J at each accepted64-step sample; not every-step/global certification'}
    write(out/'input.json',diag);warm=advance(state,dt,1);jax.block_until_ready(warm)
    path=[ex.observe_free(state)];steps=0;reject=[];checks=[];next_cp=0;targets=[.05,.10,.15,.20];rng=np.random.default_rng(51)
    @jax.jit
    def tangent(dq,da,q,a,t):
        v=jax.jvp(lambda x,y:ex._acceleration_free(x,y,t),(q,a),(dq,da))[1];return -v[0],-v[1]
    def current_estimate(s):
        dq=jnp.asarray(rng.normal(size=s[0].shape));da=jnp.asarray(rng.normal(size=(2,)));lam=0.
        for _ in range(25):
            vf=ex.full_velocity(dq,da);norm=jnp.sqrt(jnp.sum(ex.nodem[:,None]*vf**2))
            dq,da=tangent(dq/norm,da/norm,s[0],s[1],s[-1]);vf=ex.full_velocity(dq,da);lam=float(jnp.sqrt(jnp.sum(ex.nodem[:,None]*vf**2)))
        if not math.isfinite(lam) or lam<=0:raise ValueError('Invalid current frequency estimate')
        return min(float(ex.dt_estimate)*.5,.6/math.sqrt(lam)),lam
    last_save=last_print=time.time();save_state(out,state,path)
    print(json.dumps({'kind':args.kind,'stage':'running','initial_dt_s':dt,'estimated_steps':math.ceil(end/dt),'preparation_s':time.time()-START}),flush=True)
    status='complete';failure=None
    try:
        while float(state[-1])<end-1e-12:
            if stop:raise InterruptedError('Termination requested')
            if time.time()>deadline:raise TimeoutError('JAX or overall wall budget reached')
            count=min(64,max(1,math.ceil((end-float(state[-1]))/dt-1e-9)));local=min(dt,(end-float(state[-1]))/count)
            trial=advance(state,local,count);jax.block_until_ready(trial)
            try:row=ex.observe_free(trial)
            except ValueError as exc:
                reject.append({'time':float(state[-1]),'dt_s':dt,'reason':str(exc)});write(out/'rejections.json',reject)
                if dt<=initial_dt/16:raise
                dt*=.5;continue
            state=trial;steps+=count;row['dt_s']=local;row['rim_contact_penetration_mm']=row['cap_penetration_phi_ge_0p01_mm'];path.append(row)
            progress={'kind':args.kind,'steps':steps,'wall_seconds':time.time()-START,'remaining_budget_seconds':deadline-time.time(),'dt_s':dt,'last':row}
            write(out/'progress.json',progress)
            if next_cp<len(targets) and row['compression']>=targets[next_cp]-1e-12:
                target=targets[next_cp];save_state(out,state,path,f'state_{round(target*100):02d}pct.npz');cs=time.time();allowed,lam=current_estimate(state)
                checks.append({'target_compression':target,'actual_compression':row['compression'],'time':row['time'],'allowed_dt_s':allowed,
                    'old_dt_s':dt,'omega2_estimate':lam,'not_a_proven_global_bound':True,'wall_seconds':time.time()-cs})
                write(out/'stability_checks.json',checks)
                if allowed<initial_dt/16:raise ValueError('Current estimated stable step below original reduction floor')
                dt=min(dt,allowed);next_cp+=1
            now=time.time()
            if now-last_save>45:save_state(out,state,path);last_save=now
            if now-last_print>30:print(json.dumps(progress),flush=True);last_print=now
    except (Exception,KeyboardInterrupt) as exc:
        status=type(exc).__name__;failure={'type':status,'message':str(exc)};write(out/'failure.json',failure)
    save_state(out,state,path);complete=abs(float(state[-1])-end)<1e-10
    result={'status':status,'complete':complete,'full20':complete and path[-1]['compression']>=.20-1e-12,'new_design_AD':False,
        'steps':steps,'wall_seconds':time.time()-START,'initial_dt_s':initial_dt,'terminal_dt_s':dt,'rejections':reject,
        'current_stability_checks':checks,'failure':failure,'path':path,'scope':cfg['stage'],'sample_validation':diag['sample_validation']}
    write(out/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k!='path'}),flush=True)
    if not complete:raise SystemExit(1)

if __name__=='__main__':
    try:main()
    except (Exception,KeyboardInterrupt) as exc:
        E=B/sys.argv[1];out=E/'jax'
        if out.exists():write(out/'startup_or_driver_failure.json',{'type':type(exc).__name__,'message':str(exc),'wall_seconds':time.time()-START})
        raise

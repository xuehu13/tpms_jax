"""One G short path; bounded runtime, atomic valid-state recording."""
from pathlib import Path
import argparse,json,math,time,sys,hashlib
import numpy as np
from platen_model import PlatenXY,R,jax,jnp
from hyperelastic_fem import make_density_hyperelastic_problem
from prepare import face_cache
D=Path(__file__).resolve().parents[1]
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def build(cfg,uniform=False):
    if uniform:field=lambda p:np.ones((p.fe.num_cells,p.fe.num_quads));contact=face_cache(cfg['N'])
    else:
        with np.load(cfg['original_Gauss_field']) as z:rho=z['rho'];qp=z['physical_quad_points']
        def field(p):
            np.testing.assert_array_equal(np.asarray(p.physical_quad_points),qp);return rho
        with np.load(Path(cfg.get('contact_cache_path',D/'input/cut_face_contact.npz'))) as z:contact=dict(z)
    with jax.default_device(jax.devices('cpu')[0]):
        p=make_density_hyperelastic_problem(cfg['N'],rho_quad=field,eta=cfg['eta'],periodic_axes=(0,1,2),element_degree=2,quadrature_order=4,material_model='objective_void')
    return PlatenXY(p,cfg,contact)

def estimate_initial_dt(ex):
    """Mass-weighted power estimate with both caps active; conservative safety."""
    q=jnp.zeros((ex.nc,3)).at[:,2].set(1e-6*(jnp.asarray(ex.p.fe.points)[np.unique(np.asarray(ex.ids),return_index=True)[1],2]-.5))
    a=jnp.zeros(2);t=jnp.array(0.)
    @jax.jit
    def apply(dq,da):
        # Derivative of acceleration = -M^-1 K. Generalized mass norm below.
        v=jax.jvp(lambda x,y:ex._acceleration_free(x,y,t),(q,a),(dq,da))[1]
        return -v[0],-v[1]
    def norm(x,y):
        vf=ex.full_velocity(x,y);return jnp.sqrt(jnp.sum(ex.nodem[:,None]*vf**2))
    rng=np.random.default_rng(51);dq=jnp.asarray(rng.normal(size=q.shape));da=jnp.asarray(rng.normal(size=(2,)))
    lam=0.
    for _ in range(25):
        scale=norm(dq,da);dq,da=apply(dq/scale,da/scale);lam=float(norm(dq,da))
    dt=.3*2/math.sqrt(lam)
    return min(float(ex.dt_estimate)*.5,dt),{'power_estimated_omega2_s_minus2':lam,'power_steps':25,'safety_fraction':.3,'not_a_proven_global_bound':True}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--uniform-check',action='store_true');ap.add_argument('--config',type=Path);ap.add_argument('--out',type=Path);a=ap.parse_args()
    if a.uniform_check and (a.config or a.out):raise ValueError('Keep the original uniform check isolated')
    cfg=json.loads((a.config or D/'input/protocol.json').read_text());start=time.monotonic()
    if a.uniform_check:
        cfg={**cfg,'N':2,'target_compression':1e-4,'load_time_s':.001,'hold_time_s':.0005,
             'contact_normal_stiffness_MPa_per_mm':320.}
        out=D/'checks/uniform_contact'
    else:out=a.out.resolve() if a.out else D/'jax/short_2pct'
    if not out.is_relative_to(D):raise ValueError('Diagnostic output must stay inside r51')
    out.mkdir(exist_ok=False)
    if not a.uniform_check:
        assert json.loads((D/'checks/uniform_contact/result.json').read_text())['analytic_check_pass']
    ex=build(cfg,a.uniform_check)
    if a.uniform_check:
        rng=np.random.default_rng(51);nn=len(ex.points);ids=np.asarray(ex.ids);m=np.asarray(ex.nodem)
        B=np.zeros((3*nn,3*ex.nc+2));B[np.arange(3*nn),(3*ids[:,None]+np.arange(3)).ravel()]=1
        B[0::3,-2]=np.asarray(ex.b[:,0]);B[1::3,-1]=np.asarray(ex.b[:,1])
        mass=B.T@(np.repeat(m,3)[:,None]*B);rhs=rng.normal(size=3*ex.nc+2)
        sq,sa=ex.solve_mass(jnp.asarray(rhs[:-2].reshape(-1,3)),jnp.asarray(rhs[-2:]));sol=np.r_[np.asarray(sq).ravel(),np.asarray(sa)]
        mass_error=float(np.linalg.norm(mass@sol+rhs)/np.linalg.norm(rhs))
        q=jnp.zeros((ex.nc,3)).at[:,2].set(1e-4*(jnp.asarray(ex.p.fe.points)[np.unique(ids,return_index=True)[1],2]-.5))
        direction=jnp.asarray(rng.normal(size=q.shape));cr=ex.contact(q,0.)[0]
        analytic=float(jnp.sum(ex.reduce(cr)*direction));eps=1e-8
        fd=float((ex.contact(q+eps*direction,0.)[1]-ex.contact(q-eps*direction,0.)[1])/(2*eps))
        virtual_error=abs(analytic-fd)/max(abs(analytic),1e-30)
        separated=-q;no_contact_force=float(jnp.max(abs(ex.contact(separated,0.)[0])))
        checks={'mass_schur_relative_residual':mass_error,'contact_virtual_work_relative_error':virtual_error,
                'separated_contact_force':no_contact_force,'pass':mass_error<1e-10 and virtual_error<1e-6 and no_contact_force==0}
        write(D/'checks/algebra_contact.json',checks);assert checks['pass'],checks
    dt,bound=estimate_initial_dt(ex)
    state=ex.initial_state();advance=ex.advance_block();end=cfg['load_time_s']+cfg['hold_time_s']
    steps=math.ceil(end/dt)
    if 'diagnostic_fixed_steps' in cfg:
        requested_steps=int(cfg['diagnostic_fixed_steps'])
        requested_dt=end/requested_steps
        if requested_steps<=0 or requested_dt>dt*(1+1e-10):
            write(out/'admission_failure.json',{'stage':'stopped_before_dynamic_advance','requested_dt_s':requested_dt,'allowed_initial_dt_s':dt,'estimate':bound,'wall_seconds':time.monotonic()-start})
            raise ValueError('Requested diagnostic step exceeds the initial conservative estimate')
        bound={**bound,'automatic_initial_dt_s':dt,'diagnostic_fixed_steps':requested_steps}
        steps=requested_steps
    dt=end/steps;initial_dt=dt
    diag={**cfg,'dt_initial_s':dt,'projected_inertia_schur':np.asarray(ex.schur).tolist(),
          'schur_relative_diagonal':np.asarray(ex.schur/ex.d).tolist(),'mass_conservation_error':ex.mass_conservation_error,
          'stability_estimate':bound,'adapter_sha256':hashlib.sha256(Path(__file__).with_name('platen_model.py').read_bytes()).hexdigest()}
    write(out/'input.json',diag)
    # Compile once; discard the warm-up state.
    warm=advance(state,dt,1);jax.block_until_ready(warm)
    path=[ex.observe_free(state)];done=0;reject=[];budget=180 if a.uniform_check else 2350
    print(json.dumps({'stage':'running','dt':dt,'estimated_steps':steps,'preparation_seconds':time.monotonic()-start}),flush=True)
    try:
        while float(state[-1])<end-1e-12:
            if time.monotonic()-start>budget:raise TimeoutError('End-to-end wall budget reached')
            count=min(64,max(1,math.ceil((end-float(state[-1]))/dt-1e-9)))
            local=min(dt,(end-float(state[-1]))/count)
            trial=advance(state,local,count);jax.block_until_ready(trial)
            try:row=ex.observe_free(trial)
            except ValueError as error:
                reject.append({'time':float(state[-1]),'dt':dt,'reason':str(error)})
                write(out/'rejections.json',reject)
                if dt<=initial_dt/16:raise
                dt*=.5;continue
            state=trial;done+=count;path.append(row)
            write(out/'progress.json',{'steps':done,'wall_seconds':time.monotonic()-start,'dt':dt,'last':row})
            if len(path)%10==0:print(json.dumps(row),flush=True)
        status='complete_diagnostic'
    except (Exception,KeyboardInterrupt) as e:
        status=type(e).__name__;write(out/'failure.json',{'type':status,'message':str(e)})
    np.savez_compressed(out/'last_valid_state.npz',q=np.asarray(state[0]),macro=np.asarray(state[1]),vq=np.asarray(state[2]),va=np.asarray(state[3]),time=float(state[-1]))
    result={'status':status,'complete':abs(float(state[-1])-end)<1e-10,'steps':done,'wall_seconds':time.monotonic()-start,
            'initial_dt_s':initial_dt,'terminal_dt_s':dt,'rejections':reject,'path':path,'full20':False,'new_design_AD':False}
    if a.uniform_check and result['complete']:
        hold=[x for x in path if x['time']>=.00125];force=float(np.mean([x['force_N'] for x in hold]));reference=cfg['E_MPa']*100*cfg['target_compression']
        result.update(analytic_small_strain_force_N=reference,hold_force_N=force,force_relative_bias=force/reference-1,
            mean_lateral_strain=float(np.mean([(x['ax_mm']+x['ay_mm'])/20 for x in hold])),expected_lateral_strain=.3e-4)
        result['analytic_check_pass']=abs(force/reference-1)<.05 and abs(result['mean_lateral_strain']/.3e-4-1)<.1
    write(out/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k!='path'},indent=2),flush=True)
    if not result['complete'] or (a.uniform_check and not result['analytic_check_pass']):raise SystemExit(1)
if __name__=='__main__':main()

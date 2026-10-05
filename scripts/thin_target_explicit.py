"""Minimal XYZ physical Explicit experiment, reusing JAX-FEM NH residuals.

No tangent/global solve, contact, plasticity, mass scaling or damping is added.
The void has a declared numerical mass floor equal to its stiffness floor.
Pure central differences remain differentiable; path gradients are not certified.
"""
from pathlib import Path
import argparse, hashlib, json, math, os, resource, shutil, sys, time
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
ROOT=Path(__file__).resolve().parents[1]
if not (ROOT/'hyperelastic_fem.py').exists():ROOT=Path('/home/xuehu/projects/tpms_jax')
sys.path.insert(0,str(ROOT))
import numpy as np
import jax
import jax.numpy as jnp
from hyperelastic_fem import make_density_hyperelastic_problem,neo_hookean_energy,MU,KAPPA
jax.config.update('jax_enable_x64',True)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

class ExplicitXYZ:
    def __init__(self,p,L=10.,density=1e-9):
        self.p=p;self.L=L;self.points=jnp.asarray(p.fe.points)
        self.ids=jnp.asarray(p.class_ids,dtype=jnp.int32);self.pin=p.fixed_class_id
        self.nc=int(np.max(p.class_ids))+1;self.scale=p.stiffness_scale
        # m_norm = rho*L^2 integral N_i*(eta+(1-eta)*phi) dV_norm.
        cellmass=np.einsum('qn,cq,cq->cn',np.asarray(p.fe.shape_vals),
                           np.asarray(p.fe.JxW),np.asarray(self.scale))*density*L**2
        nodem=np.zeros(len(p.fe.points));np.add.at(nodem,np.asarray(p.fe.cells).ravel(),cellmass.ravel())
        self.nodem=jnp.asarray(nodem)
        self.mass=jnp.zeros(self.nc).at[self.ids].add(self.nodem)
        self.reduce=lambda r:jnp.zeros((self.nc,3)).at[self.ids].add(r)
        self.force=jax.jit(self._force)
        self.cp=math.sqrt((KAPPA+4*MU/3)/(density*L**2))
        self.dt_estimate=.2/(round(self.nc**(1/3))*self.cp)
        # These COO indices are only used for tangent assembly, never here.
        del p.I,p.J

    def _force(self,q,h):
        H=jnp.zeros((3,3)).at[2,2].set(h)
        internal=[jnp.broadcast_to(H,(*self.scale.shape,3,3)),self.scale]
        return self.p.compute_residual_vars([q[self.ids]],internal,[])[0]

    def acceleration(self,q,h,hdd):
        r=self._force(q,h);gd=self.points*jnp.array([0.,0.,hdd])
        acc=-(self.reduce(r)+self.reduce(self.nodem[:,None]*gd))/self.mass[:,None]
        return acc.at[self.pin].set(0.)

    def block(self,dt,steps,load_time,wave=False):
        def motion(t):
            if wave:return jnp.array([0.,0.,0.])
            s=jnp.clip(t/load_time,0.,1.)
            h=-.2*(10*s**3-15*s**4+6*s**5)
            hd=-.2*(30*s**2-60*s**3+30*s**4)/load_time
            hdd=-.2*(60*s-180*s**2+120*s**3)/load_time**2
            return jnp.array([h,hd,hdd])
        def one(carry,_):
            q,vhalf,t=carry;h,_,hdd=motion(t)
            vhalf=(vhalf+dt*self.acceleration(q,h,hdd)).at[self.pin].set(0.)
            q=(q+dt*vhalf).at[self.pin].set(0.)
            return (q,vhalf,t+dt),None
        return jax.jit(lambda state:jax.lax.scan(one,state,None,length=steps)[0]),motion

    def observe(self,state,dt,motion):
        q,vhalf,t=state;h,hd,hdd=motion(t)
        H=jnp.zeros((3,3)).at[2,2].set(h)
        w=q[self.ids];F=jnp.eye(3)+H+self.p.fe.sol_to_grad(w)
        J=jnp.linalg.det(F);minJ=float(J.min())
        if not np.isfinite(np.asarray(q)).all() or not math.isfinite(minJ) or minJ<=0:
            raise ValueError('Nonfinite displacement or nonpositive reference Gauss detF')
        r=self.force(q,h)
        gd=self.points*jnp.array([0.,0.,hdd])
        acc=(-(self.reduce(r)+self.reduce(self.nodem[:,None]*gd))/self.mass[:,None]).at[self.pin].set(0.)
        velocity=(vhalf+.5*dt*acc)[self.ids]+self.points*jnp.array([0.,0.,hd])
        fullacc=acc[self.ids]+self.points*jnp.array([0.,0.,hdd])
        weights=jnp.asarray(self.p.fe.JxW)
        W=jax.vmap(neo_hookean_energy)(F.reshape((-1,3,3))).reshape(self.scale.shape)
        U=float(jnp.sum(W*self.scale*weights))*self.L**3
        KE=float(.5*jnp.sum(self.nodem[:,None]*velocity**2))*self.L**3
        qstatic=float(jnp.sum(r[:,2]*self.points[:,2]))*self.L**2
        qdynamic=float(jnp.sum((r[:,2]+self.nodem*fullacc[:,2])*self.points[:,2]))*self.L**2
        return {'time':float(t),'compression':float(-h),'Fz_N':qdynamic,'internal_macro_Fz_N':qstatic,
                'energy_N_mm':U,'KE_N_mm':KE,'KE_over_U':KE/max(U,1e-30),'J_min':minJ}

def run(a):
    a.output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    if a.action=='wave':
        N=8;p=make_density_hyperelastic_problem(N,rho_quad=1.,eta=1e-4,periodic_axes=(0,1,2))
        cfg={'N':N,'role':'small known periodic P-wave integration check; not TPMS accuracy evidence'}
    else:
        cfg=json.loads((a.case/'step2/diagnostic_xyz/input.json').read_text());N=cfg['N']
        with np.load(a.case/'gauss_field.npz') as f:rho=f['rho'];qp=f['physical_quad_points'];qw=f['JxW']
        def field(p):
            assert np.array_equal(np.asarray(p.physical_quad_points),qp)
            assert np.array_equal(np.asarray(p.fe.JxW),qw)
            return rho
        p=make_density_hyperelastic_problem(N,rho_quad=field,eta=cfg['eta'],periodic_axes=(0,1,2))
        del rho,qp,qw
    ex=ExplicitXYZ(p);dt=ex.dt_estimate*(.25 if a.action=='wave' else 1.)
    cfg.update(method='physical central difference; existing JAX-FEM NH residual',dt_seconds=dt,
        cell_size_mm=10.,solid_density_tonne_per_mm3=1e-9,void_mass_floor=p.eta,
        mass_model='rho_s*(eta+(1-eta)*phi); numerical void mass, not a second physical material',
        affine_inertia_included=True,no_mass_scaling=True,no_contact=True,no_damping=True,
        source_sha256={n:sha(ROOT/n) for n in ('hyperelastic_fem.py','pbc.py','fem.py','pixi.lock')},
        experiment_sha256=sha(Path(__file__)),source_case=str(a.case),load_time_seconds=a.load_time,
        devices=[str(d) for d in jax.devices()])
    snap=a.output/'source_at_run';snap.mkdir();shutil.copy2(__file__,snap/'thin_target_explicit.py')
    q=jnp.zeros((ex.nc,3));wave=a.action=='wave'
    if wave:
        xyz=np.zeros((ex.nc,3));xyz[np.asarray(ex.ids)]=np.asarray(p.fe.points)%1.
        q=q.at[:,0].set(1e-7*jnp.sin(2*jnp.pi*jnp.asarray(xyz[:,0]))).at[ex.pin].set(0.)
        omega=2*ex.cp*N*math.sin(math.pi/N);end=2*math.pi/omega
    else:end=a.load_time*1.1
    steps=32 if a.action=='probe' else math.ceil(end/dt)
    if a.action!='probe':dt=end/steps
    else:end=steps*dt
    initial_dt=dt;rejections=[]
    cfg.update(dt_seconds=dt,total_steps=steps,adaptive_block_rejection=a.adaptive,
               minimum_dt_seconds=dt/16 if a.adaptive else dt)
    write(a.output/'input.json',cfg)
    acc0=ex.acceleration(q,0.,0.);state=(q,-.5*dt*acc0,jnp.asarray(0.))
    chunk=8 if a.action=='probe' else min(128,max(1,steps//100))
    advance,motion=ex.block(dt,chunk,a.load_time,wave)
    t0=time.perf_counter();warm=advance(state);jax.block_until_ready(warm)
    compile_seconds=time.perf_counter()-t0
    # Warm result is discarded; then every measured step belongs to this path.
    path=[ex.observe(state,dt,motion)];mode=[];walk=time.perf_counter();done=0
    if wave:
        shape=q[:,0];norm=jnp.sum(ex.mass*shape**2)
        modal=lambda s:float(jnp.sum(ex.mass*s[0][:,0]*shape)/norm)
        mode=[modal(state)]
    try:
        while float(state[2])<end-1e-12*end:
            remaining=math.ceil((end-float(state[2]))/dt-1e-9)
            remain=min(chunk,remaining)
            local_dt=(end-float(state[2]))/remain if remaining<=chunk else dt
            fn=advance if remain==chunk and local_dt==dt else ex.block(local_dt,remain,a.load_time,wave)[0]
            prior=state;trial=fn(prior);jax.block_until_ready(trial)
            try:row=ex.observe(trial,local_dt,motion)
            except ValueError as invalid:
                record={'time':float(prior[2]),'attempted_end_time':float(trial[2]),'dt':local_dt,
                        'reason':str(invalid),'last_valid':path[-1]}
                rejections.append(record)
                np.savez_compressed(a.output/f'rejected_block_{len(rejections):02d}.npz',
                                    q=np.asarray(trial[0]),vhalf=np.asarray(trial[1]),time=float(trial[2]))
                write(a.output/'rejected_blocks.json',rejections)
                print('REJECTED_BLOCK '+json.dumps(record),flush=True)
                if not a.adaptive or dt<=initial_dt/16:raise
                # Recover the same centered velocity before changing the half-step.
                h,_,hdd=motion(prior[2]);acc=ex.acceleration(prior[0],h,hdd)
                vcenter=prior[1]+.5*dt*acc
                dt*=.5;state=(prior[0],vcenter-.5*dt*acc,prior[2])
                advance,motion=ex.block(dt,chunk,a.load_time,wave)
                continue
            state=trial;done+=remain;row['dt_seconds']=local_dt;path.append(row)
            if wave:mode.append(modal(state))
            elapsed=time.perf_counter()-walk
            write(a.output/'progress.json',{'steps':done,'remaining_steps_estimate':math.ceil(max(0.,end-float(state[2]))/dt),
                'compression':row['compression'],'dt':dt,'rejected_blocks':len(rejections),
                'elapsed_seconds':elapsed,'J_min':row['J_min'],'KE_over_U':row['KE_over_U']})
            if a.action=='probe':continue
            if done%(chunk*10)==0:print(json.dumps(path[-1]),flush=True)
        cost=(time.perf_counter()-walk)/done
        result={'status':'complete_diagnostic','steps':done,'rejected_blocks':rejections,
            'initial_dt_seconds':initial_dt,'terminal_dt_seconds':dt,'compile_seconds':compile_seconds,
            'seconds_per_step_with_monitoring':cost,'body_seconds':time.perf_counter()-started,
            'path':path,'no_contact_diagnostic':True,'peak_RSS_GiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2}
        if wave:
            expected=np.cos(omega*np.array([r['time'] for r in path]))
            err=float(np.max(np.abs(np.array(mode)-expected)))
            e0=path[0]['energy_N_mm'];drift=max(abs((r['energy_N_mm']+r['KE_N_mm'])/e0-1) for r in path)
            result.update(modal_max_absolute_error=err,energy_relative_max_drift=drift,discrete_omega_rad_per_s=omega,
                          status='integration_check_pass' if err<.002 and drift<.005 else 'integration_check_failed')
        elif a.action=='probe':
            result.update(estimated_T0p020_seconds=math.ceil(.022/dt)*cost,
                          estimated_T0p004_seconds=math.ceil(.0044/dt)*cost)
        np.savez_compressed(a.output/'field.npz',q=np.asarray(state[0]),vhalf=np.asarray(state[1]),time=float(state[2]),N=N,dt=dt)
        write(a.output/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k!='path'},indent=2),flush=True)
    except Exception as exc:
        failed=trial if 'trial' in locals() else state
        np.savez_compressed(a.output/'failed_field.npz',q=np.asarray(failed[0]),vhalf=np.asarray(failed[1]),time=float(failed[2]))
        if 'prior' in locals():
            np.savez_compressed(a.output/'last_valid_field.npz',q=np.asarray(prior[0]),vhalf=np.asarray(prior[1]),time=float(prior[2]),N=N,dt=dt)
        write(a.output/'failure.json',{'type':type(exc).__name__,'message':str(exc),'completed_steps':done,
                                      'body_seconds':time.perf_counter()-started,'last_valid':path[-1],
                                      'accepted_path':path,'rejected_blocks':rejections})
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['wave','probe','target'])
    p.add_argument('--case',type=Path,default=ROOT/'validation/thin_target_20261004_r5')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--load-time',type=float,default=.02)
    p.add_argument('--adaptive',action='store_true',help='Reject invalid blocks and halve dt, keeping the previous valid state; no detF clipping')
    a=p.parse_args();assert a.load_time>0;run(a)

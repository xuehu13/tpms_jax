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
        self.mass_density_factor=density*L**2
        self.ids=jnp.asarray(p.class_ids,dtype=jnp.int32);self.pin=p.fixed_class_id
        self.nc=int(np.max(p.class_ids))+1;self.scale=p.stiffness_scale
        # m_norm = rho*L^2 integral N_i*(eta+(1-eta)*phi) dV_norm.
        shapes=np.asarray(p.fe.shape_vals);weights=np.asarray(p.fe.JxW)
        rowmass=np.einsum('qn,cq,cq->cn',shapes,weights,np.asarray(self.scale))
        self.mass_lumping='row_sum'
        if getattr(p,'element_degree',1)==2:
            # HRZ diagonal scaling: positive lumped mass, conserving each
            # cell's integrated material mass. Row sums can be negative in Q2.
            diag=np.einsum('qn,cq,cq->cn',shapes**2,weights,np.asarray(self.scale))
            total=np.sum(weights*np.asarray(self.scale),axis=1)
            cellmass=diag*(total/diag.sum(axis=1))[:,None]
            self.mass_lumping='HRZ_positive_diagonal_scaling'
        else:cellmass=rowmass
        self.negative_row_mass_entries=int(np.sum(rowmass<=0))
        self.mass_conservation_error=float(np.max(np.abs(
            cellmass.sum(axis=1)/np.sum(weights*np.asarray(self.scale),axis=1)-1)))
        if not np.isfinite(cellmass).all() or np.min(cellmass)<=0:
            raise ValueError('Nonpositive/nonfinite lumped material mass')
        cellmass*=density*L**2
        nodem=np.zeros(len(p.fe.points));np.add.at(nodem,np.asarray(p.fe.cells).ravel(),cellmass.ravel())
        self.nodem=jnp.asarray(nodem)
        self.mass=jnp.zeros(self.nc).at[self.ids].add(self.nodem)
        self.reduce=lambda r:jnp.zeros((self.nc,3)).at[self.ids].add(r)
        self.kernel_geometry=None
        if getattr(p,'element_degree',1)==2:
            # These regular Cartesian cells have the same reference map.
            # Verify it before retaining one kernel pattern, rather than
            # repeated GPU copies of several large geometric arrays.
            grad0=np.asarray(p.shape_grads[0]);weight0=np.asarray(p.JxW[0])
            gerror=0.;werror=0.;norm=np.max(np.abs(grad0))
            for start in range(0,p.fe.num_cells,512):
                gerror=max(gerror,float(np.max(np.abs(np.asarray(p.shape_grads[start:start+512])-grad0)))/norm)
                werror=max(werror,float(np.max(np.abs(np.asarray(p.JxW[start:start+512])/weight0-1))))
            if gerror>5e-12 or werror>5e-12:
                raise ValueError('Compact Q2 geometry requires identical Cartesian reference maps')
            self.common_geometry_relative_errors={'gradient':gerror,'weight':werror}
            self.kernel_geometry=(jnp.asarray(p.physical_quad_points),jnp.asarray(grad0),
                                  jnp.asarray(weight0),jnp.asarray(p.v_grads_JxW[0]),self.scale)
            del p.shape_grads,p.v_grads_JxW,p.fe.shape_grads,p.fe.v_grads_JxW
            compiled_force=jax.jit(self._force)
            self.force_for_geometry=compiled_force
            self.force=lambda q,h:compiled_force(q,h,self.kernel_geometry)
            self.stats=jax.jit(self._quadratic_stats)
            self.device_cells=jnp.asarray(p.fe.cells)
        else:self.force=jax.jit(self._force)
        self.cp=math.sqrt((KAPPA+4*MU/3)/(density*L**2))
        self.dt_estimate=.2/(round(self.nc**(1/3))*self.cp)
        # These COO indices are only used for tangent assembly, never here.
        del p.I,p.J

    def _force(self,q,h,geometry=None):
        H=jnp.zeros((3,3)).at[2,2].set(h)
        internal=[jnp.broadcast_to(H,(*self.scale.shape,3,3)),self.scale]
        if geometry is not None:
            # The installed JAX-FEM cell kernel and residual scatter are
            # unchanged. Only execute its batches through one compiled map.
            coords,grads,weights,vgrads,scale=geometry
            cells=q[self.ids][self.p.fe.cells].reshape((self.p.fe.num_cells,-1))
            inputs=[cells,coords,jnp.broadcast_to(H,(*scale.shape,3,3)),scale]
            batch=math.gcd(self.p.fe.num_cells,2048)
            packed=[x.reshape((-1,batch,*x.shape[1:])) for x in inputs]
            def kernel(x):
                return self.p.kernel(x[0],x[1],jnp.broadcast_to(grads,(batch,*grads.shape)),
                    jnp.broadcast_to(weights,(batch,*weights.shape)),
                    jnp.broadcast_to(vgrads,(batch,*vgrads.shape)),x[2],x[3])
            values=jax.lax.map(kernel,packed).reshape(cells.shape)
            return self.p.compute_residual_vars_helper(values,[])[0]
        return self.p.compute_residual_vars([q[self.ids]],internal,[])[0]

    def material_fields(self,scale):
        """Traceable Q2 HRZ mass using the original cell mass convention.

        scale must include the declared void floor. No mass scaling or extra
        mechanics is introduced; this interface is not gradient certification.
        """
        if self.kernel_geometry is None:
            raise ValueError('Design-dependent mass currently requires HEX27')
        weights=jnp.asarray(self.p.fe.JxW)
        shapes=jnp.asarray(self.p.fe.shape_vals)
        diag=jnp.einsum('qn,cq,cq->cn',shapes**2,weights,scale)
        total=jnp.sum(weights*scale,axis=1)
        cellmass=diag*(total/diag.sum(axis=1))[:,None]*self.mass_density_factor
        nodem=jnp.zeros(len(self.points)).at[self.device_cells.ravel()].add(cellmass.ravel())
        mass=jnp.zeros(self.nc).at[self.ids].add(nodem)
        return scale,nodem,mass

    def acceleration(self,q,h,hdd,geometry=None,nodem=None,mass=None):
        if geometry is None:geometry=self.kernel_geometry
        if nodem is None:nodem=self.nodem
        if mass is None:mass=self.mass
        r=self._force(q,h,geometry);gd=self.points*jnp.array([0.,0.,hdd])
        acc=-(self.reduce(r)+self.reduce(nodem[:,None]*gd))/mass[:,None]
        return acc.at[self.pin].set(0.)

    def block(self,dt,steps,load_time,wave=False):
        def motion(t):
            if wave:return jnp.array([0.,0.,0.])
            s=jnp.clip(t/load_time,0.,1.)
            h=-.2*(10*s**3-15*s**4+6*s**5)
            hd=-.2*(30*s**2-60*s**3+30*s**4)/load_time
            hdd=-.2*(60*s-180*s**2+120*s**3)/load_time**2
            return jnp.array([h,hd,hdd])
        def advance(state,count,geometry,nodem=None,mass=None):
            # A masked tail reuses the same compiled block. Recompiling large
            # constants for each short tail needlessly multiplies host memory.
            def active(carry):
                q,vhalf,t=carry;h,_,hdd=motion(t)
                vhalf=(vhalf+dt*self.acceleration(q,h,hdd,geometry,nodem,mass)).at[self.pin].set(0.)
                q=(q+dt*vhalf).at[self.pin].set(0.)
                return q,vhalf,t+dt
            def one(carry,k):
                return jax.lax.cond(k<count,active,lambda c:c,carry),None
            return jax.lax.scan(one,state,jnp.arange(steps))[0]
        compiled=jax.jit(advance)
        def apply(state,count=steps,material=None):
            if material is None:
                # Preserve the original default constant-mass compilation.
                return compiled(state,count,self.kernel_geometry)
            return compiled(state,count,*material)
        return apply,motion

    @staticmethod
    def _quadratic_stats(w,h,grads,cells,weights,scale):
        H=jnp.zeros((3,3)).at[2,2].set(h)
        F=jnp.eye(3)+H+jnp.einsum('cni,qnj->cqij',w[cells],grads)
        J=jnp.linalg.det(F)
        W=jax.vmap(neo_hookean_energy)(F.reshape((-1,3,3))).reshape(scale.shape)
        return jnp.min(J),jnp.sum(W*scale*weights)

    def observables(self,state,dt,motion,material=None):
        """JAX scalar response; caller handles finite/positive-volume protection."""
        geometry,nodem,mass=(self.kernel_geometry,self.nodem,self.mass) if material is None else material
        q,vhalf,t=state;h,hd,hdd=motion(t)
        H=jnp.zeros((3,3)).at[2,2].set(h);w=q[self.ids]
        if geometry is None:
            F=jnp.eye(3)+H+self.p.fe.sol_to_grad(w);minJ=jnp.linalg.det(F).min()
            W=jax.vmap(neo_hookean_energy)(F.reshape((-1,3,3))).reshape(self.scale.shape)
            U_norm=jnp.sum(W*self.scale*jnp.asarray(self.p.fe.JxW))
            r=self.force(q,h)
        else:
            minJ,U_norm=self.stats(w,h,geometry[1],self.device_cells,geometry[2][0],geometry[4])
            r=self.force_for_geometry(q,h,geometry)
        gd=self.points*jnp.array([0.,0.,hdd])
        acc=(-(self.reduce(r)+self.reduce(nodem[:,None]*gd))/mass[:,None]).at[self.pin].set(0.)
        velocity=(vhalf+.5*dt*acc)[self.ids]+self.points*jnp.array([0.,0.,hd])
        fullacc=acc[self.ids]+self.points*jnp.array([0.,0.,hdd])
        U=U_norm*self.L**3;KE=.5*jnp.sum(nodem[:,None]*velocity**2)*self.L**3
        qstatic=jnp.sum(r[:,2]*self.points[:,2])*self.L**2
        qdynamic=jnp.sum((r[:,2]+nodem*fullacc[:,2])*self.points[:,2])*self.L**2
        return {'time':t,'compression':-h,'Fz_N':qdynamic,'internal_macro_Fz_N':qstatic,
                'energy_N_mm':U,'KE_N_mm':KE,'KE_over_U':KE/jnp.maximum(U,1e-30),'J_min':minJ}

    def observe(self,state,dt,motion,material=None):
        values=self.observables(state,dt,motion,material)
        row={k:float(v) for k,v in values.items()}
        if not np.isfinite(np.asarray(state[0])).all() or not math.isfinite(row['J_min']) or row['J_min']<=0:
            raise ValueError('Nonfinite displacement or nonpositive reference Gauss detF')
        return row

def run(a):
    a.output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    if a.action=='wave':
        N=8;p=make_density_hyperelastic_problem(N,rho_quad=1.,eta=1e-4,periodic_axes=(0,1,2),element_degree=a.element_degree)
        cfg={'N':N,'role':'small known periodic P-wave integration check; not TPMS accuracy evidence'}
    else:
        source=json.loads((a.case/'step2/diagnostic_xyz/input.json').read_text())
        cfg={k:source[k] for k in ('case_id','N','cell_size_mm','thickness_mm',
                                  'E_MPa','nu','eta','interface_10_90_mm')}
        cfg.update(source_linear_case_input_sha256=sha(a.case/'step2/diagnostic_xyz/input.json'),
                   mechanical_periodic_axes=[0,1,2],target_compression=.2,
                   bc='XYZ periodic fluctuation; macro Hzz from zero to -0.2, lateral strain zero')
        N=a.cells or cfg['N']
        field_path=a.gauss_field or a.case/'gauss_field.npz'
        with np.load(field_path) as f:
            rho=f['rho'];qp=f['physical_quad_points'];qw=f['JxW']
            if a.thickness_mm is not None:
                # Only the declared physical thickness changes; keep distance,
                # interface width, material and eta. Used for path derivatives.
                from scipy.special import expit
                ell_normal=.005/(2*np.log(9))  # 0.05 mm / L=10 mm
                rho=expit((a.thickness_mm/20-f['distance'])/ell_normal)
                cfg['base_cached_thickness_mm']=cfg['thickness_mm']
                cfg['thickness_mm']=a.thickness_mm
                cfg['physical_thickness_override_mm']=a.thickness_mm
                cfg['occupancy_override']='same actual Gauss distance, same 0.05 mm interface; thickness perturbation only'
        cfg['evaluated_occupancy_sha256']=hashlib.sha256(np.ascontiguousarray(rho).tobytes()).hexdigest()
        def field(p):
            assert np.array_equal(np.asarray(p.physical_quad_points),qp)
            actual_weights=np.asarray(p.fe.JxW)
            weight_error=float(np.max(np.abs(actual_weights/qw-1)))
            if a.element_degree==1:assert np.array_equal(actual_weights,qw)
            else:assert weight_error<5e-12, 'Quadrature weights differ beyond roundoff'
            cfg['Gauss_weight_relative_max_difference']=weight_error
            return rho
        p=make_density_hyperelastic_problem(N,rho_quad=field,eta=cfg['eta'],periodic_axes=(0,1,2),element_degree=a.element_degree)
        cfg.update(N=N,Gauss_field_path=str(field_path),Gauss_field_sha256=sha(field_path))
        del rho,qp,qw
    ex=ExplicitXYZ(p);dt=ex.dt_estimate*(.25 if a.action=='wave' else 1.)
    cfg.update(method='physical central difference; existing JAX-FEM NH residual',dt_seconds=dt,
        element_type=p.fe.ele_type,element_degree=a.element_degree,nodes=len(p.fe.points),
        elements=p.fe.num_cells,Gauss_points_per_cell=p.fe.num_quads,
        mass_lumping=ex.mass_lumping,negative_unmodified_row_mass_entries=ex.negative_row_mass_entries,
        cell_mass_conservation_error=ex.mass_conservation_error,
        common_reference_geometry_errors=getattr(ex,'common_geometry_relative_errors',None),
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
        omega=2*ex.cp*N*math.sin(math.pi/N) if a.element_degree==1 else 2*math.pi*ex.cp
        cfg['wave_reference']='HEX8 discrete frequency' if a.element_degree==1 else 'continuum P-wave frequency; finite Q2 spatial approximation remains'
        end=2*math.pi/omega
    else:end=a.load_time*1.1
    steps=32 if a.action=='probe' else math.ceil(end/dt)
    if a.action!='probe':dt=end/steps
    else:end=steps*dt
    initial_dt=dt;rejections=[]
    cfg.update(dt_seconds=dt,total_steps=steps,adaptive_block_rejection=a.adaptive,
               minimum_dt_seconds=dt/16 if a.adaptive else dt)
    write(a.output/'input.json',cfg)
    # The target starts from the undeformed, exactly stress-free state.
    acc0=ex.acceleration(q,0.,0.) if wave else jnp.zeros_like(q)
    state=(q,-.5*dt*acc0,jnp.asarray(0.))
    chunk=8 if a.action=='probe' else min(128,max(1,steps//100))
    advance,motion=ex.block(dt,chunk,a.load_time,wave)
    t0=time.perf_counter();warm=advance(state,chunk);jax.block_until_ready(warm)
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
            local_dt=dt
            prior=state;trial=advance(prior,remain);jax.block_until_ready(trial)
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
    p.add_argument('--element-degree',type=int,choices=(1,2),default=1)
    p.add_argument('--cells',type=int,help='Cells per axis; periodic node levels also include midside nodes in Q2')
    p.add_argument('--gauss-field',type=Path,help='Actual Gauss cache for the selected element/quadrature')
    p.add_argument('--thickness-mm',type=float,help='Physical thickness perturbation at cached distances; interface/material unchanged')
    a=p.parse_args()
    if a.load_time<=0:p.error('--load-time must be positive')
    if a.cells is not None and a.cells<=0:p.error('--cells must be positive')
    if a.thickness_mm is not None and (a.thickness_mm<=0 or a.action=='wave'):
        p.error('--thickness-mm must be positive and is only for a target/probe')
    if a.element_degree==2 and a.action!='wave' and (a.cells is None or a.gauss_field is None):
        p.error('HEX27 target/probe requires --cells and its actual --gauss-field cache')
    run(a)

"""Bounded N64 resource preparation; reuse the sole explicit entry and material.

The adapter omits unused global tangent COO indices and shares identical
Cartesian reference maps. It does not change basis, quadrature, mass or forces.
No completed cost-probe state is a production compression trajectory.
"""
from pathlib import Path
import argparse, copy, hashlib, importlib.util, inspect, json, math, os, resource, time
os.environ.setdefault('JAX_PLATFORMS', 'cuda,cpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
R = Path('/home/xuehu/projects/tpms_jax')
D = R/'validation/n64_resolution_20261008_r28'
spec = importlib.util.spec_from_file_location('tpms_r28_entry', R/'scripts/thin_target_explicit.py')
entry = importlib.util.module_from_spec(spec); spec.loader.exec_module(entry)
import numpy as np
import jax
import jax.numpy as jnp
import hyperelastic_fem as hf
import jax_fem.problem as library_problem

def write(name, value):
    (D/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

def integrity():
    frozen = json.loads((D/'frozen_before.json').read_text())
    return {name: hashlib.sha256((R/name).read_bytes()).hexdigest()==digest
            for name,digest in frozen.items()}

def compact_geometry(self, fes_points=None):
    """Installed FE mapping on one verified regular Cartesian pattern."""
    assert len(self.fes)==1 and not self.boundary_inds_list
    fe = self.fes[0]
    assert fe.ele_type=='HEX27' and fe.num_quads==27
    first = copy.copy(fe); first.cells = fe.cells[:1]
    grads, weights = first.get_shape_grads()
    grads = np.asarray(grads); weights = np.asarray(weights)
    local0 = np.asarray(fe.points)[fe.cells[0]]
    local0 = local0-local0[0]
    qp = np.empty((fe.num_cells, fe.num_quads, 3), dtype=np.float64)
    error = 0.
    for start in range(0, fe.num_cells, 1024):
        cells = np.asarray(fe.cells[start:start+1024])
        coords = np.asarray(fe.points)[cells]
        error = max(error, float(np.max(np.abs(coords-coords[:,:1]-local0))))
        qp[start:start+len(cells)] = np.einsum('qn,cnd->cqd', np.asarray(fe.shape_vals), coords)
    if error>1e-14: raise ValueError('Cells do not share a Cartesian mapping')
    fe.shape_grads = np.broadcast_to(grads, (fe.num_cells,*grads.shape[1:]))
    fe.JxW = np.broadcast_to(weights, (fe.num_cells,fe.num_quads))
    fe.v_grads_JxW = np.broadcast_to(grads[:,:,:,None,:]*weights[:,:,None,None,None],
        (fe.num_cells,fe.num_quads,fe.num_nodes,1,3))
    self.shape_grads = fe.shape_grads; self.v_grads_JxW = fe.v_grads_JxW
    self.JxW = fe.JxW[:,None,:]; self.physical_quad_points = qp
    self.selected_face_shape_grads=[]; self.nanson_scale=[]
    self.selected_face_shape_vals=[]; self.physical_surface_quad_points=[]
    self.compact_mapping_coordinate_error = error

def compact_constructor():
    # Retain the installed initialization, excluding indices never used by
    # ExplicitXYZ (it deletes I/J without reading them after construction).
    import textwrap
    source = textwrap.dedent(inspect.getsource(library_problem.Problem.__post_init__))
    old = ('self.I = onp.repeat(inds[:, :, None], inds.shape[1], axis=2).reshape(-1)\n'
           '    self.J = onp.repeat(inds[:, None, :], inds.shape[1], axis=1).reshape(-1)')
    assert source.count(old)==1, 'Installed initialization changed'
    source = source.replace(old, 'self.I = onp.empty(0, dtype=onp.int32)\n    self.J = onp.empty(0, dtype=onp.int32)')
    (D/'effective_initialization.py').write_text(source)
    namespace={}; exec(compile(source,str(D/'effective_initialization.py'),'exec'),library_problem.__dict__,namespace)
    class CompactDensity(hf.DensityHyperelasticity):
        __post_init__ = namespace['__post_init__']
        initialize_geometric_quantities = compact_geometry
    return CompactDensity

def make_compact(*args, **kwargs):
    original_type = hf.DensityHyperelasticity
    try:
        hf.DensityHyperelasticity = compact_constructor()
        return original_make(*args,**kwargs)
    finally:
        hf.DensityHyperelasticity = original_type

original_make = entry.make_density_hyperelastic_problem

def equivalence():
    started=time.perf_counter()
    def phi(p):
        z=np.asarray(p.physical_quad_points)[...,2]
        return .5+.4*np.sin(2*np.pi*z)
    arguments=dict(rho_quad=phi,eta=1e-4,periodic_axes=(0,1,2),element_degree=2,
        quadrature_order=4,material_model='objective_void')
    with jax.default_device(jax.devices('cpu')[0]):
        full=original_make(2,**arguments); compact=make_compact(2,**arguments)
    errors={}
    def compare(name,a,b):
        a=np.asarray(a); b=np.asarray(b)
        assert a.shape==b.shape, name
        if a.dtype==bool or b.dtype==bool:
            errors[name]=0. if np.array_equal(a,b) else 1.
        else:
            floor=1e-30 if name in ('node_mass','periodic_mass') else 1.
            errors[name]=float(np.max(np.abs(a-b))/max(floor,float(np.max(np.abs(a)))))
        if errors[name]>1e-11: raise ValueError('Storage equivalence failed: '+name)
    for name in ('physical_quad_points','shape_grads','v_grads_JxW','JxW','class_ids','stiffness_scale'):
        compare(name,getattr(full,name),getattr(compact,name))
    a=entry.ExplicitXYZ(full,force_batch_cells=8); b=entry.ExplicitXYZ(compact,force_batch_cells=8)
    compare('node_mass',a.nodem,b.nodem); compare('periodic_mass',a.mass,b.mass)
    assert (full.P_mat!=compact.P_mat).nnz==0
    q=1e-4*jnp.sin(jnp.arange(a.nc*3,dtype=float).reshape(a.nc,3)*.3)
    q=q.at[a.pin].set(0.); dt=min(a.dt_estimate,b.dt_estimate)*.01
    state=(q,jnp.zeros_like(q),jnp.asarray(.0001))
    aa,motion=a.block(dt,2,.004); bb,_=b.block(dt,2,.004)
    compare('nonuniform_force',a.force(q,-.02),b.force(q,-.02))
    for name,x,y in zip(('q_after_two_steps','vhalf_after_two_steps','time_after_two_steps'),aa(state),bb(state)):
        compare(name,x,y)
    xa=a.observables(state,dt,motion); xb=b.observables(state,dt,motion)
    for name in xa: compare('observable_'+name,xa[name],xb[name])
    ba=entry.conservative_step_bound(a,8)(q,-.02); bc=entry.conservative_step_bound(b,8)(q,-.02)
    for name in ba: compare('bound_'+name,ba[name],bc[name])
    result={'passed':True,'relative_or_abs_errors':errors,'wall_seconds':time.perf_counter()-started,
        'scope':'Small nonuniform state compares storage, basis, PBC, HRZ, force, dynamic observables, two advances and tangent bound; not N64/20pct accuracy or design AD.'}
    write('storage_equivalence.json',result); print('STORAGE_EQUIVALENCE '+json.dumps(result),flush=True)

def args(out):
    old=R/'validation/geometry_transfer_20261006_r15'
    return argparse.Namespace(action='target',case=old,case_input=old/'input.json',output=out,
        load_time=.004,material_model='objective_void',adaptive=False,diagnose_first_failure=False,
        state_step_control=True,control_budget_seconds=3600.,stability_states=None,
        stability_batch_cells=128,replay_state=None,replay_dt_factor=.125,
        checkpoint_compressions=[.01,.05,.10,.12,.14,.16,.18,.20],element_degree=2,cells=64,
        geometry_on_cpu=True,force_batch_cells=2048,quadrature_order=4,
        gauss_field=None,surface_geometry=old/'surface_geometry.npz',thickness_mm=None)

def resource_probe(ex,a,cfg,N):
    """Compile/warm then two initial 128-step blocks, discard all states."""
    start=time.perf_counter()
    q=jnp.zeros((ex.nc,3));state=(q,jnp.zeros_like(q),jnp.asarray(0.))
    bound=entry.conservative_step_bound(ex,a.stability_batch_cells)
    t=time.perf_counter(); raw=bound(q,0.); jax.block_until_ready(raw); compile_bound=time.perf_counter()-t
    if not bool(raw['all_material_tangents_finite']) or int(raw['invalid_material_points']):
        raise ValueError('Invalid initial N64 tangent')
    Rbound=float(raw['row_sum_bound_s_minus2']);dt=min(ex.dt_estimate,1.6/math.sqrt(Rbound))
    advance,motion=ex.block(ex.dt_estimate,128,a.load_time)
    t=time.perf_counter(); warm=advance(state,128,step_dt=jnp.asarray(dt));jax.block_until_ready(warm)
    compile_advance=time.perf_counter()-t
    t=time.perf_counter(); row=ex.observe(warm,dt,motion);compile_observe=time.perf_counter()-t
    costs=[]
    for _ in range(2):
        t=time.perf_counter(); trial=advance(state,128,step_dt=jnp.asarray(dt));jax.block_until_ready(trial)
        advance_seconds=time.perf_counter()-t
        t=time.perf_counter(); row=ex.observe(trial,dt,motion);observe_seconds=time.perf_counter()-t
        t=time.perf_counter(); check=bound(trial[0],-row['compression']);jax.block_until_ready(check)
        bound_seconds=time.perf_counter()-t
        assert bool(check['all_material_tangents_finite']) and not int(check['invalid_material_points'])
        costs.append(dict(advance_seconds=advance_seconds,observe_seconds=observe_seconds,bound_seconds=bound_seconds))
    steps=math.ceil(.0044/dt)
    avg=sum(sum(x.values()) for x in costs)/len(costs)
    input_field=np.load(a.output/'gauss_field.npz')
    vf=float(np.sum(input_field['rho']*input_field['JxW'])/np.sum(input_field['JxW']))
    total_tonne=float(jnp.sum(ex.nodem))*ex.L
    result=dict(status='N64_initial_resource_probe_completed',N=N,nodes=len(ex.p.fe.points),
        elements=ex.p.fe.num_cells,Gauss_points_per_cell=ex.p.fe.num_quads,
        integrated_occupancy=vf,total_material_plus_floor_mass_tonne=total_tonne,
        nominal_dt_seconds=ex.dt_estimate,initial_safe_dt_seconds=dt,
        initial_bound_s_minus2=Rbound,compile_bound_seconds=compile_bound,
        compile_advance_seconds=compile_advance,compile_observe_seconds=compile_observe,
        block_costs=costs,constant_initial_dt_estimated_steps=steps,
        optimistic_body_estimate_seconds=math.ceil(steps/128)*avg,
        probe_seconds=time.perf_counter()-start,peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        probe_endpoint=row,from_zero_full20_started=False,
        scope='Initial-state cost only; two discarded duplicate short blocks, no full path and no peak/20pct or AD claim. Later dt reduction and 16-step monitoring can substantially increase cost.')
    cfg.update(resource_probe_only=True,dynamic_jump_allowed=True,diagnostic_support_energy_factor=1.,
        initial_safe_dt_seconds=dt,integrated_occupancy=vf,total_mass_tonne=total_tonne)
    entry.write(a.output/'input.json',cfg);write('resource_result.json',result)
    print('N64_RESOURCE_RESULT '+json.dumps(result),flush=True)

def full_controller():
    """Retain production control, record high KE without its old quality stop."""
    source=inspect.getsource(entry.state_step_target)
    changes=[('if high_KE_time>.05*a.load_time:', 'if False and high_KE_time>.05*a.load_time:'),
        ('irrecoverable_high_KE_time_limit=.05*a.load_time,',
         'irrecoverable_high_KE_time_limit=None,high_KE_duration_is_recorded_not_stop=True,')]
    for old,new in changes:
        if source.count(old)!=1:raise RuntimeError('Production controller changed')
        source=source.replace(old,new)
    (D/'effective_full_controller.py').write_text(source)
    namespace={};exec(compile(source,str(D/'effective_full_controller.py'),'exec'),entry.__dict__,namespace)
    return namespace['state_step_target']

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('equivalence','resource','full'))
    parser.add_argument('--budget-seconds',type=float)
    options=parser.parse_args();stage=options.stage
    assert all(integrity().values()), 'Frozen evidence/source changed'
    start=time.perf_counter();code=0; receipt={'stage':stage,'started_unix':time.time()}
    try:
        if stage=='equivalence':equivalence()
        else:
            assert json.loads((D/'storage_equivalence.json').read_text())['passed']
            entry.make_density_hyperelastic_problem=make_compact
            if stage=='resource':
                entry.state_step_target=resource_probe
                entry.run(args(D/'resource_probe'))
            else:
                import fcntl
                decision=json.loads((D/'launch_decision.json').read_text())
                if (not options.budget_seconds or not math.isfinite(options.budget_seconds)
                    or options.budget_seconds!=decision['body_budget_seconds']
                    or not decision['full_attempt_authorized']):
                    raise ValueError('Explicit selected budget and launch decision required')
                assert json.loads((D/'resource_result.json').read_text())['status']=='N64_initial_resource_probe_completed'
                lock=Path('/tmp/tpms_jax_large_compression.lock').open('a')
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                controlled=full_controller()
                def deadline_controller(ex,a,cfg,N):
                    # The user's total wall budget includes reference job and
                    # geometry setup; reserve time for evidence extraction.
                    remaining=decision['deadline_unix']-time.time()-120.
                    if remaining<=0:raise TimeoutError('Total wall budget expired before forward start')
                    a.control_budget_seconds=min(options.budget_seconds,remaining)
                    cfg.update(user_total_budget_seconds=10800.,wall_deadline_unix=decision['deadline_unix'],
                        actual_body_budget_seconds=a.control_budget_seconds,dynamic_jump_allowed=True,
                        diagnostic_support_energy_factor=1.,load_time_seconds=decision['load_time_seconds'],
                        hold_time_seconds=.1*decision['load_time_seconds'],no_design_AD=True)
                    return controlled(ex,a,cfg,N)
                entry.state_step_target=deadline_controller
                a=args(D/'full_from_zero');a.control_budget_seconds=options.budget_seconds
                a.load_time=decision['load_time_seconds']
                receipt.update(body_budget_seconds=options.budget_seconds,
                    dynamic_jump_allowed=True,virtual_support_multiplier=1.,from_zero=True,
                    protocol_sha256=hashlib.sha256((D/'protocol.json').read_bytes()).hexdigest(),
                    adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),pid=os.getpid())
                write('full_launch.json',receipt)
                entry.run(a)
    except Exception as exc:
        receipt.update(exception_type=type(exc).__name__,exception_message=str(exc));code=1
        raise
    finally:
        receipt.update(wall_seconds=time.perf_counter()-start,returncode=code,frozen_unchanged=integrity())
        write(stage+'_receipt.json',receipt)
    return code

if __name__=='__main__':raise SystemExit(main())

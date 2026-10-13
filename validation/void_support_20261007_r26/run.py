"""One-factor adapter; imports the sole production explicit entry unchanged."""
from pathlib import Path
import argparse, hashlib, importlib.util, inspect, json, math, os, time
os.environ.setdefault('JAX_PLATFORMS','cuda,cpu')
R=Path('/home/xuehu/projects/tpms_jax'); D=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('tpms_explicit_r26',R/'scripts/thin_target_explicit.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
jax=entry.jax;jnp=entry.jnp
from hyperelastic_fem import void_nh_weight

def apply_support(p,factor):
    """Inject before first JIT; force/stats/bound all read this shared energy."""
    if p.material_model!='objective_void':raise ValueError('Diagnostic requires objective_void')
    original_energy=p.material_energy
    def energy(F,scale):
        rho=(scale-p.eta)/(1-p.eta)
        return (factor+(1-factor)*void_nh_weight(rho))*original_energy(F,scale)
    p.material_energy=energy
    p.material_stress=jax.grad(energy,argnums=0)
    p.diagnostic_support_factor=factor
    return p

def adapted_controller():
    source=inspect.getsource(entry.state_step_target)
    changes=[
      ('end=1.1*a.load_time;nominal=end/math.ceil(end/ex.dt_estimate);floor=nominal/16',
       'original_end=1.1*a.load_time;end=a.diagnostic_end_time;nominal=original_end/math.ceil(original_end/ex.dt_estimate);floor=nominal/16'),
      ('if high_KE_time>.05*a.load_time:',
       'if False and high_KE_time>.05*a.load_time:')]
    for old,new in changes:
        if source.count(old)!=1:raise RuntimeError('Production controller changed; review adapter first')
        source=source.replace(old,new)
    return source

def arguments(out):
    lo,hi=0.,1.
    for _ in range(80):
        mid=(lo+hi)/2;value=.2*(10*mid**3-15*mid**4+6*mid**5)
        if value<.17:lo=mid
        else:hi=mid
    old=R/'validation/geometry_transfer_20261006_r15'
    return argparse.Namespace(action='target',case=old,case_input=old/'input.json',
      output=out,load_time=.004,material_model='objective_void',adaptive=False,
      diagnose_first_failure=False,state_step_control=True,control_budget_seconds=1500.,
      stability_states=None,stability_batch_cells=128,replay_state=None,replay_dt_factor=.125,
      checkpoint_compressions=[.10,.12,.14,.16,.17],element_degree=2,cells=32,
      geometry_on_cpu=True,force_batch_cells=2048,quadrature_order=4,
      gauss_field=old/'gauss_field.npz',surface_geometry=None,thickness_mm=None,
      diagnostic_end_time=.004*(lo+hi)/2)

def main():
    before=json.loads((D/'freeze_before.json').read_text())
    for p,h in before.items():
        if hashlib.sha256((R/p).read_bytes()).hexdigest()!=h:raise RuntimeError('Frozen input changed: '+p)
    source=adapted_controller()
    (D/'effective_controller.txt').write_text(source)
    namespace={};exec(compile(source,str(D/'effective_controller.txt'),'exec'),entry.__dict__,namespace)
    entry.state_step_target=namespace['state_step_target']
    original_make=entry.make_density_hyperelastic_problem
    def make(*args,**kwargs):return apply_support(original_make(*args,**kwargs),.1)
    entry.make_density_hyperelastic_problem=make
    out=D/'forward';a=arguments(out)
    receipt={'scope':'single from-zero support-energy intervention, not default/20pct/design AD',
      'factor':.1,'stop_compression':.17,'stop_time_seconds':a.diagnostic_end_time,
      'production_unchanged':True,'dynamic_jump_allowed':True,'started_unix':time.time(),
      'protocol_sha256':hashlib.sha256((D/'PROTOCOL.md').read_bytes()).hexdigest(),
      'adapter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (D/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
    started=time.perf_counter();code=0
    try:entry.run(a)
    except Exception as exc:
        receipt.update(exception_type=type(exc).__name__,exception_message=str(exc));code=1
    finally:
        receipt.update(total_seconds=time.perf_counter()-started,exit_code=code,
          frozen_files_unchanged=all(hashlib.sha256((R/p).read_bytes()).hexdigest()==h for p,h in before.items()))
        if (out/'input.json').exists():
            cfg=json.loads((out/'input.json').read_text())
            cfg.update(diagnostic_support_energy_factor=.1,diagnostic_support_gate=[.001,.01],
              diagnostic_mass_unchanged=True,diagnostic_stop_compression=.17,
              diagnostic_stop_time_seconds=a.diagnostic_end_time,diagnostic_dynamic_jump_allowed=True,
              irrecoverable_high_KE_time_limit=None,high_KE_duration_is_recorded_not_stop=True,
              adaptive_block_rejection=True,scope=receipt['scope'])
            (out/'input.json').write_text(json.dumps(cfg,indent=2)+'\n')
        (D/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('R26_RECEIPT '+json.dumps(receipt),flush=True)
    return code
if __name__=='__main__':raise SystemExit(main())

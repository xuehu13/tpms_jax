"""One frozen binary27 forward diagnostic; production entry/material unchanged."""
from pathlib import Path
import argparse,fcntl,hashlib,importlib.util,inspect,json,math,os,time
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
R=Path('/home/xuehu/projects/tpms_jax');D=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('tpms_r44_entry',R/'scripts/thin_target_explicit.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
def write(n,v):(D/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def unchanged():return {n:sha(R/n)==h for n,h in json.loads((D/'frozen_before.json').read_text()).items()}
def controller():
 source=inspect.getsource(e.state_step_target)
 changes=[('floor=nominal/16','floor=nominal/4096'),
 ('if high_KE_time>.05*a.load_time:','if False and high_KE_time>.05*a.load_time:'),
 ('irrecoverable_high_KE_time_limit=.05*a.load_time,','irrecoverable_high_KE_time_limit=None,high_KE_duration_is_recorded_not_stop=True,'),
 ('unchanged initial/16 safeguard','predeclared binary diagnostic initial/4096 safeguard')]
 for old,new in changes:
  if source.count(old)!=1 and old!='unchanged initial/16 safeguard':raise RuntimeError('Entry source changed: '+old)
  if old=='unchanged initial/16 safeguard' and source.count(old)!=2:raise RuntimeError('Entry safeguard text changed')
  source=source.replace(old,new)
 (D/'effective_controller.py').write_text(source)
 ns={};exec(compile(source,str(D/'effective_controller.py'),'exec'),e.__dict__,ns)
 return ns['state_step_target']
def main():
 p=json.loads((D/'protocol.json').read_text());assert all(unchanged().values()),'Old evidence changed'
 lock=Path('/tmp/tpms_jax_large_compression.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 start=time.perf_counter();deadline=time.time()+p['total_wall_budget_seconds'];record={'started_unix':time.time(),'pid':os.getpid(),'protocol_sha256':sha(D/'protocol.json'),'adapter_sha256':sha(Path(__file__)),'one_from_zero_attempt':True,'returncode':0}
 controlled=controller()
 def apply(ex,a,cfg,N):
  remaining=deadline-time.time()
  if remaining<=0:raise TimeoutError('Total wall budget expired during model construction')
  a.control_budget_seconds=remaining
  cfg.update(binary_occupancy=True,interface_10_90_mm=0.,original_smooth_interface_width_mm=.05,dynamic_jump_allowed=True,
    user_total_wall_budget_seconds=p['total_wall_budget_seconds'],wall_deadline_unix=deadline,
    occupancy_rule=p['occupancy_rule'],controller_adapter_sha256=sha(D/'effective_controller.py'),diagnostic_adapter_sha256=sha(Path(__file__)),
    integrated_binary_occupancy=p['binary_volume_fraction'],integrated_binary_mass_tonne=p['binary_mass_tonne'],no_design_AD=True)
  return controlled(ex,a,cfg,N)
 e.state_step_target=apply
 a=argparse.Namespace(action='target',case=R/'validation/geometry_transfer_20261006_r15',case_input=D/'input.json',output=D/'full_from_zero',
  load_time=p['load_time_seconds'],material_model='objective_void',adaptive=False,diagnose_first_failure=False,state_step_control=True,
  control_budget_seconds=p['total_wall_budget_seconds'],stability_states=None,stability_batch_cells=128,replay_state=None,replay_dt_factor=.125,
  checkpoint_compressions=p['checkpoints'],element_degree=2,cells=32,geometry_on_cpu=True,force_batch_cells=2048,quadrature_order=4,
  gauss_field=D/'binary_gauss_field.npz',surface_geometry=None,thickness_mm=None)
 write('launch.json',record);print('R44_LAUNCH '+json.dumps(record),flush=True)
 try:e.run(a)
 except BaseException as exc:
  record.update(returncode=1,exception_type=type(exc).__name__,exception_message=str(exc));raise
 finally:
  record.update(total_wall_seconds=time.perf_counter()-start,frozen_unchanged=unchanged());write('receipt.json',record)
if __name__=='__main__':main()

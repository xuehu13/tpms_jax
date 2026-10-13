"""One exact checkpoint continuation if the original run ends ONLY at wall budget.

Uses the same explicit entry, material, occupancy, HRZ and time-step controller.
Preparation grants no extra computation budget; a selected budget is required.
"""
from pathlib import Path
import argparse,fcntl,hashlib,importlib.util,json,math,os,time
R=Path('/home/xuehu/projects/tpms_jax');D=Path(__file__).resolve().parent
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
spec=importlib.util.spec_from_file_location('r44_parent_adapter',D/'run.py');parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
e=parent.e;np=e.np;jnp=e.jnp;jax=e.jax
C=D/'continuation_from_budget'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def resumed_controller():
 source=(D/'effective_controller.py').read_text()
 replacements=[
 ('state=(jnp.zeros((ex.nc,3)),jnp.zeros((ex.nc,3)),jnp.asarray(0.))','state=R44_RESUME_STATE'),
 ('state_dt=nominal;dt=nominal;path=[];rejections=[];bounds=[];saved=set()',
  'state_dt=R44_RESUME_DT;dt=R44_RESUME_DT;path=list(R44_PRIOR_PATH);rejections=list(R44_PRIOR_REJECTIONS);bounds=list(R44_PRIOR_BOUNDS);saved=set(R44_PRIOR_SAVED)'),
 ('high_KE_time=0.;done=0;bound_cost=0.;observing_cost=0.;started=time.perf_counter()',
  'high_KE_time=R44_PRIOR_HIGH_KE_TIME;done=R44_PRIOR_STEPS;bound_cost=0.;observing_cost=0.;started=time.perf_counter()'),
 ('dt=min(nominal,current[\'safe_dt_seconds\'])','dt=min(state_dt,current[\'safe_dt_seconds\'])'),
 ('path=[observe(state,state_dt)]','R44_CHECK_CONTINUITY(observe(state,state_dt));path=list(R44_PRIOR_PATH)'),
 ('if time.perf_counter()-started>a.control_budget_seconds:','if time.time()>R44_TOTAL_WALL_DEADLINE:')]
 for old,new in replacements:
  if source.count(old)!=1:raise RuntimeError('Frozen controller initialization changed: '+old)
  source=source.replace(old,new)
 (D/'effective_continuation_controller.py').write_text(source)
 ns={};exec(compile(source,str(D/'effective_continuation_controller.py'),'exec'),e.__dict__,ns)
 return ns['state_step_target']
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--total-budget-seconds',type=float,required=True);opts=parser.parse_args()
 if not math.isfinite(opts.total_budget_seconds) or not 10800<opts.total_budget_seconds<=21600:raise ValueError('Select a finite extended TOTAL budget in (3,6] hours before launch')
 if not (D/'selected_continuation_budget.json').exists():raise ValueError('Budget choice not recorded; preparation is not authorization')
 choice=json.loads((D/'selected_continuation_budget.json').read_text())
 if choice['total_budget_seconds']!=opts.total_budget_seconds:raise ValueError('Budget selection mismatch')
 assert all(parent.unchanged().values()),'Old evidence changed'
 first=D/'full_from_zero';failure=json.loads((first/'failure.json').read_text());receipt=json.loads((D/'receipt.json').read_text())
 if failure['type']!='TimeoutError' or 'budget' not in failure['message'].lower():raise ValueError('Continue only a wall-budget stop; never a numerical failure')
 if (first/'result.json').exists() or C.exists():raise ValueError('Only one continuation of the saved budget endpoint, no replay/retry')
 assert all(receipt['frozen_unchanged'].values())
 deadline=json.loads((D/'launch.json').read_text())['started_unix']+opts.total_budget_seconds
 if deadline-time.time()<=0:raise TimeoutError('Extended TOTAL wall deadline already expired')
 lock=Path('/tmp/tpms_jax_large_compression.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 prior=json.loads((first/'accepted_path.json').read_text());saved=first/'last_valid_field.npz'
 with np.load(saved) as f:
  assert int(f['N'])==32
  state=(jnp.asarray(f['q']),jnp.asarray(f['vhalf']),jnp.asarray(f['time']));old_dt=float(f['dt'])
 assert float(state[2])==prior[-1]['time'] and old_dt==prior[-1]['dt_seconds']
 retained={str(p.relative_to(D)):sha(p) for p in first.rglob('*') if p.is_file()}
 retained.update({n:sha(D/n) for n in ['receipt.json','launch.json','protocol.json','run.py','effective_controller.py','binary_gauss_field.npz','input.json']})
 write(D/'first_phase_frozen_before_continuation.json',retained)
 continuity={}
 def check(row):
  for k,v in row.items():
   expected=prior[-1][k]
   if isinstance(v,bool):err=0. if v==expected else 1.
   elif expected is None:err=0. if v is None else 1.
   else:err=abs(float(v)-float(expected))/max(abs(float(expected)),1e-30)
   continuity[k]=err
   if err>1e-11:raise ValueError('Exact restart observable mismatch: '+k)
  write(D/'continuity_check.json',dict(passed=True,relative_errors=continuity,retained_q_vhalf_time_dt=True,source_state_sha256=sha(saved),no_reset_or_short_replay=True))
 for n,value in dict(R44_RESUME_STATE=state,R44_RESUME_DT=old_dt,R44_PRIOR_PATH=prior,
  R44_PRIOR_REJECTIONS=json.loads((first/'rejected_blocks.json').read_text()),R44_PRIOR_BOUNDS=json.loads((first/'stability_path.json').read_text()),
  R44_PRIOR_SAVED=[float(p.stem[10:]) for p in first.glob('accepted_a*.npz')],R44_PRIOR_HIGH_KE_TIME=failure['sampled_high_KE_loading_seconds'],R44_PRIOR_STEPS=failure['completed_steps'],R44_CHECK_CONTINUITY=check,R44_TOTAL_WALL_DEADLINE=deadline).items():e.__dict__[n]=value
 controlled=resumed_controller();start=time.perf_counter();record={'started_unix':time.time(),'pid':os.getpid(),'source_state':str(saved),'source_state_sha256':sha(saved),'selected_total_budget_seconds':opts.total_budget_seconds,'returncode':0,'scope':'Exact continuation of the one from-zero trajectory; physical and numerical controller parameters unchanged.'}
 write(D/'continuation_launch.json',record)
 def apply(ex,a,cfg,N):
  if state[0].shape!=(ex.nc,3) or state[1].shape!=state[0].shape:raise ValueError('Saved periodic state dimensions differ')
  if not np.array_equal(np.asarray(state[0][ex.pin]),np.zeros(3)) or not np.array_equal(np.asarray(state[1][ex.pin]),np.zeros(3)):raise ValueError('Saved pin changed')
  a.control_budget_seconds=deadline-time.time()
  if a.control_budget_seconds<=0:raise TimeoutError('Total extended wall budget expired before exact continuation')
  cfg.update(binary_occupancy=True,interface_10_90_mm=0.,original_smooth_interface_width_mm=.05,dynamic_jump_allowed=True,
   integrated_binary_occupancy=json.loads((D/'protocol.json').read_text())['binary_volume_fraction'],
   exact_checkpoint_continuation=True,parent_from_zero_directory=str(first),source_state_sha256=sha(saved),
   selected_total_budget_seconds=opts.total_budget_seconds,wall_deadline_unix=deadline,remaining_body_budget_seconds=a.control_budget_seconds,
   initial_parent_nominal_dt_seconds=json.loads((first/'input.json').read_text())['initial_dt_seconds'],
   continuation_controller_sha256=sha(D/'effective_continuation_controller.py'),no_design_AD=True)
  return controlled(ex,a,cfg,N)
 e.state_step_target=apply
 p=json.loads((D/'protocol.json').read_text())
 a=argparse.Namespace(action='target',case=R/'validation/geometry_transfer_20261006_r15',case_input=D/'input.json',output=C,load_time=.004,material_model='objective_void',adaptive=False,diagnose_first_failure=False,state_step_control=True,control_budget_seconds=opts.total_budget_seconds,stability_states=None,stability_batch_cells=128,replay_state=None,replay_dt_factor=.125,checkpoint_compressions=p['checkpoints'],element_degree=2,cells=32,geometry_on_cpu=True,force_batch_cells=2048,quadrature_order=4,gauss_field=D/'binary_gauss_field.npz',surface_geometry=None,thickness_mm=None)
 try:e.run(a)
 except BaseException as exc:record.update(returncode=1,exception_type=type(exc).__name__,exception_message=str(exc));raise
 finally:
  record.update(phase_wall_seconds=time.perf_counter()-start,total_elapsed_since_first_launch_seconds=time.time()-json.loads((D/'launch.json').read_text())['started_unix'],first_phase_unchanged={n:sha(D/n)==h for n,h in retained.items()},old_frozen_unchanged=parent.unchanged());write(D/'continuation_receipt.json',record)
if __name__=='__main__':main()

"""Bounded matched-rate binary diagnostic; reuse production and r28 storage only."""
from pathlib import Path
import argparse,fcntl,hashlib,importlib.util,inspect,json,math,os,time
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
R=Path('/home/xuehu/projects/tpms_jax');D=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('r45_compact_reference',R/'validation/n64_resolution_20261008_r28/run.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base);base.D=D
e=base.entry;np=e.np;jnp=e.jnp;jax=e.jax
P=json.loads((D/'protocol.json').read_text());DEADLINE=P['started_unix']+P['total_budget_seconds']-P['postprocess_reserve_seconds']
def write(name,v):(D/name).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def intact():return {n:sha(R/n)==h for n,h in json.loads((D/'frozen_before.json').read_text()).items()}
def args(N,stage,T):
 return argparse.Namespace(action='target',case=R/'validation/geometry_transfer_20261006_r15',case_input=D/f'input_N{N}.json',output=D/(P.get('output_directory',f'{stage}_N{N}') if stage=='full' else f'{stage}_N{N}'),load_time=T,material_model='objective_void',adaptive=False,diagnose_first_failure=False,state_step_control=True,control_budget_seconds=max(0.,DEADLINE-time.time()),stability_states=None,stability_batch_cells=128,replay_state=None,replay_dt_factor=.125,checkpoint_compressions=[.01,.05,.10,.12,.14,.16,.18,.20],element_degree=2,cells=N,geometry_on_cpu=True,force_batch_cells=2048,quadrature_order=4,gauss_field=(D/'binary_N64_gauss.npz' if N==64 else R/'validation/binary_full20_20261008_r44/binary_gauss_field.npz'),surface_geometry=None,thickness_mm=None)
def probe(ex,a,cfg,N):
 q=jnp.zeros((ex.nc,3));state=(q,jnp.zeros_like(q),jnp.asarray(0.));b=e.conservative_step_bound(ex,a.stability_batch_cells)
 t=time.perf_counter();raw=b(q,0.);jax.block_until_ready(raw);compile_b=time.perf_counter()-t
 assert bool(raw['all_material_tangents_finite']) and not int(raw['invalid_material_points'])
 dt=min(ex.dt_estimate,1.6/math.sqrt(float(raw['row_sum_bound_s_minus2'])))
 advance,motion=ex.block(ex.dt_estimate,128,a.load_time,False)
 t=time.perf_counter();warm=advance(state,128,step_dt=jnp.asarray(dt));jax.block_until_ready(warm);compile_a=time.perf_counter()-t
 t=time.perf_counter();ex.observe(warm,dt,motion);compile_o=time.perf_counter()-t
 t=time.perf_counter();trial=advance(state,128,step_dt=jnp.asarray(dt));jax.block_until_ready(trial);advance_cost=time.perf_counter()-t
 t=time.perf_counter();row=ex.observe(trial,dt,motion);observe_cost=time.perf_counter()-t
 t=time.perf_counter();check=b(trial[0],-row['compression']);jax.block_until_ready(check);bound_cost=time.perf_counter()-t
 assert bool(check['all_material_tangents_finite']) and not int(check['invalid_material_points'])
 cost=advance_cost+observe_cost+bound_cost
 result={'N':N,'dt_initial_seconds':dt,'advance_128_seconds':advance_cost,'observe_seconds':observe_cost,'bound_seconds':bound_cost,'compile_bound_seconds':compile_b,'compile_advance_seconds':compile_a,'compile_observe_seconds':compile_o,'estimated_optimistic_body_seconds_by_T':{str(T):math.ceil(math.ceil(1.1*T/dt)/128)*cost for T in P['T_candidates_seconds']},'observed_discarded_probe_endpoint':row,'full_from_zero_started':False,'scope':'Duplicate zero-state warm/probe discarded; optimistic initial dt estimate only. Later dt reduction and 16-step bound checks can multiply cost.'}
 cfg.update(binary_occupancy=True,resource_probe_only=True,no_design_AD=True);e.write(a.output/'input.json',cfg);write('resource_result.json',result);print('RESOURCE '+json.dumps(result),flush=True)
def controlled():
 s=inspect.getsource(e.state_step_target)
 changes=[('floor=nominal/16','floor=nominal/4096'),('if high_KE_time>.05*a.load_time:','if False and high_KE_time>.05*a.load_time:'),('irrecoverable_high_KE_time_limit=.05*a.load_time,','irrecoverable_high_KE_time_limit=None,high_KE_duration_is_recorded_not_stop=True,'),('if time.perf_counter()-started>a.control_budget_seconds:','if time.time()>R45_CURRENT_DEADLINE:'),('unchanged initial/16 safeguard','predeclared r45 initial/4096 safeguard')]
 for old,new in changes:
  assert s.count(old)==(2 if old=='unchanged initial/16 safeguard' else 1),old;s=s.replace(old,new)
 (D/'effective_controller.py').write_text(s);ns={};exec(compile(s,str(D/'effective_controller.py'),'exec'),e.__dict__,ns);return ns['state_step_target']
def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['probe','full']);ap.add_argument('--N',type=int,choices=[32,64],default=64);opts=ap.parse_args();assert all(intact().values())
 assert json.loads((R/'validation/n64_resolution_20261008_r28/storage_equivalence.json').read_text())['passed']
 lock=Path('/tmp/tpms_jax_large_compression.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if opts.N==64:e.make_density_hyperelastic_problem=base.make_compact
 if opts.stage=='probe':e.state_step_target=probe;T=.001
 else:
  assert P['load_time_seconds'] is not None and P['matched_peak_N'] is not None
  T=P['load_time_seconds'];ctl=controlled();e.__dict__['R45_CURRENT_DEADLINE']=min(DEADLINE,P.get('N'+str(opts.N)+'_deadline_unix',DEADLINE))
  def apply(ex,a,cfg,N):
   remaining=e.__dict__['R45_CURRENT_DEADLINE']-time.time()
   if remaining<=0:raise TimeoutError('Shared 2h budget expired before from-zero solver')
   a.control_budget_seconds=remaining;cfg.update(binary_occupancy=True,no_design_AD=True,dynamic_jump_allowed=True,shared_total_budget_seconds=7200.,load_time_seconds=T,hold_time_seconds=.1*T,wall_deadline_unix=e.__dict__['R45_CURRENT_DEADLINE']);return ctl(ex,a,cfg,N)
  e.state_step_target=apply
 record={'stage':opts.stage,'N':opts.N,'pid':os.getpid(),'started_unix':time.time(),'load_time_seconds':T,'returncode':0};write((P.get('attempt_tag',opts.stage) if opts.stage=='full' else opts.stage)+f'_N{opts.N}_launch.json',record);start=time.perf_counter()
 try:e.run(args(opts.N,opts.stage,T))
 except BaseException as x:record.update(returncode=1,exception_type=type(x).__name__,exception_message=str(x));raise
 finally:record.update(wall_seconds=time.perf_counter()-start,total_elapsed_seconds=time.time()-P['started_unix'],frozen_unchanged=intact());write((P.get('attempt_tag',opts.stage) if opts.stage=='full' else opts.stage)+f'_N{opts.N}_receipt.json',record)
if __name__=='__main__':main()

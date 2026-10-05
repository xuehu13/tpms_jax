"""Single-thickness path JVP on the existing explicit kernel; no second FEM."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,resource,shutil,sys,time
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
R=Path('/home/xuehu/projects/tpms_jax');sys.path[:0]=[str(R),str(R/'scripts')]
import numpy as np
import jax
import jax.numpy as jnp
from hyperelastic_fem import make_density_hyperelastic_problem
from thin_target_explicit import ExplicitXYZ
from surface_distance import thickness_occupancy
p=argparse.ArgumentParser();p.add_argument('action',choices=['probe','jvp','minus','plus']);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(exist_ok=False);started=time.perf_counter()
root=R/'validation/large_compression_20261005_r6';base=root/'quadratic_candidate';experiment=root/'gradient20_path_20261005'
def write(name,obj):(a.output/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
with np.load(base/'gauss_field.npz') as cache:rho=cache['rho'];distance=jnp.asarray(cache['distance']);points=cache['physical_quad_points']
def field(p):
 assert np.array_equal(np.asarray(p.physical_quad_points),points)
 return rho
problem=make_density_hyperelastic_problem(32,rho_quad=field,eta=1e-4,periodic_axes=(0,1,2),element_degree=2)
ex=ExplicitXYZ(problem);del rho,points
@jax.jit
def fields(theta):return ex.material_fields(1e-4+(1-1e-4)*thickness_occupancy(distance,theta/10,.005/(2*np.log(9))))
def material(f):return ((*ex.kernel_geometry[:4],f[0]),f[1],f[2])
central,tangent_fields=jax.jvp(fields,(jnp.array(.5),),(jnp.array(1.),))
jax.block_until_ready(central)
cfg={'purpose':'Complete discrete path JVP or independent common-grid forward difference',
 'element':'HEX27','cells_per_axis':32,'thickness_mm':.5,'load_time_seconds':.004,
 'design_occupancy_mass_affine_inertia_included':True,'fixed_physics':True,
 'baseline_result_sha256':hashlib.sha256((base/'T0p004_compact/result.json').read_bytes()).hexdigest(),
 'source_sha256':{rel:hashlib.sha256((R/rel).read_bytes()).hexdigest() for rel in ['scripts/thin_target_explicit.py','hyperelastic_fem.py','surface_distance.py','pixi.lock']},
 'experiment_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'gradient20_certified':False}
write('input.json',cfg)
shutil.copy2(__file__,a.output/'experiment.py')
if a.action=='probe':
 mass_error=float(jnp.max(jnp.abs(central[1]/ex.nodem-1)))
 class_error=float(jnp.max(jnp.abs(central[2]/ex.mass-1)))
 assert max(mass_error,class_error)<1e-12
 # Replay the old class's unchanged algorithm with identical data, without
 # constructing a second Problem or retaining a second mechanics solver.
 oldfile=experiment/'source_before_path_ad/thin_target_explicit.py'
 spec=importlib.util.spec_from_file_location('old_explicit_snapshot',oldfile)
 oldmodule=importlib.util.module_from_spec(spec);spec.loader.exec_module(oldmodule)
 old=oldmodule.ExplicitXYZ.__new__(oldmodule.ExplicitXYZ);old.__dict__.update(ex.__dict__)
 with np.load(base/'T0p004_compact/field.npz') as cache:
  saved=(jnp.asarray(cache['q']),jnp.asarray(cache['vhalf']),jnp.asarray(cache['time']))
  dt=float(cache['dt'])
 records=[]
 for label,state,count in [('initial',(jnp.zeros((ex.nc,3)),jnp.zeros((ex.nc,3)),jnp.asarray(0.)),32),('saved20_fixed_initial_state',saved,32)]:
  advance,motion=ex.block(dt,count,.004);oldadvance,_=old.block(dt,count,.004)
  oldstate=oldadvance(state);default=advance(state);jax.block_until_ready(default)
  state_error=max(float(jnp.max(jnp.abs(x-y))) for x,y in zip(oldstate,default))
  component_errors=[float(jnp.max(jnp.abs(x-y))) for x,y in zip(oldstate,default)]
  # q is normalized length, v is length/second, t is seconds. Compare
  # displacement effects, not an absolute maximum of unlike dimensions.
  scaled_state_error=max(component_errors[0],dt*component_errors[1],(.2/.004)*component_errors[2])
  default_force=ex.observe(default,dt,motion)['Fz_N'];old_force=old.observe(oldstate,dt,motion)['Fz_N']
  def response(initial,theta):
   f=fields(theta);end=advance(initial,material=material(f))
   return ex.observables(end,dt,motion,material(f))['Fz_N']
  compiled=jax.jit(response)
  gradient=jax.jit(lambda initial,theta:jax.jvp(lambda u:response(initial,u),(theta,),(jnp.array(1.),)))
  start=time.perf_counter();value,derivative=gradient(state,jnp.array(.5));jax.block_until_ready(derivative);cold=time.perf_counter()-start
  start=time.perf_counter();value,derivative=gradient(state,jnp.array(.5));jax.block_until_ready(derivative);warm=time.perf_counter()-start
  delta=.00025;minus=float(compiled(state,.5-delta));plus=float(compiled(state,.5+delta));difference=(plus-minus)/(2*delta)
  relative=abs(float(derivative)-difference)/max(abs(difference),1e-8)
  end=advance(state,material=material(central));quality=ex.observe(end,dt,motion,material(central))
  records.append({'case':label,'steps':count,'old_default_state_max_absolute_difference':state_error,
   'old_default_q_v_time_absolute_differences':component_errors,
   'old_default_displacement_scaled_state_error':scaled_state_error,
   'old_default_force_absolute_difference_N':abs(default_force-old_force),'force_N':float(value),
   'block_derivative_N_per_mm':float(derivative),'independent_block_difference_N_per_mm':difference,
   'derivative_relative_difference':relative,'J_min':quality['J_min'],
   'cold_JVP_seconds':cold,'warm_JVP_seconds':warm,'estimated_full_path_JVP_seconds':warm/count*23586})
  write('probe_progress.json',{'records':records,'gradient20_certified':False})
  assert scaled_state_error<1e-11 and abs(default_force-old_force)<1e-10
  assert relative<.01 and np.isfinite(float(derivative))
 result={'status':'block_probe_pass','mass_equivalence_relative_max':mass_error,
 'class_mass_equivalence_relative_max':class_error,'records':records,'body_seconds':time.perf_counter()-started,
 'peak_RSS_GiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
 'total_lumped_mass_derivative_per_mm':float(jnp.sum(tangent_fields[1])),
 'scope':'Actual target Q2 fixed-state local blocks; not complete 20% path gradient certification',
 'gradient20_certified':False}
 write('result.json',result);print(json.dumps(result,indent=2),flush=True)
else:
 assert json.loads((experiment/'path_ad_probe_units/result.json').read_text())['status']=='block_probe_pass'
 assert json.loads((experiment/'path_comparison.json').read_text())['branch_screen_same_mode']
 baseline=json.loads((base/'T0p004_compact/result.json').read_text())
 schedule=[];previous=0.
 for row in baseline['path'][1:]:
  dt=row['dt_seconds'];count=round((row['time']-previous)/dt)
  assert 0<count<=128 and abs(count*dt-(row['time']-previous))<dt*1e-6
  schedule.append({'dt':dt,'count':count});previous=row['time']
 schedule_hash=hashlib.sha256(json.dumps(schedule,sort_keys=True).encode()).hexdigest()
 theta={'minus':.4975,'plus':.5025,'jvp':.5}[a.action]
 f=fields(jnp.array(theta));jax.block_until_ready(f)
 state=(jnp.zeros((ex.nc,3)),jnp.zeros((ex.nc,3)),jnp.asarray(0.))
 tangent=jax.tree.map(jnp.zeros_like,state);compiled={};path=[];derivatives=[]
 prior_dt=schedule[0]['dt'];last=time.perf_counter();count_steps=0
 cfg.update(thickness_mm=theta,schedule_sha256=schedule_hash,
            schedule_definition='Replay baseline accepted blocks and half-step recentering; never differentiate reject/halve decisions',
            blocks=len(schedule),steps=sum(r['count'] for r in schedule))
 write('input.json',cfg);write('schedule.json',schedule)
 def functions(dt):
  if dt not in compiled:
   advance,motion=ex.block(dt,128,.004)
   def advance_fields(s,f,count):return advance(s,count,material(f))
   step=jax.jit(advance_fields)
   jvp=jax.jit(lambda s,ds,f,df,count:jax.jvp(lambda z,zz:advance_fields(z,zz,count),(s,f),(ds,df)))
   value=jax.jit(lambda s,f:ex.observables(s,dt,motion,material(f))['Fz_N'])
   value_jvp=jax.jit(lambda s,ds,f,df:jax.jvp(lambda z,zz:value(z,zz),(s,f),(ds,df)))
   compiled[dt]=(step,jvp,motion,value,value_jvp)
  return compiled[dt]
 for block,spec in enumerate(schedule):
  dt=spec['dt'];count=spec['count'];step,dual,motion,value,value_dual=functions(dt)
  if dt!=prior_dt:
   def recenter(s,f):
    q,v,t=s;h,_,hdd=motion(t)
    acc=ex.acceleration(q,h,hdd,*material(f))
    return q,v+.5*(prior_dt-dt)*acc,t
   if a.action=='jvp':state,tangent=jax.jvp(recenter,(state,f),(tangent,tangent_fields))
   else:state=recenter(state,f)
  if a.action=='jvp':state,tangent=dual(state,tangent,f,tangent_fields,count)
  else:state=step(state,f,count)
  jax.block_until_ready(state)
  try:row=ex.observe(state,dt,motion,material(f))
  except Exception as invalid:
   np.savez_compressed(a.output/'failed_field.npz',q=np.asarray(state[0]),vhalf=np.asarray(state[1]),time=float(state[2]))
   write('failure.json',{'block':block,'message':str(invalid),'accepted_path':path,'schedule_sha256':schedule_hash})
   raise
  row['dt_seconds']=dt;path.append(row)
  if a.action=='jvp':
   scalar,derivative=value_dual(state,tangent,f,tangent_fields);d=float(derivative)
   if not np.isfinite(d) or not all(np.isfinite(np.asarray(v)).all() for v in tangent):
    write('failure.json',{'block':block,'message':'Nonfinite path tangent','last_valid':row})
    raise ValueError('Nonfinite path tangent')
   derivatives.append(d)
  count_steps+=count;prior_dt=dt
  if block%10==0 or block==len(schedule)-1:
   progress={'block':block+1,'blocks':len(schedule),'steps':count_steps,'compression':row['compression'],
             'wall_seconds':time.perf_counter()-started,'J_min':row['J_min'],'Fz_N':row['Fz_N']}
   if a.action=='jvp':progress['instantaneous_total_derivative_N_per_mm']=derivatives[-1]
   write('progress.json',progress);print('PROGRESS '+json.dumps(progress),flush=True)
 t=np.array([r['time'] for r in path]);hold=t>=.0042-1e-12;force=np.array([r['Fz_N'] for r in path])
 compression=np.array([r['compression'] for r in path]);aa,index=np.unique(compression,return_index=True)
 result={'status':'complete_common_schedule','action':a.action,'thickness_mm':theta,
 'path':path,'schedule_sha256':schedule_hash,'steps':count_steps,
 'hold_mean_Fz_N':float(force[hold].mean()),'body_seconds':time.perf_counter()-started,
 'peak_RSS_GiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
 'force_at_compression_N':{str(c):float(np.interp(c,aa,force[index])) for c in [.1,.15,.2]},
 'gradient20_certified':False}
 if a.action=='jvp':
  derivative=np.array(derivatives);result.update(path_derivative_N_per_mm=derivatives,
   hold_mean_total_derivative_N_per_mm=float(derivative[hold].mean()),
   derivatives_at_compression_N_per_mm={str(c):float(np.interp(c,aa,derivative[index])) for c in [.1,.15,.2]})
 np.savez_compressed(a.output/'field.npz',q=np.asarray(state[0]),vhalf=np.asarray(state[1]),time=float(state[2]),thickness_mm=theta)
 if a.action=='jvp':np.savez_compressed(a.output/'tangent_field.npz',q=np.asarray(tangent[0]),vhalf=np.asarray(tangent[1]))
 write('result.json',result);print('RESULT '+json.dumps({k:v for k,v in result.items() if k not in ['path','path_derivative_N_per_mm']}),flush=True)

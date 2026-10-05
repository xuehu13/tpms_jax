"""Compare same-duration saved-state tangent growth at original/half dt."""
from pathlib import Path
import hashlib,json,os,resource,shutil,sys,time
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
R=Path('/home/xuehu/projects/tpms_jax');sys.path[:0]=[str(R),str(R/'scripts')]
import numpy as np
import jax
import jax.numpy as jnp
from hyperelastic_fem import make_density_hyperelastic_problem
from thin_target_explicit import ExplicitXYZ
from surface_distance import thickness_occupancy
E=R/'validation/large_compression_20261005_r6/gradient20_path_20261005'
out=E/'short_stability_probe';out.mkdir(exist_ok=False);started=time.perf_counter()
with np.load(R/'validation/large_compression_20261005_r6/quadratic_candidate/gauss_field.npz') as f:rho=f['rho'];points=f['physical_quad_points'];distance=jnp.asarray(f['distance'])
def density(p):
    assert np.array_equal(np.asarray(p.physical_quad_points),points)
    return rho
p=make_density_hyperelastic_problem(32,rho_quad=density,eta=1e-4,periodic_axes=(0,1,2),element_degree=2)
ex=ExplicitXYZ(p);del rho,points
@jax.jit
def fields(t):return ex.material_fields(1e-4+(1-1e-4)*thickness_occupancy(distance,t/10,.005/(2*np.log(9))))
f,df=jax.jvp(fields,(jnp.array(.5),),(jnp.array(1.),))
def material(ff):return ((*ex.kernel_geometry[:4],ff[0]),ff[1],ff[2])
with np.load(E/'path_ad_full/field.npz') as a:state=(jnp.asarray(a['q']),jnp.asarray(a['vhalf']),jnp.asarray(a['time']))
with np.load(E/'path_ad_full/tangent_field.npz') as a:tangent=(jnp.asarray(a['q']),jnp.asarray(a['vhalf']),jnp.array(0.))
dt=json.loads((E/'path_ad_full/schedule.json').read_text())[-1]['dt'];duration=64*dt
fullmass=ex.material_fields(jnp.ones_like(f[0]))[2]
fraction=f[2]/fullmass;norms=jnp.sum(tangent[0]**2,axis=1)
node=jnp.argmax(norms)
void_metric={'nodal_mass_fraction_definition':'occupied/reference-full HRZ class mass, not exact Gauss solid/void classification',
 'mass_fraction_at_maximum_q_tangent_node':float(fraction[node]),
 'q_tangent_mass_norm_fraction_in_nodes_with_mass_fraction_below_0p001':float(jnp.sum(f[2]*norms*(fraction<.001))/jnp.sum(f[2]*norms))}
records=[]
for factor,steps in [(1.,64),(.5,128)]:
    trialdt=dt*factor;advance,motion=ex.block(trialdt,steps,.004)
    s,ds=state,tangent
    if factor!=1:
        def recenter(ss,ff):
            q,v,t=ss;h,_,hdd=motion(t)
            return q,v+.5*(dt-trialdt)*ex.acceleration(q,h,hdd,*material(ff)),t
        s,ds=jax.jvp(recenter,(s,f),(ds,df))
    def evolve(ss,ff):return advance(ss,material=material(ff))
    solve=jax.jit(lambda ss,dds:jax.jvp(evolve,(ss,f),(dds,df)))
    begin=time.perf_counter();end,dend=solve(s,ds);jax.block_until_ready(dend)
    row=ex.observe(end,trialdt,motion,material(f))
    force,dual=jax.jvp(lambda ss,ff:ex.observables(ss,trialdt,motion,material(ff))['Fz_N'],(end,f),(dend,df))
    metric=lambda d:float(jnp.sqrt(jnp.sum(f[2][:,None]*(d[0]**2+(duration*d[1])**2))))
    records.append({'dt_factor':factor,'steps':steps,'physical_duration_seconds':duration,'time':float(end[2]),
      'q_tangent_mass_norm_growth':float(jnp.sqrt(jnp.sum(f[2][:,None]*dend[0]**2)/jnp.sum(f[2][:,None]*ds[0]**2))),
      'same_duration_displacement_scaled_q_vhalf_mass_norm_growth':metric(dend)/metric(ds),
      'max_q_tangent_after':float(jnp.max(jnp.abs(dend[0]))),'max_vhalf_tangent_after':float(jnp.max(jnp.abs(dend[1]))),
      'instantaneous_Fz_N':float(force),'instantaneous_total_derivative_N_per_mm':float(dual),
      'J_min':row['J_min'],'body_seconds':time.perf_counter()-begin})
    print(json.dumps(records[-1]),flush=True)
result={'status':'same_duration_local_tangent_probe_complete','records':records,'void_node_indicator':void_metric,
 'body_seconds':time.perf_counter()-started,'peak_RSS_GiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
 'scope':'Continuation from an already amplified tangent; neither full-path half-step certification nor proof of physical instability',
 'inputs_sha256':{rel:hashlib.sha256((E/rel).read_bytes()).hexdigest() for rel in ['path_ad_full/field.npz','path_ad_full/tangent_field.npz']}}
(out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');shutil.copy2(__file__,out/'experiment.py')
print(json.dumps(result,indent=2),flush=True)

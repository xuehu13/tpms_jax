"""Targeted local mass-normalized tangent probe, no new loading path."""
from pathlib import Path
import hashlib,json,os,resource,shutil,sys,time
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
R=Path('/home/xuehu/projects/tpms_jax');sys.path[:0]=[str(R),str(R/'scripts')]
import numpy as np
import jax
import jax.numpy as jnp
from hyperelastic_fem import make_density_hyperelastic_problem
from thin_target_explicit import ExplicitXYZ
E=R/'validation/large_compression_20261005_r6/gradient20_path_20261005'
out=E/'path_stability_probe';out.mkdir(exist_ok=False);started=time.perf_counter()
with np.load(R/'validation/large_compression_20261005_r6/quadratic_candidate/gauss_field.npz') as f:rho=f['rho'];points=f['physical_quad_points']
def density(p):
    assert np.array_equal(np.asarray(p.physical_quad_points),points)
    return rho
problem=make_density_hyperelastic_problem(32,rho_quad=density,eta=1e-4,periodic_axes=(0,1,2),element_degree=2)
ex=ExplicitXYZ(problem);del rho,points
with np.load(E/'path_ad_full/field.npz') as f:q=jnp.asarray(f['q'])
with np.load(E/'path_ad_full/tangent_field.npz') as f:dq=jnp.asarray(f['q'])
dt=json.loads((E/'path_ad_full/schedule.json').read_text())[-1]['dt']
sqrtmass=jnp.sqrt(ex.mass)[:,None]
@jax.jit
def operator(z):
    d=(z/sqrtmass).at[ex.pin].set(0.)
    r=jax.jvp(lambda x:ex.force(x,-.2),(q,),(d,))[1]
    return (ex.reduce(r)/sqrtmass).at[ex.pin].set(0.)
rng=np.random.default_rng(20261005)
z=jnp.asarray(rng.normal(size=q.shape)).at[ex.pin].set(0.);z/=jnp.linalg.norm(z)
u=jnp.asarray(rng.normal(size=q.shape)).at[ex.pin].set(0.);u/=jnp.linalg.norm(u)
az=operator(z);au=operator(u);jax.block_until_ready(au)
left=float(jnp.vdot(u,az));right=float(jnp.vdot(au,z))
symmetry=abs(left-right)/max(abs(left),abs(right),1e-30)
history=[]
for k in range(24):
    az=operator(z);rayleigh=float(jnp.vdot(z,az));norm=float(jnp.linalg.norm(az))
    residual=float(jnp.linalg.norm(az-rayleigh*z))/max(norm,1e-30)
    row={'iteration':k+1,'rayleigh_per_s2':rayleigh,'dt_squared_rayleigh':dt**2*rayleigh,'relative_eigen_residual':residual}
    history.append(row);z=az/norm
    if k%4==0:print(json.dumps(row),flush=True)
t=sqrtmass*dq;t/=jnp.linalg.norm(t)
at=operator(t);tangent_rayleigh=float(jnp.vdot(t,at))
maximum=max(r['rayleigh_per_s2'] for r in history)
result={'status':'local_tangent_probe_complete','terminal_dt_seconds':dt,
 'symmetry_relative_difference':symmetry,'power_history':history,
 'max_observed_positive_Rayleigh_per_s2':maximum,
 'dt_squared_positive_Rayleigh':dt**2*maximum,
 'positive_frequency_dt_upper_limit_from_lower_bound_seconds':2/np.sqrt(maximum) if maximum>0 else None,
 'existing_dt_violates_local_central_difference_positive_mode_limit':bool(maximum>0 and dt**2*maximum>4),
 'actual_path_tangent_Rayleigh_per_s2':tangent_rayleigh,
 'actual_path_tangent_dt_squared_Rayleigh':dt**2*tangent_rayleigh,
 'scope':'Local saved-state Rayleigh estimates, not global spectrum/stability or a full-path derivative certificate',
 'body_seconds':time.perf_counter()-started,'peak_RSS_GiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
 'inputs_sha256':{rel:hashlib.sha256((E/rel).read_bytes()).hexdigest() for rel in ['path_ad_full/field.npz','path_ad_full/tangent_field.npz','path_ad_full/result.json']},
 'sources_sha256':{rel:hashlib.sha256((R/rel).read_bytes()).hexdigest() for rel in ['scripts/thin_target_explicit.py','hyperelastic_fem.py']}}
(out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
shutil.copy2(__file__,out/'experiment.py')
print(json.dumps({k:v for k,v in result.items() if k not in ['power_history','inputs_sha256','sources_sha256']},indent=2),flush=True)

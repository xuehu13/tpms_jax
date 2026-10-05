"""Validate full-path JVP and one curve loss against independent path differences."""
from pathlib import Path
import json,hashlib,os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64',True)
R=Path('/home/xuehu/projects/tpms_jax');root=R/'validation/large_compression_20261005_r6';out=root/'gradient20_path_20261005'
read=lambda p:json.loads(p.read_text())
paths=read(out/'path_comparison.json');ad=read(out/'path_ad_full/result.json')
assert paths['all_accepted_schedules_identical'],'Independent adaptive paths require common-grid recomputation'
assert ad['schedule_sha256']==paths['accepted_schedule_hashes']['center']
f0=ad['hold_mean_Fz_N'];baseline=paths['paths']['center']['hold_mean_Fz_N']
center_gap=abs(f0/baseline-1)
central_path=read(root/'quadratic_candidate/T0p004_compact/result.json')['path'][1:]
assert len(central_path)==len(ad['path'])
time_gap=max(abs(x['time']-y['time']) for x,y in zip(central_path,ad['path']))
assert time_gap<1e-12
old_force=np.array([row['Fz_N'] for row in central_path])
new_force=np.array([row['Fz_N'] for row in ad['path']])
curve_primal_gap=float(np.sqrt(np.mean((old_force-new_force)**2))/np.max(np.abs(old_force)))
physical_finite=bool(all(np.isfinite(list(row.values())).all() and row['J_min']>0 for row in ad['path']))
delta=.0025;fd=(paths['paths']['plus']['hold_mean_Fz_N']-paths['paths']['minus']['hold_mean_Fz_N'])/(2*delta)
derivative=ad['hold_mean_total_derivative_N_per_mm'];error=abs(derivative-fd)/max(abs(fd),1e-8)
samples=[.1,.15,.2];target=[]
shell=read(root/'abaqus/explicit_T0p040/shell.json');rows=shell['force_path']
aa=np.array([r['compression'] for r in rows]);ff=np.array([r['Fz_N'] for r in rows]);aa,index=np.unique(aa,return_index=True)
for c in samples:target.append(shell['hold_mean_Fz_N'] if c==.2 else float(np.interp(c,aa,ff[index])))
peak=read(root/'comparison_summary.json')['Standard_peak_force_N'];target=np.array(target)
force=np.array([ad['hold_mean_Fz_N'] if c==.2 else ad['force_at_compression_N'][str(c)] for c in samples])
dual=np.array([derivative if c==.2 else ad['derivatives_at_compression_N_per_mm'][str(c)] for c in samples])
plus=np.array([paths['paths']['plus']['hold_mean_Fz_N'] if c==.2 else paths['paths']['plus']['force_at_compression_N'][str(c)] for c in samples])
minus=np.array([paths['paths']['minus']['hold_mean_Fz_N'] if c==.2 else paths['paths']['minus']['force_at_compression_N'][str(c)] for c in samples])
loss=lambda forces:.5*jnp.sum(((forces-jnp.asarray(target))/peak)**2)
value,total_loss_derivative=jax.jvp(loss,(jnp.asarray(force),),(jnp.asarray(dual),))
loss_difference=(float(loss(plus))-float(loss(minus)))/(2*delta)
loss_error=abs(float(total_loss_derivative)-loss_difference)/max(abs(loss_difference),1e-8)
curvefd=(plus-minus)/(2*delta);sample_errors=np.abs(dual-curvefd)/np.maximum(np.abs(curvefd),1e-8)
finite=bool(np.isfinite(dual).all() and np.isfinite(float(total_loss_derivative)))
passed=bool(center_gap<1e-6 and curve_primal_gap<1e-6 and error<=.01 and np.max(sample_errors)<=.01 and loss_error<=.01 and finite and physical_finite)
result={'status':'path_gradient_pass' if passed else 'path_gradient_not_passed',
 'same_accepted_schedule':True,'central_primal_hold_force_relative_difference':center_gap,
 'central_primal_curve_RMS_over_peak_difference':curve_primal_gap,
 'central_time_grid_max_absolute_difference_seconds':time_gap,
 'all_observed_states_finite_and_positive_Gauss_volume':physical_finite,
 'central_primal_equivalence_limit':1e-6,'gradient_relative_limit':.01,
 'hold_force_JVP_N_per_mm':derivative,'hold_force_independent_difference_N_per_mm':fd,
 'hold_gradient_relative_difference':error,
 'compression_samples':samples,'force_N':force.tolist(),'target_Abaqus_force_N':target.tolist(),
 'sample_JVP_N_per_mm':dual.tolist(),'sample_differences_N_per_mm':curvefd.tolist(),
 'sample_derivative_relative_differences':sample_errors.tolist(),
 'loss_definition':'0.5 sum(((signed F at 10%,15%,20% - frozen Abaqus reference)/Standard peak)^2); 20% uses hold mean',
 'loss':float(value),'loss_total_derivative_per_mm':float(total_loss_derivative),
 'loss_independent_difference_per_mm':loss_difference,'loss_derivative_relative_difference':loss_error,
 'gradient20_certified_for_local_thickness_discrete_path':passed,
 'physical_shape_gradient_or_contact_certified':False,
 'independent_Abaqus20_thickness_derivative_checked':False,
 'scope':'Local thickness derivative of the declared XYZ elastic explicit discrete path; no rejected-control derivative or training',
 'hashes':{name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ['path_ad_full/result.json','path_comparison.json']}}
(out/'gradient_validation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))

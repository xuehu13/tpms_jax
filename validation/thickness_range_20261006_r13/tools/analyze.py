"""Matched thickness analysis; reuse the frozen r12 saved-field evaluator.

The only evaluator adaptations are physical thickness occupancy and the matched
shell path; no material, FEM, or time stepping implementation is introduced.
"""
from pathlib import Path
import argparse, ast, hashlib, json, os, sys, time
os.environ.setdefault('JAX_PLATFORMS','cpu')
import numpy as np
from scipy.special import expit
import basix, jax, jax.numpy as jnp

R=Path('/home/xuehu/projects/tpms_jax');B=R/'validation/thickness_range_20261006_r13'
p=argparse.ArgumentParser();p.add_argument('tag',choices=['t0p45','t0p55']);a=p.parse_args()
C=B/a.tag;P=C/'T0p004';O=C/'analysis';O.mkdir(exist_ok=False)
sys.path.insert(0,str(R))
from jax_fem.basis import get_elements
from hyperelastic_fem import (objective_void_energy,objective_void_first_piola,void_nh_weight,
                              void_nh_cutoff,objective_void_requires_positive_J)
from surface_distance import PeriodicSurfaceDistance
Q=R/'validation/large_compression_20261005_r6/quadratic_candidate'
G=R/'validation/thin_target_20261004_r5/gauss_field.npz'
A=C/'abaqus';L,N,ETA=10.,32,1e-4
T=json.loads((B/'preparation.json').read_text())[a.tag]['thickness_mm']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

original=R/'validation/void_continuation_20261006_r12/experiment.py'
source=original.read_text()
changes=[("oldphi=f['rho'];oldqp=", "oldphi=expit((T/(2*L)-f['distance'])/(.05/(2*np.log(9))/L));oldqp="),
         ("    with np.load(Q/'T0p004_compact/field.npz') as f:ow=fluct(f['q'])\n",''),
         ("for name,ref in [('old_fast',ow),('shell',sw)]:", "for name,ref in [('shell',sw)]:")]
for old,new in changes:
    assert source.count(old)==1,old
    source=source.replace(old,new)
tree=ast.parse(source)
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['curve','probe']]
assert len(functions)==2
exec(compile(ast.Module(body=functions,type_ignores=[]),str(original)+' [matched-thickness evaluator]','exec'),globals())

start=time.perf_counter()
complete=(P/'result.json').exists();record=P/('result.json' if complete else 'failure.json')
rec=read(record);rows=rec['path'] if complete else rec['accepted_path'];cfg=read(P/'input.json')
state=P/('field.npz' if complete else 'last_valid_field.npz')
shell=read(A/'explicit_T0p040/shell.json');shellcfg=read(A/'explicit_T0p040/input.json')
base=read(R/'validation/void_continuation_20261006_r12/T0p004/input.json')
fixed=['N','cell_size_mm','E_MPa','nu','eta','interface_10_90_mm','mechanical_periodic_axes',
       'load_time_seconds','Gauss_field_sha256','element_type','element_degree','Gauss_points_per_cell',
       'mass_lumping','solid_density_tonne_per_mm3','void_mass_floor','source_sha256','experiment_sha256']
for k in fixed:assert cfg[k]==base[k],(k,cfg[k],base[k])
assert cfg['thickness_mm']==shellcfg['thickness_mm']==T
assert cfg['material_model']=='objective_void'
assert cfg['virtual_continuation_max_J']==base['virtual_continuation_max_J']==.1
assert cfg['virtual_NH_weight_bounds']==base['virtual_NH_weight_bounds']==[.001,.01]
assert cfg['evaluated_occupancy_sha256']!=base['evaluated_occupancy_sha256']
for k in ['cell_size_mm','E_MPa','nu','mechanical_periodic_axes']:assert cfg[k]==shellcfg[k]
assert shellcfg['load_time_seconds']==.04 and shellcfg['no_contact'] and shellcfg['no_plasticity']
files=[record,state,P/'input.json',A/'explicit_T0p040/shell.json',A/'explicit_T0p040/input.json',
       A/'explicit_T0p040/shell_field.npz',Q/'gauss_field.npz',G,original,
       R/'hyperelastic_fem.py',R/'scripts/thin_target_explicit.py',R/'surface_distance.py',R/'pixi.lock']
before={str(f):sha(f) for f in files}
write(O/'analysis_manifest.json',{'input_sha256':before,'analyzer_sha256':sha(Path(__file__)),
       'reused_evaluator_sha256':sha(original),'explicit_evaluator_adaptations':changes,
       'environment':{'python':sys.version,'numpy':np.__version__,'jax':jax.__version__,'basix':basix.__version__},
       'matched_thickness_mm':T,'fixed_input_keys':fixed,'time_grid_note':'Each physical thickness estimates its own stable dt.',
       'new_FEM_implementations':0,'new_full_AD_jobs':0})

aa,ff=curve(rows);sa,sf=curve(shell['force_path']);grid=np.linspace(.001,.2,1000)
grid=grid[grid<=aa[-1]+1e-12];reference=np.interp(grid,sa,sf);candidate=np.interp(grid,aa,ff)
den=read(B/'input.json')['gates']['fixed_curve_denominator_N']
metric={'thickness_mm':T,'candidate_completed_20':complete,'changed_factor':'Physical wall thickness in both models',
        'last_accepted_compression':rows[-1]['compression'],
        'curve_RMS_over_Standard_peak':float(np.sqrt(np.mean((candidate-reference)**2))/den),
        'curve_RMS_denominator_N':den,'maximum_absolute_curve_difference_N':float(np.max(abs(candidate-reference))),
        'common_range':[float(grid[0]),float(grid[-1])],
        'rejected_blocks':len(rec['rejected_blocks']),'body_seconds':rec['body_seconds'],
        'minimum_monitored_actual_J':min(r['J_min'] for r in rows),
        'minimum_monitored_required_positive_J':min(r['required_positive_J_min'] for r in rows if r['required_positive_J_min'] is not None),
        'monitored_invalid_material_points':max(r['invalid_material_points'] for r in rows),
        'reference_quality_certified':all(shell['checks'].values()),'shell_checks':shell['checks'],
        'shell_energy_drift':shell['global_energy_drift_relative_work'],
        'shell_loading_fraction_KE_below_5pct':shell['loading_time_fraction_KE_below_5pct'],
        'physical_replacement_certified':False,'gradient20_certified':False,
        'rate_check_this_thickness_completed':False,'sampling_is_everywhere_certificate':False}
if complete:
    times=np.array([r['time'] for r in rows]);compression=np.array([r['compression'] for r in rows])
    forces=np.array([r['Fz_N'] for r in rows]);u=np.array([r['energy_N_mm'] for r in rows]);ke=np.array([r['KE_N_mm'] for r in rows])
    hold=times>=1.05*cfg['load_time_seconds']-1e-12;assert hold.sum()>=2
    mean=float(np.mean(forces[hold]));work=float(np.sum(.5*(forces[1:]+forces[:-1])*(-L*np.diff(compression))))
    dt=np.r_[0.,np.diff(times)];loading=(compression>=.01)&(times<=cfg['load_time_seconds']+1e-12)
    ratio=ke/np.maximum(u,1e-30)
    metric.update(hold_mean_Fz_N=mean,shell_hold_mean_Fz_N=shell['hold_mean_Fz_N'],
        terminal_force_difference_vs_shell=abs(mean/shell['hold_mean_Fz_N']-1),
        hold_min_max_Fz_N=[float(forces[hold].min()),float(forces[hold].max())],
        input_work_N_mm=work,shell_input_work_N_mm=shell['macro_work_from_RF_U_N_mm'],
        input_work_difference_vs_shell=abs(work/shell['macro_work_from_RF_U_N_mm']-1),
        energy_work_gap=abs(u[-1]+ke[-1]-u[0]-ke[0]-work)/abs(work),
        loading_time_fraction_KE_below_5pct=float(np.sum(dt[loading]*(ratio[loading]<=.05))/np.sum(dt[loading])),
        terminal_KE_over_U=float(ratio[-1]),terminal_energy_N_mm=float(u[-1]),
        peak_magnitude_N=float(ff.max()),peak_compression=float(aa[ff.argmax()]),
        shell_peak_magnitude_N=float(sf.max()),shell_peak_compression=float(sa[sf.argmax()]),
        peak_magnitude_difference=abs(float(ff.max()/sf.max())-1),
        initial_dt_seconds=rec['initial_dt_seconds'],terminal_dt_seconds=rec['terminal_dt_seconds'],
        peak_RSS_GiB=rec['peak_RSS_GiB'])
    metric['response_target_only_pass']=all(metric[k]<=.1 for k in ['terminal_force_difference_vs_shell','curve_RMS_over_Standard_peak','input_work_difference_vs_shell'])
    metric['response_at_compressions']=[{'compression':c,'candidate_force_magnitude_N':float(np.interp(c,aa,ff)),
                                        'shell_force_magnitude_N':float(np.interp(c,sa,sf))} for c in [.01,.05,.1,.15,.2]]
else:
    metric.update(failure=rec['message'],response_target_only_pass=False)
write(O/'comparison_before_probe.json',metric)
with np.load(state) as f:
    q=f['q'];assert int(f['N'])==N and abs(float(f['time'])-rows[-1]['time'])<1e-12
rules,mode=probe(q,rows[-1])
metric.update(dense_sampling_J_min=rules['8']['raw_J_min'],dense_sampling_nonpositive_points=rules['8']['raw_nonpositive_points'],
    dense_sampling_required_positive_J_min=rules['8']['required_positive_J_min'],
    dense_sampling_invalid_material_points=rules['8']['invalid_material_points'],
    dense_sampling_material_defined=rules['8']['candidate_full_energy_defined'],mode=mode,
    analysis_seconds=time.perf_counter()-start)
metric['material_domain_sampled_pass']=bool(metric['monitored_invalid_material_points']==0 and all(
    r['invalid_material_points']==0 and r['candidate_full_energy_defined'] for r in rules.values()))
metric['JAX_energy_inertia_gates_pass']=bool(complete and metric['energy_work_gap']<.01 and
    metric['terminal_KE_over_U']<.05 and metric['loading_time_fraction_KE_below_5pct']>=.95)
metric['scoped_engineering_response_pass']=bool(metric['response_target_only_pass'] and
    metric['material_domain_sampled_pass'] and metric['JAX_energy_inertia_gates_pass'])
write(O/'comparison.json',metric)
np.savetxt(O/'curves.csv',np.c_[grid,candidate,reference],delimiter=',',header='compression,JAX_force_N,matched_shell_force_N',comments='')
assert before=={str(f):sha(f) for f in files}
print(json.dumps(metric,indent=2),flush=True)

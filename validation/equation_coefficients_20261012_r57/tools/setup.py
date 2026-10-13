"""Prepare isolated r57; old scientific files remain read-only."""
from pathlib import Path
import sys,json,hashlib,shutil,time
import numpy as np
R=Path('/home/xuehu/projects/tpms_jax');B=R/'validation/equation_coefficients_20261012_r57'
stage=Path(__file__).resolve().parent
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not B.exists(),'Never overwrite a previous batch'
for p in ['input','tools','checks','AD','FD','analysis']:(B/p).mkdir(parents=True,exist_ok=True)
for p in stage.glob('*.py'):shutil.copy2(p,B/'tools'/p.name)
old=R/'validation/platen_xy_diverse_batch_20261012_r55'
cfg=json.loads((old/'diverse_28/input/protocol.json').read_text())
cfg.update(case_id='diverse_28_true_equation_coefficients_r57',hold_time_s=0.,
           stage='Equation coefficient derivative verification: strict implicit midsurface, moving rim, fixed local contact partitions; no Abaqus comparison',
           fixed_dt_max_s=5.8e-8)
write(B/'input/model.json',cfg)
with np.load(old/'sources/diverse_28/gauss_field.npz') as z:
    qp=z['physical_quad_points']; gauss_shape=z['distance'].shape
with np.load(old/'diverse_28/input/cut_face_contact.npz') as z:cap=dict(z)
with np.load(old/'diverse_28/input/contact_assignment.npz') as z:assignment=z['rim_assignment']
# Raw cap weights are independent face-Gauss weights; do not divide by occupancy.
sys.path.insert(0,str(R/'validation/platen_xy_G_20261010_r51/tools'))
from prepare import face_cache
raw=face_cache(32)
np.testing.assert_array_equal(raw['points'],cap['points'])
np.savez_compressed(B/'input/queries.npz',points=np.vstack([qp.reshape((-1,3)),cap['points']]),n_gauss=int(np.prod(gauss_shape)),
                    gauss_shape=gauss_shape,cap_raw_area=raw['area_weight'],assignment=assignment)
guard_paths=[R/p for p in ['hyperelastic_fem.py','surface_distance.py','pbc.py','fem.py','scripts/thin_target_explicit.py',
 'validation/platen_xy_gradients_20261011_r53/tools/design.py','validation/platen_xy_gradients_20261011_r53/tools/reverse.py',
 'validation/platen_xy_gradients_20261011_r53/status.json']]
guard_paths += [old/p for p in ['sources/diverse_28/surface.npz','sources/diverse_28/gauss_field.npz',
 'sources/diverse_28/case_used.json','diverse_28/input/rimline_contact.npz','diverse_28/input/contact_assignment.npz']]
write(B/'checks/source_guard_before.json',{p.relative_to(R).as_posix():sha(p) for p in guard_paths})
protocol=dict(authorization='2026-10-12 user: 试一下曲面方程的各系数，看看jax-fem能不能正确处理。',
 equation='f=c1*cos(X)*cos(Z)+c2*sin(X)*sin(Y)*sin(Z)+c0=0; X=2pi*x/L',
 parameters=['c1','c2','c0'],units=['dimensionless']*3,baseline=[-3.4,4.6,2.],
 relative_fd_step=1e-5,fd_halving=True,parameter_scales=[3.4,4.6,2.],
 geometry='strict nonlinear roots along fixed boundary-tangent baseline normals; fixed triangle connectivity; true periodic triangle distance; t=.5mm,w=.05mm unchanged',
 contact='moving rim Q2 interpolation differentiated; fixed reference cap/rim partition, audited against regenerated nearest assignment at all declared perturbations',
 required_objective='full trajectory loading work; complete VJP all three coefficients; complete JVP c2 check',
 auxiliary_objective='terminal force: complete JVP/FD only for c2, FD only for other coefficients; not all-force AD certification',
 thresholds=dict(relative_AD_FD=.01,relative_FD_halving=.01,nearzero_scaled_cutoff=1e-6,nearzero_scaled_absolute_error=1e-5,objective_scales=[1.,1.]),
 candidate_compressions=[.01,.005],selection='Before full paths, choose1% if benchmark predicts within180min total, else0.5%; if neither fits stop. Never choose endpoint by gradient agreement.',
 total_wall_budget_s=10800,geometry_and_benchmark_budget_s=1500,
 failure_policy='Preserve failures; no tolerance relaxation, no automatic additional FD ladder, no long reverse without geometry and short duality checks',
 excludes=['Abaqus','training','Git','heartbeat','r53 resumption','20% gradient','topology changes'],started_unix=time.time())
write(B/'input/protocol.json',protocol)
write(B/'status.json',dict(stage='prepared_geometry_pending',active_jobs=[],new_scope='true_equation_coefficients',r53='unchanged_paused',heartbeat='PAUSED'))
print(json.dumps(dict(batch=str(B),queries=len(qp.reshape((-1,3)))+len(cap['points']),parameters=protocol['parameters'])))

"""Prepare two bounded r51 contact diagnostics; preserve all original results."""
from pathlib import Path
import json, hashlib, shutil, sys, datetime
import numpy as np

R = Path('/home/xuehu/projects/tpms_jax')
D = R / 'validation/platen_xy_G_20261010_r51'
OLD = D / 'diagnostics/contact_origin_20261010'
B = D / 'diagnostics/contact_support_control_20261010'
sys.path.insert(0, str(D / 'tools'))
from contact_origin import shapes_at

def write(p, value):
    p.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

if B.exists():
    assert not (B/'input/protocol.json').exists() and not (B/'status.json').exists(), 'Do not overwrite an experiment'
    assert not list((B/'jax').iterdir()), 'No recovery after a job starts'
for part in ('input', 'jax', 'analysis'):
    (B / part).mkdir(parents=True, exist_ok=True)
protected = json.loads((OLD / 'input/frozen_before.json').read_text())
protected['validation/platen_xy_G_20261010_r51/tools/platen_model.py'] = hashlib.sha256((D/'tools/platen_model.py').read_bytes()).hexdigest()
for p in OLD.rglob('*'):
    if p.is_file() and p.suffix not in ('.log', '.pyc'):
        protected[str(p.relative_to(R))] = hashlib.sha256(p.read_bytes()).hexdigest()
for rel, sha in protected.items():
    assert hashlib.sha256((R/rel).read_bytes()).hexdigest() == sha, rel
write(B/'input/frozen_before.json', protected)
shutil.copy2(D/'tools/run_jax.py', B/'input/run_jax_before.py')
cfg = json.loads((D/'input/protocol.json').read_text())
line_result = json.loads((OLD/'jax/rimline_T1ms/result.json').read_text())
steps = round((cfg['load_time_s'] + cfg['hold_time_s']) / line_result['initial_dt_s'])
assert steps == 19877 and not line_result['rejections']
fixed_dt = (cfg['load_time_s'] + cfg['hold_time_s']) / steps
assert abs(fixed_dt / line_result['initial_dt_s'] - 1) < 1e-13
protocol = {
    'frozen_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'question': 'Separate time discretization from end-contact support; test intermediate spatial support without fitting the shell curve.',
    'fixed': 'G geometry, t=.5mm, NH E10 nu.3, N32 HEX27/27 points, occupancy, HRZ, mass, XY/free ax ay, kn320, T1ms, 2pct plus .2ms hold.',
    'max_new_dynamic_jobs': 2, 'per_job_wall_budget_seconds': 2350,
    'common_nominal_steps': steps, 'common_dt_s': fixed_dt,
    'original_actual_accepted_steps': line_result['steps'],
    'step_count_note': 'The existing runner may add one final fractional step due to floating-point time accumulation; the initial dt and final time are the controlled values.',
    'case_1': 'Original cap contact; only time step reduced to match existing rimline run.',
    'case_2': 'Same common time step; points halfway between original cap points and assigned existing rim nodes; all original weights conserved.',
    'diagnostic_not_physical_correction': True,
    'support_limit': 'Halfway map changes support and spatial interpolation, including tangential nearest-node effects. It is not a new physical thickness, optimized parameter, or geometry AD.',
    'reference': str(D/'abaqus/result.json'),
    'loading_RMS_denominator_N': 0.7241182386842194,
    'loading_grid': '1001 equally spaced compressions from 0 to .02; linear interpolation; loading segment only.',
    'hold_window_s': [.0011, .0012],
    'hold_average': 'Trapezoidal time integral with interpolated window endpoints.',
    'comparison': 'Record signed 1pct/2pct force, loading RMS, hold mean, work, displacements, J, energy and actual steps/rejections. No post-hoc threshold changes.',
    'full20_submitted': False, 'new_design_AD': False,
}
write(B/'input/protocol.json', protocol)
with np.load(D/'input/cut_face_contact.npz') as z:
    cap = dict(z)
with np.load(OLD/'input/rimline_contact.npz') as z:
    line = dict(z)
with np.load(OLD/'input/contact_assignment.npz') as z:
    assignment = z['rim_assignment']
half_points = .5 * (cap['points'] + line['points'][assignment])
ids, shape = shapes_at(half_points, cfg['N'])
half = {**cap, 'points': half_points, 'node_ids': ids, 'shape': shape}
np.savez_compressed(B/'input/half_support_contact.npz', **half)
for side in (0, 1):
    np.testing.assert_array_equal(half['area_weight'][half['side']==side], cap['area_weight'][cap['side']==side])
# This scalar local rocking probe describes the contact operator only, not FE response.
delta = cap['points'] - line['points'][assignment]
slope = np.array([.7, -.4, 0.])
gap = delta @ slope
rock = lambda g: .5*cfg['contact_normal_stiffness_MPa_per_mm']*cfg['L_mm']*np.sum(cap['area_weight']*np.maximum(-g, 0.)**2)
ratio = rock(.5*gap) / rock(gap)
np.testing.assert_allclose(ratio, .25, atol=1e-14)
# Independent contact energy/force virtual work at the new interpolation points.
nn = 2*cfg['N']+1
grid_z = np.tile(np.arange(nn)/(nn-1), nn*nn)
q = np.where(grid_z < .5, -1e-4, 1e-4)
direction = np.random.default_rng(51).normal(size=q.shape)
def energy_and_grad(v):
    z = half_points[:,2] + np.sum(v[ids]*shape, axis=1)
    gap = np.where(cap['side']==0, z, 1-z)
    pen = np.maximum(-gap, 0.)
    f = cfg['contact_normal_stiffness_MPa_per_mm']*cfg['L_mm']*cap['area_weight']*pen
    grad = np.zeros_like(v)
    np.add.at(grad, ids.ravel(), (np.where(cap['side']==0, -1., 1.)[:,None]*f[:,None]*shape).ravel())
    return .5*np.sum(f*pen), grad
eps = 1e-8
anal = float(energy_and_grad(q)[1]@direction)
fd = (energy_and_grad(q+eps*direction)[0]-energy_and_grad(q-eps*direction)[0])/(2*eps)
vw = abs(fd-anal)/max(abs(anal),1e-30)
assert vw < 1e-6
write(B/'input/contact_checks.json', {
    'area_preserved_exactly': True, 'all_weights_nonnegative': bool(np.all(half['area_weight']>=0)),
    'contact_points': len(half_points),
    'uniform_gap_force_and_energy_unchanged': True,
    'local_affine_rocking_energy_half_over_full': ratio,
    'rocking_probe_scope': 'Assigned local rim-node centers; uniform prescribed local slope; one-sided penalty. Not a dynamic FE result or isolated moment measurement.',
    'contact_virtual_work_relative_error': vw,
    'mean_occupied_distance_to_assigned_rim_mm': [float(np.sum(cap['area_weight']*np.linalg.norm(s*delta,axis=1))/cap['area_weight'].sum()*cfg['L_mm']) for s in [1.,.5,0.]],
})
for name, cache, stage in [
    ('cap_common_dt', D/'input/cut_face_contact.npz', 'Original cap, matched fine time step; 2pct diagnostic'),
    ('half_support_common_dt', B/'input/half_support_contact.npz', 'Artificial halfway contact support, matched fine time step; 2pct diagnostic'),
]:
    case = {**cfg, 'case_id': 'G_r51_'+name, 'contact_cache_path': str(cache),
            'diagnostic_fixed_steps': steps, 'stage': stage,
            'contact_cache_sha256': hashlib.sha256(cache.read_bytes()).hexdigest()}
    write(B/'input'/f'{name}.json', case)
# Minimal experimental-runner option; old inputs retain their existing behavior.
p = D/'tools/run_jax.py'
source = p.read_text()
before = '    steps=math.ceil(end/dt);dt=end/steps;initial_dt=dt\n'
after = '''    steps=math.ceil(end/dt)
    if 'diagnostic_fixed_steps' in cfg:
        requested_steps=int(cfg['diagnostic_fixed_steps'])
        requested_dt=end/requested_steps
        if requested_steps<=0 or requested_dt>dt*(1+1e-10):raise ValueError('Requested diagnostic step exceeds the initial conservative estimate')
        bound={**bound,'automatic_initial_dt_s':dt,'diagnostic_fixed_steps':requested_steps}
        steps=requested_steps
    dt=end/steps;initial_dt=dt
'''
assert source.count(before)==1
p.write_text(source.replace(before,after))
write(B/'status.json', {'stage':'prepared_two_bounded_diagnostics','active_jobs':[], 'full20_submitted':False,'new_design_AD':False})
print(json.dumps({'directory':str(B),'steps':steps,'dt_s':fixed_dt,'protected_files':len(protected),'virtual_work_error':vw},indent=2))

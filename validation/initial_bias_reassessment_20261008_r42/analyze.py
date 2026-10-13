"""Read existing equilibria; account for occupancy change without a new solve."""
from pathlib import Path
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import sys, json, hashlib, time
import numpy as np
import basix
from scipy.special import expit

R = Path('/home/xuehu/projects/tpms_jax')
D = R/'validation/initial_bias_reassessment_20261008_r42'
sys.path.insert(0, str(R))
from jax_fem.basis import get_elements
from hyperelastic_fem import MU, KAPPA
from surface_distance import PeriodicSurfaceDistance

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def freeze():
    for name, digest in read(D/'frozen_before.json').items():
        assert hashlib.sha256((R/name).read_bytes()).hexdigest() == digest, name

freeze()
cfg = read(D/'protocol.json')
tick = time.perf_counter()
c = np.load(R/'validation/geometry_transfer_20261006_r15/gauss_field.npz')
sel = np.load(R/'validation/local_quadrature_20261008_r33/selected_cells.npy')
assert len(sel) == 1941
fam, cell, _, _, degree, order = get_elements('HEX27')
el = basix.create_element(fam, cell, degree)
origins = 2*np.indices((32,)*3).reshape(3, -1).T
local = np.rint(2*el.points[order]).astype(int)
idx = origins[:, None, :] + local[None, :, :]
cells = (idx[:, :, 0]*65 + idx[:, :, 1])*65 + idx[:, :, 2]
g0 = el.tabulate(1, c['physical_quad_points'][0]*32)[1:, :, :, 0].transpose(1, 2, 0)[:, order, :]*32
ell = .05/(2*np.log(9))
assert np.max(np.abs(expit((.25-c['distance']*10)/ell)-c['rho'])) < 1e-12
lam = KAPPA-2*MU/3
eta = cfg['eta']

def unit_energy(q, ids, which, grad, H):
    nodal = q[ids[cells[which]]]
    v = H + np.einsum('cni,qnj->cqij', nodal, grad, optimize=True)
    e = .5*(v + v.swapaxes(-1, -2))
    return .5*lam*np.trace(e, axis1=-2, axis2=-1)**2 + MU*np.sum(e*e, axis=(-1, -2))

def accumulate(acc, qs, qb, ids, H, which, grad, weights, smooth, binary):
    us = unit_energy(qs, ids, which, grad, H)
    ub = unit_energy(qb, ids, which, grad, H)
    ud = unit_energy(qs-qb, ids, which, grad, np.zeros((3, 3)))
    ss = eta+(1-eta)*smooth
    sb = eta+(1-eta)*binary
    terms = {
        'U_s_q_s': ss*us, 'U_b_q_s': sb*us,
        'U_s_q_b': ss*ub, 'U_b_q_b': sb*ub,
        'compatible_difference_energy_b': sb*ud,
        'fixed_state_inside_addition': (1-eta)*binary*(1-smooth)*us,
        'fixed_state_outside_removal': -(1-eta)*(1-binary)*smooth*us,
    }
    for key, value in terms.items():
        acc[key] = acc.get(key, 0.) + float(np.sum(value*weights))

pairs = [
    ('original27', 'initial_tangent_20261008_r30', 'jax_result.json',
     'binary_occupancy_20261008_r40/original27', 'jax_result.json'),
    ('selected12_rest27', 'local_reequilibrium_20261008_r34', 'result.json',
     'binary_occupancy_20261008_r40/mixed12', 'result.json'),
]
results = []
for label, smooth_dir, sr, binary_dir, br in pairs:
    sd = R/'validation'/smooth_dir
    bd = R/'validation'/binary_dir
    a = np.load(sd/'linear_state.npz'); b = np.load(bd/'linear_state.npz')
    rs, rb = read(sd/sr), read(bd/br)
    assert rs['status'] == rb['status'] == 'ok'
    assert np.array_equal(a['class_ids'], b['class_ids'])
    assert np.array_equal(a['H'], b['H'])
    assert a['H'][2, 2] == cfg['macro_Hzz']
    ids, H = a['class_ids'], a['H']
    acc = {}
    base_cells = np.arange(32768)
    if label == 'selected12_rest27':
        base_cells = np.setdiff1d(base_cells, sel)
    for start in range(0, len(base_cells), 256):
        which = base_cells[start:start+256]
        accumulate(acc, a['q'], b['q'], ids, H, which, g0,
                   c['JxW'][which]*1000, c['rho'][which],
                   (c['distance'][which]*10 <= .25).astype(float))
    if label == 'selected12_rest27':
        z, w = np.polynomial.legendre.leggauss(12); z = (z+1)/2; w /= 2
        ref = np.array([[x, y, z_] for x in z for y in z for z_ in z])
        wt = np.array([x*y*z_ for x in w for y in w for z_ in w])/32**3*1000
        gd = el.tabulate(1, ref)[1:, :, :, 0].transpose(1, 2, 0)[:, order, :]*32
        surface = PeriodicSurfaceDistance(c['surface_vertices'], c['surface_triangles'])
        for start in range(0, len(sel), 16):
            assert time.perf_counter()-tick < cfg['budget_seconds']
            which = sel[start:start+16]
            distance = surface.query(origins[which, None, :]/64 + ref[None, :, :]/32)*10
            accumulate(acc, a['q'], b['q'], ids, H, which, gd, wt,
                       expit((.25-distance)/ell), (distance <= .25).astype(float))
    direct = acc['U_b_q_s']-acc['U_s_q_s']
    release = acc['U_b_q_s']-acc['U_b_q_b']
    net = acc['U_b_q_b']-acc['U_s_q_s']
    scale = acc['U_s_q_s']
    checks = {
        'smooth_energy_reproduced': abs(acc['U_s_q_s']/rs['energy_N_mm']-1) <= cfg['energy_relative_reproduction_tolerance'],
        'binary_energy_reproduced': abs(acc['U_b_q_b']/rb['energy_N_mm']-1) <= cfg['energy_relative_reproduction_tolerance'],
        'addition_removal_balance': abs(direct-acc['fixed_state_inside_addition']-acc['fixed_state_outside_removal'])/scale < 1e-10,
        'nonnegative_relaxation': release/scale >= -cfg['variational_identity_relative_tolerance'],
        'compatible_variational_identity': abs(release-acc['compatible_difference_energy_b'])/scale <= cfg['variational_identity_relative_tolerance'],
        'finite': all(np.isfinite(x) for x in acc.values()),
        'budget': time.perf_counter()-tick < cfg['budget_seconds'],
    }
    result = {'rule': label, 'status': 'ok' if all(checks.values()) else 'not_accepted',
              'smooth_source': smooth_dir, 'binary_source': binary_dir,
              'energies_N_mm': acc, 'checks': checks,
              'relative_to_smooth_equilibrium_energy': {
                  'inside_addition': acc['fixed_state_inside_addition']/scale,
                  'outside_removal': acc['fixed_state_outside_removal']/scale,
                  'fixed_state_net_coefficient_change': direct/scale,
                  'reequilibration_release': release/scale,
                  'net_equilibrium_change': net/scale,
                  'identity_residual': (release-acc['compatible_difference_energy_b'])/scale}}
    results.append(result)
    print(json.dumps(result, allow_nan=False), flush=True)
freeze()
output = {'status': 'ok' if all(r['status'] == 'ok' for r in results) else 'not_accepted',
          'pairs': results, 'new_solve': False, 'production_changed': False,
          'wall_seconds': time.perf_counter()-tick,
          'interpretation': cfg['interpretation'], 'outside_selected_not_densely_checked': True}
(D/'result.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n', encoding='utf-8')
assert output['status'] == 'ok', 'Preserve failed accounting; do not change frozen gates'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8.5, 4.4))
names = ['27 points/cell', 'Selected 12^3; rest 27']
x = np.arange(2); width = .23
for off, key, label, color in [(-1, 'fixed_state_net_coefficient_change', 'Fixed-state weight change', '#487d9f'),
                             (0, 'reequilibration_release', 'Energy released by re-equilibration', '#ba8c42'),
                             (1, 'net_equilibrium_change', 'Net equilibrium change', '#546957')]:
    values = [r['relative_to_smooth_equilibrium_energy'][key]*100 for r in results]
    bars = ax.bar(x+off*width, values, width, label=label, color=color)
    ax.bar_label(bars, labels=[f'{v:+.3f}%' for v in values], padding=4, fontsize=10)
ax.axhline(0, color='#555', linewidth=.8)
ax.set_ylim(-5.2, 4.3)
ax.set_xticks(x, names); ax.set_ylabel('% of same-rule smooth equilibrium energy')
ax.set_title('diverse_04: same-rule smooth → binary energy accounting', fontsize=12)
ax.legend(loc='upper center', bbox_to_anchor=(.5, -.14), ncol=1, frameon=False, fontsize=9)
ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(D/'energy_balance.png', dpi=180, bbox_inches='tight')

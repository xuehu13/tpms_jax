"""Bounded full-chain diagnostic: nine forwards, one scalar adjoint, no inverse run."""
from pathlib import Path
import hashlib
import json
import resource
import sys
import time

ROOT = Path('/home/xuehu/projects/tpms_jax')
sys.path.insert(0, str(ROOT))
import numpy as np
import jax
import jax.numpy as jnp
from voxel_field import VoxelDesign, sample_gyroid

OUT = ROOT / 'validation/near_term_20261003/step3'
OUT.mkdir(exist_ok=False)
mapping = json.loads((OUT.parent / 'step2/summary.json').read_text())
meaning = ('Compatible input route: small-grid derivative correctness only'
           if 64 in mapping['array_route_passed_M'] else
           'Diagnostic only: distinguish correct derivative implementation from insufficient forward representation')
start = time.perf_counter()
N = M = 8
values = sample_gyroid(M, .541062, 40.)
coordinates = (jnp.arange(M, dtype=jnp.float64)+.5)/M
xyz = jnp.stack(jnp.meshgrid(coordinates, coordinates, coordinates, indexing='ij'), axis=-1)
window = values*(1-values)
raw_directions = [window*(1+.4*jnp.cos(2*jnp.pi*xyz[...,0])),
                  window*(.35+jnp.cos(2*jnp.pi*(xyz[...,0]+xyz[...,1])))]
directions = [d/jnp.linalg.norm(d) for d in raw_directions]
steps = (1e-3, 5e-4)
for d in directions:
    assert np.isfinite(np.asarray(d)).all()
    for h in steps:
        for sign in (-1, 1):
            perturbed = np.asarray(values+sign*h*d)
            assert perturbed.min() >= 0 and perturbed.max() <= 1
plan = {'N': N, 'M': M, 'beta_input': 40., 'eta': 1e-4, 'c': .541062,
        'lateral': 'fixed', 'map': 'periodic_trilinear', 'steps': steps,
        'directions': ['phi*(1-phi)*(1+.4*cos(2pi*x))', 'phi*(1-phi)*(.35+cos(2pi*(x+y)))'],
        'direction_normalization': 'unit discrete L2 norm', 'meaning': meaning,
        'relative_derivative_target': 1e-3,
        'near_zero_rule': '|AD| <= 1e-8*max(|K_baseline|,1); use absolute error <=1e-7*max(|K_baseline|,1)',
        'max_forward_solves': 9, 'max_scalar_adjoint_solves': 1}
(OUT / 'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
np.savez(OUT / 'inputs.npz', values=np.asarray(values), direction_1=np.asarray(directions[0]), direction_2=np.asarray(directions[1]))
build_start = time.perf_counter()
design = VoxelDesign(N, M, beta=40., emin_ratio=1e-4)
build_seconds = time.perf_counter()-build_start
forward_start = time.perf_counter()
(base_values, base_w), pullback = jax.vjp(design._outputs, values)
base_values.block_until_ready()
base_forward_seconds = time.perf_counter()-forward_start
adjoint_start = time.perf_counter()
gradient = pullback((jnp.array([1., 0., 0.]), jnp.zeros_like(base_w)))[0]
gradient.block_until_ready()
adjoint_seconds = time.perf_counter()-adjoint_start
assert np.isfinite(np.asarray(gradient)).all()
np.save(OUT / 'gradient.npy', np.asarray(gradient))
base = design.forward(values, precomputed=(base_values, base_w))
base.pop('theta')
assert base['status'] == 'ok'
K = float(base_values[0])
assert abs(K-abs(base['Fz_top'])/.01)/abs(K) <= 1e-6
solve_count = 1
rows, forward_checks = [], [{'case': 'baseline', **base}]
for i, direction in enumerate(directions, 1):
    analytic = float(jnp.sum(gradient*direction))
    for h in steps:
        responses, durations = {}, {}
        for sign in (-1, 1):
            perturbed = values+sign*h*direction
            forward_start = time.perf_counter()
            outputs, w = design._outputs(perturbed)
            outputs.block_until_ready()
            durations[sign] = time.perf_counter()-forward_start
            solve_count += 1
            check = design.forward(perturbed, precomputed=(outputs, w))
            check.pop('theta')
            assert check['status'] == 'ok', check
            responses[sign] = float(outputs[0])
            forward_checks.append({'case': f'd{i}_h{h}_sign{sign}', **check})
        fd = (responses[1]-responses[-1])/(2*h)
        absolute = abs(fd-analytic)
        near_zero = abs(analytic) <= 1e-8*max(abs(K), 1.)
        relative = None if near_zero else absolute/abs(analytic)
        passed = absolute <= 1e-7*max(abs(K), 1.) if near_zero else relative <= 1e-3
        row = {'direction': i, 'h': h, 'AD': analytic, 'central_FD': fd,
               'absolute_error': absolute, 'relative_error': relative, 'near_zero': near_zero,
               'K_minus': responses[-1], 'K_plus': responses[1],
               'forward_seconds_minus': durations[-1], 'forward_seconds_plus': durations[1], 'passed': passed}
        rows.append(row)
        print(json.dumps(row), flush=True)
        (OUT / 'forward_checks.json').write_text(json.dumps(forward_checks, indent=2)+'\n')
        (OUT / 'directional_differences.json').write_text(json.dumps(rows, indent=2)+'\n')
assert solve_count == 9
summary = {'stage': 3, 'meaning': meaning, 'N': N, 'M': M, 'lateral': 'fixed',
           'all_derivative_criteria_passed': all(r['passed'] for r in rows),
           'all_forward_checks_passed': all(r['status'] == 'ok' for r in forward_checks),
           'forward_solves_used': solve_count, 'scalar_adjoint_solves_used': 1,
           'build_seconds': build_seconds, 'baseline_forward_seconds_including_compilation': base_forward_seconds,
           'adjoint_seconds_including_first_trace': adjoint_seconds,
           'total_seconds': time.perf_counter()-start,
           'peak_host_rss_MiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
           'jax_version': jax.__version__, 'devices': [str(d) for d in jax.devices()],
           'baseline': base, 'gradient_L2_norm': float(jnp.linalg.norm(gradient)),
           'input_min': float(values.min()), 'input_max': float(values.max()), 'comparisons': rows,
           'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                             for name in ('voxel_field.py', 'design_fem.py', 'density_fem.py', 'fem.py', 'pbc.py', 'geometry.py')},
           'claim_limits': ['No high-resolution adjoint cost', 'No physical accuracy certified at M8',
                            'No generated geometry or neural-network chain tested', 'No free-lateral or binary shape derivative']}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary), flush=True)
if not summary['all_derivative_criteria_passed']:
    raise SystemExit('Directional check failed; retain evidence without extra automatic solves')

"""Run the small predeclared fixed-lateral gradient check; save failures too."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import jax
import jax_fem.solver as installed_solver
import jax_fem.problem as installed_problem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from design_fem import GyroidDesign, OUTPUT_NAMES, SOLVER_OPTIONS, ADJOINT_OPTIONS
from scripts.m4_numerical_study import evaluate_case


def hashes():
    names = ('design_fem.py', 'scripts/check_design_gradients.py', 'geometry.py',
             'density_fem.py', 'fem.py', 'pbc.py', 'scripts/m4_numerical_study.py')
    return {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in names}


def close(actual, expected, atol, rtol):
    return np.abs(actual-expected) <= atol+rtol*np.abs(expected)


def run_case(plan, spec, result):
    a, m = plan['acceptance'], plan['model']
    theta = np.asarray(spec['theta'], dtype=float)
    model = GyroidDesign(spec['N'], m['beta'], m['emin_ratio'], m['eps_z'])
    uniform = np.array([theta[0], 0., 0., 0.])
    result['uniform_forward'] = model.forward(uniform)
    reference = evaluate_case(spec['N'], m['beta'], m['emin_ratio'], 'fixed',
                              solver_options=SOLVER_OPTIONS, c=float(theta[0]))
    result['uniform_m4_reference'] = {k:v for k,v in reference.items() if not k.startswith('_')}
    if reference['status'] != 'ok':
        raise ValueError('M4 reference failed physical acceptance')
    result['uniform_parity_pass'] = all(bool(close(result['uniform_forward'][k], reference[k], a['uniform_parity_atol'], a['uniform_parity_rtol'])) for k in ('Fz_top','U_internal'))
    if not result['uniform_parity_pass'] or result['uniform_forward']['status'] != 'ok':
        raise ValueError('Design adapter differs from existing M4 forward problem')
    base = model.forward(theta)
    result['baseline'] = base
    if base['status'] != 'ok':
        raise ValueError('Baseline failed physical acceptance')
    derivatives = model.derivatives(theta)
    result['ad'] = {k:v.tolist() for k,v in derivatives.items()}
    jac = derivatives['jacobian']
    result['ad_values_match_baseline'] = bool(np.all(close(derivatives['values'], np.asarray(base['values']), 1e-10, 1e-8)))
    result['energy_envelope_pass'] = bool(np.all(close(jac[0], derivatives['energy_envelope_gradient'], a['energy_envelope_atol'], a['energy_envelope_rtol'])))
    result['nonstationary_gradient_norm'] = float(np.linalg.norm(jac[2]))
    if not np.isfinite(jac).all() or not result['ad_values_match_baseline'] or not result['energy_envelope_pass'] or result['nonstationary_gradient_norm'] < a['nonstationary_gradient_min_norm']:
        raise ValueError('AD baseline/envelope/nonstationary check failed')
    result['directions'] = []
    for d in spec['directions']:
        d = np.asarray(d, dtype=float)
        if plan['normalize_directions']:
            d = d/np.linalg.norm(d)
        ad = jac @ d
        entry = {'direction':d.tolist(), 'ad_directional':ad.tolist(), 'steps':[]}
        result['directions'].append(entry)
        for h in plan['central_difference_steps']:
            plus, minus = model.forward(theta+h*d), model.forward(theta-h*d)
            step = {'h':h, 'plus':plus, 'minus':minus}
            entry['steps'].append(step)
            if plus['status'] != 'ok' or minus['status'] != 'ok':
                raise ValueError('Perturbed forward solve failed physical acceptance')
            fd = (np.asarray(plus['values'])-np.asarray(minus['values']))/(2*h)
            error = np.abs(fd-ad)
            limits = a['derivative_atol']+a['derivative_rtol']*np.abs(ad)
            step.update(fd=fd.tolist(), abs_error=error.tolist(), acceptance_limit=limits.tolist(),
                        pass_each_output=(error <= limits).tolist())
            print(f"N={spec['N']} {spec['name']} h={h:.1e} max_scaled_error={np.max(error/limits):.6g}", flush=True)
        entry['last_two_steps_pass'] = all(all(r['pass_each_output']) for r in entry['steps'][-2:])
    result['status'] = 'ok' if all(e['last_two_steps_pass'] for e in result['directions']) else 'gradient_failed'
    result['forward_perturbation_count'] = sum(2*len(d['steps']) for d in result['directions'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding='utf-8-sig'))
    if plan['solver_options'] != SOLVER_OPTIONS or plan['adjoint_solver_options'] != ADJOINT_OPTIONS:
        parser.error('Solver settings differ from the predeclared plan')
    if any(plan['model'][k] != v for k,v in {'E_s':10.,'nu':.3,'cell_size':1.,'periodic_axes':[0,1],'lateral':'fixed','interpolation_power':1}.items()):
        parser.error('This runner only supports the declared fixed-lateral unit cell')
    specs = [dict(plan['uniform_c_check'], name='uniform_N4')]
    specs += [dict(N=n, theta=plan['theta'], directions=plan['directions'], name=f'periodic_N{n}') for n in plan['grids']]
    if any((args.out_dir/(s['name']+'.json')).exists() for s in specs):
        parser.error('Result exists; choose a fresh output directory')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        start = time.perf_counter()
        result = {'case':spec, 'outputs':OUTPUT_NAMES, 'status':'running', 'source_sha256':hashes()}
        try:
            run_case(plan, spec, result)
        except Exception as exc:
            result['status'] = f'FAILED: {type(exc).__name__}: {exc}'
            traceback.print_exc()
        result['elapsed_seconds'] = time.perf_counter()-start
        result['runtime'] = {'jax_version':jax.__version__, 'devices':[str(d) for d in jax.devices()]}
        result['installed_source_sha256'] = {name:hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() for name,module in [('jax_fem.solver',installed_solver),('jax_fem.problem',installed_problem)]}
        if result['source_sha256'] != hashes():
            result['status'] = 'FAILED: source changed during calculation'
        with (args.out_dir/(spec['name']+'.json')).open('x') as stream:
            stream.write(json.dumps(result,indent=2,allow_nan=result['status'] != 'ok')+'\n')
        print(spec['name']+' status='+result['status'],flush=True)
        if result['status'] != 'ok':
            return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())

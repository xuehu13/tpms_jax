"""Run only the three frozen missing cases; retain logs and stop on a failed case."""
from pathlib import Path
import csv
import json
import os
import subprocess
import sys
import time

ROOT = Path('/home/xuehu/projects/tpms_jax')
OUT = ROOT / 'validation/near_term_20261003/step1'
PLAN = json.loads((OUT.parent / 'plan.json').read_text())

def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

def read(path):
    return json.loads(path.read_text())

existing = read(ROOT / 'validation/projection_effects_20261002/N64_beta40_emin4_fixed_c.json')
rows = [{'method': 'JAX_analytic', 'N': 64, 'lateral': 'fixed', 'beta': 40., 'eta': 1e-4,
         'c': .541062, 'Fz': existing['Fz_top'], 'K': abs(existing['Fz_top'])/.01,
         'Vf': existing['vf_int'], 'volume_definition': 'projected_phi_Gauss_JxW',
         'source': 'validation/projection_effects_20261002/N64_beta40_emin4_fixed_c.json', 'status': 'reused_ok'}]
for n in (24, 32, 48):
    for lateral in ('fixed', 'relaxed_free'):
        source = f'validation/abaqus_binary/binary_gyroid_G{n}_R0_C3D10_{lateral}.acceptance.json'
        data = read(ROOT / source)
        m = data['measured']
        rows.append({'method': 'Abaqus_binary_C3D10', 'N': n, 'lateral': lateral,
                     'beta': None, 'eta': None, 'c': .541062, 'Fz': m['macro_RF'][2],
                     'K': abs(m['macro_RF'][2])/.01, 'Vf': m['volume_solid'],
                     'volume_definition': 'mesh_solid_IVOL', 'source': source, 'status': data['status']})
with (OUT / 'definition_comparison.csv').open('x', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

records = []
for case in PLAN['step1_new_cases']:
    name = f'N{case["N"]}_{case["lateral"]}'
    output, log = OUT / f'{name}.json', OUT / f'{name}.console.txt'
    assert not output.exists() and not log.exists(), 'Never overwrite or silently repeat a solve'
    command = [sys.executable, str(ROOT / 'scripts/capture_binary_projection_reference.py'),
               '--N', str(case['N']), '--beta', '40', '--emin-ratio', '.0001',
               '--c', '.541062', '--lateral', case['lateral'], '--solver', 'petsc', '--out', str(output)]
    print(json.dumps({'starting': name, 'command': command}), flush=True)
    start = time.perf_counter()
    env = dict(os.environ)
    env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    with log.open('x') as stream:
        proc = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    text = log.read_text(errors='replace')
    record = {'case': name, 'return_code': proc.returncode, 'wall_seconds_including_startup_and_diagnostics': time.perf_counter()-start,
              'log': str(log.relative_to(ROOT)), 'result': str(output.relative_to(ROOT)),
              'macro_equilibrium_solves_observed': text.count('Solving the nonlinear problem...'),
              'linear_solves_observed': text.count('PETSc Solver - Solving linear system'),
              'process_environment': {k: env.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'XLA_PYTHON_CLIENT_PREALLOCATE')}}
    if output.exists():
        result = read(output)
        record.update(status=result['status'], Fz=result.get('Fz_top'), K=abs(result.get('Fz_top', 0.))/.01,
                      checks=result.get('checks', {}), peak_host_rss_MiB=result.get('runtime', {}).get('peak_host_rss_MiB'))
    else:
        record.update(status='no_result', last_log_lines=text.splitlines()[-12:])
    records.append(record)
    dump(OUT / 'execution.json', records)
    print(json.dumps({'finished': record}), flush=True)
    if proc.returncode != 0 or record['status'] != 'ok' or not all(record['checks'].values()):
        raise SystemExit('Stopped after failed case; evidence retained')
    expected = 1 if case['lateral'] == 'fixed' else 4
    assert record['macro_equilibrium_solves_observed'] == expected, record

print(json.dumps({'three_missing_cases_completed': True, 'new_macro_equilibrium_solves': sum(r['macro_equilibrium_solves_observed'] for r in records)}), flush=True)

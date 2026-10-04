"""Only the frozen M64 then conditional M32 comparisons; never calibrate inputs."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path('/home/xuehu/projects/tpms_jax')
OUT = ROOT / 'validation/near_term_20261003/step2'
assert json.loads((OUT.parent / 'step1/summary.json').read_text())['proceed_to_step2']
analytic_path = ROOT / 'validation/projection_effects_20261002/N64_beta40_emin4_fixed_c.json'
binary_path = ROOT / 'validation/abaqus_binary/binary_gyroid_G48_R0_C3D10_fixed.acceptance.json'
analytic = json.loads(analytic_path.read_text())
binary = json.loads(binary_path.read_text())['measured']['macro_RF'][2]
records = []
for M in (64, 32):
    name = f'M{M}_N64_fixed'
    output, log = OUT / f'{name}.json', OUT / f'{name}.console.txt'
    assert not output.exists() and not log.exists(), 'Never overwrite or silently repeat a solve'
    command = [sys.executable, str(ROOT / 'scripts/capture_binary_projection_reference.py'),
               '--N', '64', '--beta', '40', '--emin-ratio', '.0001', '--c', '.541062',
               '--lateral', 'fixed', '--solver', 'petsc', '--voxel-M', str(M), '--out', str(output)]
    print(json.dumps({'starting': name, 'command': command}), flush=True)
    start = time.perf_counter()
    env = dict(os.environ)
    env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    with log.open('x') as stream:
        proc = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    text = log.read_text(errors='replace')
    record = {'M': M, 'N': 64, 'return_code': proc.returncode,
              'wall_seconds': time.perf_counter()-start,
              'result': str(output.relative_to(ROOT)), 'log': str(log.relative_to(ROOT)),
              'macro_equilibrium_solves_observed': text.count('Solving the nonlinear problem...'),
              'linear_solves_observed': text.count('PETSc Solver - Solving linear system')}
    if output.exists():
        row = json.loads(output.read_text())
        mapping_difference = abs(row['Fz_top']-analytic['Fz_top'])/abs(analytic['Fz_top'])
        binary_difference = abs(row['Fz_top']-binary)/abs(binary)
        record.update(status=row['status'], Fz=row['Fz_top'], K=abs(row['Fz_top'])/.01,
                      Vf=row['vf_int'], input_representation=row['input_representation'],
                      checks=row['checks'], mapping_relative_response_difference=mapping_difference,
                      binary_relative_response_difference=binary_difference,
                      peak_host_rss_MiB=row['runtime']['peak_host_rss_MiB'])
        record['criteria'] = {'numerical_consistency': row['status'] == 'ok' and all(row['checks'].values()),
                              'mapping_response_difference<=1percent': mapping_difference <= .01,
                              'binary_response_difference<=2percent': binary_difference <= .02}
        record['passed'] = all(record['criteria'].values()) and proc.returncode == 0
    else:
        record.update(status='no_result', passed=False, last_log_lines=text.splitlines()[-12:])
    records.append(record)
    summary = {'stage': 2, 'cases': records, 'new_high_resolution_cases': len(records),
               'no_volume_calibration': True, 'no_extra_projection': True,
               'analytic_reference': str(analytic_path.relative_to(ROOT)),
               'analytic_reference_sha256': hashlib.sha256(analytic_path.read_bytes()).hexdigest(),
               'binary_reference': str(binary_path.relative_to(ROOT)),
               'binary_reference_sha256': hashlib.sha256(binary_path.read_bytes()).hexdigest(),
               'array_route_passed_M': [r['M'] for r in records if r['passed']],
               'M32_skipped_if_M64_failed': not records[0]['passed'],
               'array_route_feasibility_claim': 'Only the recorded representation and physical case',
               'new_Abaqus_jobs': 0}
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'finished': record}), flush=True)
    if record['macro_equilibrium_solves_observed'] != 1 or record['linear_solves_observed'] != 1:
        raise SystemExit('Unexpected solve count; stop and inspect saved evidence')
    if not record['passed']:
        print('Mapping gate failed: stop here, no M32, no automatic tuning or higher resolution.', flush=True)
        break

print(json.dumps({'step2_completed': True, 'new_cases': len(records), 'passed_M': summary['array_route_passed_M']}), flush=True)

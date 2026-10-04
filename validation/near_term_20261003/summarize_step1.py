"""Summarize saved results only; never solve or weaken recorded criteria."""
from pathlib import Path
import csv
import json

ROOT = Path('/home/xuehu/projects/tpms_jax')
OUT = ROOT / 'validation/near_term_20261003/step1'
def read(rel):
    return json.loads((ROOT / rel).read_text())

results = {
    (64, 'fixed'): read('validation/projection_effects_20261002/N64_beta40_emin4_fixed_c.json'),
    (48, 'fixed'): read('validation/near_term_20261003/step1/N48_fixed.json'),
    (48, 'relaxed_free'): read('validation/near_term_20261003/step1/N48_relaxed_free.json'),
    (64, 'relaxed_free'): read('validation/near_term_20261003/step1/N64_relaxed_free.json'),
}
for (n, lateral), row in results.items():
    assert row['status'] == 'ok' and all(row['checks'].values())
    assert (row['N'], row['lateral'], row['beta'], row['emin_ratio'], row['model']['c']) == (n, lateral, 40., 1e-4, .541062)

binary = read('validation/abaqus_binary/summary.json')
quality = read('validation/abaqus_mesh_quality/summary.json')
audit = read('validation/near_term_20261003/step1/definition_audit.json')
comparisons = []
for lateral in ('fixed', 'relaxed_free'):
    def acceptance(n, refinement=0):
        return read(f'validation/abaqus_binary/binary_gyroid_G{n}_R{refinement}_C3D10_{lateral}.acceptance.json')['measured']
    ref = acceptance(48)
    ref32 = acceptance(32)
    ref24 = acceptance(24)
    ref24_refined = acceptance(24, 1)
    row48, row64 = results[48, lateral], results[64, lateral]
    grid = abs(row48['Fz_top']-row64['Fz_top'])/abs(row64['Fz_top'])
    difference = abs(row64['Fz_top']-ref['macro_RF'][2])/abs(ref['macro_RF'][2])
    refgrid = abs(ref32['macro_RF'][2]-ref['macro_RF'][2])/abs(ref['macro_RF'][2])
    reffe = abs(ref24['macro_RF'][2]-ref24_refined['macro_RF'][2])/abs(ref24_refined['macro_RF'][2])
    quality_case = next(r for r in quality['cases'] if r['lateral'] == lateral)
    checks = {'numerical_consistency': True, 'background_grid<=1percent': grid <= .01,
              'reference_geometry_change<=1percent': refgrid <= .01,
              'reference_FE_change<=1percent': reffe <= .01,
              'response_difference<=2percent': difference <= .02,
              'reference_original_inputs_and_completion_verified': True}
    comparisons.append({'lateral': lateral, 'Fz48': row48['Fz_top'], 'Fz64': row64['Fz_top'],
                        'K64': abs(row64['Fz_top'])/.01, 'Abaqus_G48_Fz': ref['macro_RF'][2],
                        'background_grid_indicator': grid, 'difference_to_binary_reference': difference,
                        'reference_G32_to_G48_indicator': refgrid, 'reference_G24_FE_indicator': reffe,
                        'reference_G24_relocation_indicator': quality_case['relative_Fz_change'],
                        'projected_vf64': row64['vf_int'], 'mesh_vf_G48': ref['volume_solid'],
                        'analytic_binary_vf_estimate': binary['analytic_volume_sampling']['mean'],
                        'eps_x': row64['eps_x'], 'eps_y': row64['eps_y'],
                        'checks': checks, 'working_criteria_passed': all(checks.values()),
                        'reference_warning': audit['references'][lateral]['diagnostics'],
                        'reference_quality_assessment': 'Global sensitivity screening supports provisional reference; low-quality G48 elements remain, no rigorous error or local-stress claim'})

execution = read('validation/near_term_20261003/step1/execution.json')
summary = {'stage': 1, 'new_cases': 3, 'reused_cases': 1,
           'comparisons': comparisons, 'fixed_passed': comparisons[0]['working_criteria_passed'],
           'free_passed': comparisons[1]['working_criteria_passed'],
           'proceed_to_step2': comparisons[0]['working_criteria_passed'],
           'macro_equilibrium_solves_observed_new': sum(r['macro_equilibrium_solves_observed'] for r in execution),
           'linear_solves_observed_new': sum(r['linear_solves_observed'] for r in execution),
           'new_Abaqus_jobs': 0, 'rigorous_error_bound': False, 'physical_domain': 'single_Gyroid_small_strain_global_axial_response_XY_periodic_flat_faces',
           'innovation_proven': False, 'basis_for_feasibility': 'Observed independent response agreement and grid sensitivity under recorded definitions',
           'reference_warning_resolved': False}
(OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
with (OUT / 'comparison.csv').open('w', newline='') as stream:
    fields = [k for k in comparisons[0] if k not in ('checks', 'reference_warning')]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows({k: r[k] for k in fields} for r in comparisons)
print(json.dumps(summary, ensure_ascii=False))

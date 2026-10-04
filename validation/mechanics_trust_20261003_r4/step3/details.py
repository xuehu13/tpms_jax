from pathlib import Path
import json,re
P=Path('/home/xuehu/projects/tpms_jax');O=P/'validation/mechanics_trust_20261003_r4/step3'
A=Path('/mnt/e/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus/mechanics_trust_20261004_r4/step3_uniform')
def load(f):return json.loads(f.read_text(encoding='utf-8-sig'))
result={}
for tag in ('N2_d010','N4_d010','N4_d005'):
    v=load(O/(tag+'.json'));rows=v['rows'];text=(O/(tag+'.log')).read_text()
    result[tag]={'max_reduced_residual':max(r['reduced_residual_l2'] for r in rows),
        'max_balance':max(abs(r['Fz_top']+r['Fz_bottom']) for r in rows),
        'max_reaction_piola_difference':max(abs(r['Fz_top']-r['mean_first_piola'][2][2]) for r in rows),
        'max_affine_displacement_error':max(r['max_affine_displacement_error'] for r in rows),
        'max_analytic_force_difference':max(abs(r['Fz_top']-r['analytic']['Fz']) for r in rows),
        'max_analytic_energy_difference':max(abs(r['energy']-r['analytic']['energy']) for r in rows),
        'seeded_log_evidence':[line for line in text.splitlines() if 'Newton iterations' in line or '[TIMER]' in line][-25:],
        'half_Fu_relative_error_at_twenty_percent':abs(-rows[-1]['Fz_top']*.2/2-rows[-1]['energy'])/rows[-1]['energy']}
for tag in ('N4_d010','N4_d005'):
    case='uniform_finite_'+tag
    v=load(A/tag/'work'/(case+'.acceptance.json'));last=v['rows'][-1]
    result['Abaqus_'+tag]={'diagnostics':load(A/tag/'work'/(case+'.diagnostics.json')),
        'datacheck_diagnostics':load(A/tag/'work'/(case+'_check.diagnostics.json')),
        'ALLWK_final':last['ALLWK'],'ALLSE_final':last['ALLSE'],'SENER_IVOL_energy_final':last['IP_energy'],
        'max_nodal_error':max(r.get('max_affine_displacement_error',0) for r in v['rows']),
        'max_logarithmic_J_error':max(r.get('logarithmic_J_error',0) for r in v['rows'])}
(O/'details.json').write_text(json.dumps(result,indent=2)+'\n')
(O/'details.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(result,indent=2))

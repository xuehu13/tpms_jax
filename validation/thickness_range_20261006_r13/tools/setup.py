"""Prepare two matched thickness cases; no new FEM or Abaqus geometry."""
from pathlib import Path
import hashlib, json, shutil, subprocess

R = Path('/home/xuehu/projects/tpms_jax')
O = R/'validation/thickness_range_20261006_r13'
native = Path('/mnt/e/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus')
old = native/'large_compression_20261005_r6_explicit_T0p040'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, data):
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n')

resume = O.exists()
if resume:
    assert (O/'input.json').exists() and (O/'before_manifest.json').exists()
    assert not (O/'preparation.json').exists()
    before = json.loads((O/'before_manifest.json').read_text())
    assert all(sha(R/s)==h for s,h in before.items())
else:
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=R).strip()
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R).decode().strip() == 'dd66775a58fe329ae9f0d11ee2b333d08362093f'
O.mkdir(exist_ok=resume)
tracked = subprocess.check_output(['git','ls-files','-z'],cwd=R).decode().split('\0')
if not resume:
    write(O/'before_manifest.json', {s:sha(R/s) for s in tracked if s})
frozen_input = {
    'research_question':'Does the frozen candidate retain its response agreement at adjacent physical wall thicknesses?',
    'main_plan_step':3, 'new_thicknesses_mm':[.45,.55], 'reuse_thickness_mm':.5,
    'baseline_commit':'dd66775a58fe329ae9f0d11ee2b333d08362093f',
    'fixed':{'midsurface':'diverse_28','L_mm':10.,'XYZ_periodic':True,'lateral_macro_strain':0.,
             'solid':'compressible Neo-Hookean','E_MPa':10.,'nu':.3,'eta':1e-4,
             'interface_10_90_mm':.05,'JAX':'HEX27 N32 / 27 Gauss / HRZ / existing explicit',
             'candidate':'objective_void with frozen occupancy-dependent C2 continuation',
             'phi_bounds':[.001,.01],'maximum_continuation_J':.1,
             'JAX_load_seconds':.004,'shell_load_seconds':.040,'hold_fraction':.1},
    'gates':{'hold_force_difference':.1,'curve_RMS_over_fixed_Standard_peak':.1,
             'fixed_curve_denominator_N':2.2589142322540283,'input_work_difference':.1,
             'JAX_energy_work_gap':.01,'terminal_KE_over_U':.05,'loading_fraction_KE_below_5pct':.95,
             'shell_global_energy_drift':.01,'required_actual_J':'strictly positive for phi>=.01',
             'final_material_probe_points_per_cell':[27,125]},
    'notes':['Thickness changes occupancy AND HRZ mass and shell section simultaneously.',
             'Different loading times are the established quasistatic choices, not identical dynamic histories.',
             'No full AD, training, contact, plasticity, mesh or numerical parameter scan.',
             'Thickness range is not a local derivative certificate; shell is a conditional reference.'],
    'reused_baseline':'validation/void_continuation_20261006_r12',
    'solver_source_sha256':{p:sha(R/p) for p in ['hyperelastic_fem.py','scripts/thin_target_explicit.py','surface_distance.py','pbc.py','fem.py','pixi.lock']},
    'Gauss_cache_sha256':sha(R/'validation/large_compression_20261005_r6/quadratic_candidate/gauss_field.npz'),
    'original_shell_deck_sha256':sha(old/'thin_shell.inp'),
    'original_shell_input_sha256':sha(old/'input.json')}
if resume:
    assert json.loads((O/'input.json').read_text()) == frozen_input
else:
    write(O/'input.json', frozen_input)

deck = (old/'thin_shell.inp').read_bytes()
eol = b'\r\n' if b'\r\n' in deck else b'\n'
needle = eol.join([b'*Shell Section, elset=WALL, material=BASE',b'0.5, 5',b''])
assert deck.count(needle)==1
assert b'*Include' not in deck
records = {}
for tag,t in [('t0p45',.45),('t0p55',.55)]:
    case = O/tag
    case.mkdir()
    dest = native/f'thickness_range_20261006_r13_{tag}_explicit_T0p040'
    assert not dest.exists()
    dest.mkdir()
    replacement = eol.join([b'*Shell Section, elset=WALL, material=BASE',f'{t:.2f}, 5'.encode(),b''])
    changed = deck.replace(needle,replacement)
    assert changed.replace(replacement,needle)==deck
    (dest/'thin_shell.inp').write_bytes(changed)
    cfg = json.loads((old/'input.json').read_text())
    cfg.update(case_id=f'diverse28_{tag}',thickness_mm=t,research_round=13,
               role='Matched physical thickness range, frozen objective_void+C2 JAX candidate',
               source_case=str(case),shell_inp_sha256=sha(dest/'thin_shell.inp'))
    for k in ['reuse_previous_state','source_sha256','input_sha256']:
        cfg.pop(k,None)
    cfg['N'] = 32
    cfg['N_note'] = 'JAX background cells per axis; shell uses the unchanged 17986-element S3R midsurface mesh.'
    cfg['provenance'] = {'source_shell_input':str(old/'input.json'),'source_shell_input_sha256':sha(old/'input.json'),
                         'source_deck_sha256':sha(old/'thin_shell.inp'),
                         'only_deck_change':'physical shell section thickness',
                         'extractor_sha256':sha(R/'scripts/extract_thin_explicit.py')}
    write(dest/'input.json',cfg)
    shutil.copy2(R/'scripts/extract_thin_explicit.py',dest/'extract_thin_explicit.py')
    records[tag]={'native_job_directory':str(dest),'input_sha256':sha(dest/'input.json'),
                  'deck_sha256':sha(dest/'thin_shell.inp'),'thickness_mm':t,
                  'only_deck_change':'0.5, 5 -> '+f'{t:.2f}, 5',
                  'source_mesh_unchanged':True,'PBC_unchanged':True,'material_unchanged':True}
write(O/'preparation.json',records)
print(json.dumps(records,indent=2))

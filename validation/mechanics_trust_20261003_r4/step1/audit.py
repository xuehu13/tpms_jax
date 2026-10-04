"""One-off round4 step1 audit. Reads old evidence; does not solve or edit FEM."""
from pathlib import Path
import ast
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

P = Path('/home/xuehu/projects/tpms_jax')
W = Path(__file__).resolve().parent
O = P / 'validation/mechanics_trust_20261003_r4/step1'
A = Path('/mnt/e/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus')
sys.dont_write_bytecode = True
sys.path.insert(0, str(P))
from binary_gyroid import audit_linear


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(p):
    return json.loads(p.read_text())


def dump(p, value):
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def run_git(*args):
    return subprocess.check_output(['git', '-C', str(P), *args], text=True).strip()


def inp_audit(path, points, cells, ex):
    """Read actual INP and test constraints against the stated physical field."""
    section = ''
    physical_nodes = 0
    element_count = 0
    equations, bcs, elastic = [], [], []
    terms = 0
    keywords = []
    for raw in path.open():
        line = raw.strip()
        if not line or line.startswith('**'):
            continue
        if line.startswith('*'):
            section = line.upper()
            keywords.append(section)
            continue
        tokens = [s.strip() for s in line.split(',') if s.strip()]
        if section.startswith('*NODE, NSET=PHYSICAL'):
            label = int(tokens[0])
            assert label == physical_nodes + 1
            assert np.array_equal(np.array(tokens[1:], float), points[label-1])
            physical_nodes += 1
        elif section.startswith('*ELEMENT, TYPE='):
            assert 'TYPE=C3D10' in section and 'ELSET=SOLID' in section
            label = int(tokens[0])
            assert label == element_count + 1
            assert np.array_equal(np.array(tokens[1:], int)-1, cells[label-1])
            element_count += 1
        elif section == '*ELASTIC':
            elastic.append([float(x) for x in tokens])
        elif section == '*EQUATION':
            if len(tokens) == 1:
                terms = int(tokens[0])
            else:
                assert len(tokens) == 3*terms
                equations.append(tuple((int(tokens[i]), int(tokens[i+1]), float(tokens[i+2]))
                                       for i in range(0, len(tokens), 3)))
        elif section == '*BOUNDARY':
            label, first, last = map(int, tokens[:3])
            assert first == last
            bcs.append((label, first, float(tokens[3])))
    assert physical_nodes == len(points) and element_count == len(cells)
    assert elastic == [[10., .3]]
    assert '*STEP, NAME=COMPRESSION, NLGEOM=NO' in keywords
    assert '*STATIC' in keywords
    assert any(k.startswith('*SOLID SECTION,') and 'MATERIAL=SOLID_MATERIAL' in k for k in keywords)
    assert not any(k.startswith(('*CLOAD', '*DLOAD', '*CONTACT', '*SHELL')) for k in keywords)
    controls = ex['controls']
    top = set(np.flatnonzero(points[:, 2] == 1)+1)
    bottom = set(np.flatnonzero(points[:, 2] == 0)+1)
    anchor = int(np.flatnonzero(np.all(points == 0, axis=1))[0])+1
    wanted_bcs = {(int(i), 3, 0.) for i in bottom}
    wanted_bcs.update({(anchor, 1, 0.), (anchor, 2, 0.), (controls[2], 3, -.01)})
    if ex['lateral'] == 'fixed':
        wanted_bcs.update({(controls[0], 1, 0.), (controls[1], 2, 0.)})
    assert len(bcs) == len(set(bcs)) and set(bcs) == wanted_bcs
    required = {(i, 3) for i in top}
    slaves = np.flatnonzero((points[:, 0] == 1) | (points[:, 1] == 1)) + 1
    for i in slaves:
        required.update((int(i), comp) for comp in (1, 2, 3)
                        if not (comp == 3 and points[i-1, 2] in (0, 1)))
    eliminated = []
    reference_dofs = set()
    for eq in equations:
        i, comp, coefficient = eq[0]
        assert coefficient == 1.
        eliminated.append((i, comp))
        reference_dofs.update(t[:2] for t in eq[1:])
        if len(eq) == 2 and eq[1][0] == controls[2]:
            assert i in top and comp == 3 and eq[1] == (controls[2], 3, -1.)
        else:
            master, mc, coefficient = eq[1]
            assert mc == comp and coefficient == -1.
            jump = points[i-1] - points[master-1]
            target = np.array([int(points[i-1, 0] == 1), int(points[i-1, 1] == 1), 0])
            assert np.any(target) and np.max(np.abs(jump-target)) < 1e-12
            if comp < 3 and target[comp-1]:
                assert eq[2:] == ((controls[comp-1], comp, -1.),)
            else:
                assert len(eq) == 2
    fixed_dofs = {b[:2] for b in bcs}
    assert len(eliminated) == len(set(eliminated))
    assert set(eliminated) == required
    assert not set(eliminated) & (reference_dofs | fixed_dofs)
    assert len(equations) == ex['equations']
    return {'actual_INP_matches_mesh': True, 'material_and_small_strain': True,
            'complete_XY_periodic_and_flat_load_constraints': True,
            'fixed_or_zero_lateral_control_loading': True,
            'no_duplicate_or_reused_eliminated_dofs': True,
            'equations': len(equations), 'boundary_rows': len(bcs)}


start = time.monotonic()
assert not O.exists(), 'Use a new audit directory; never overwrite completed evidence'
O.mkdir(parents=True)
(O/'plan.json').write_bytes((W/'plan.json').read_bytes())
(O/'audit.py').write_bytes(Path(__file__).read_bytes())
rounds = load(W/'plan.json')['frozen_rounds']
frozen = [f for name in rounds for f in (P/'validation'/name).rglob('*') if f.is_file()]
source = list(P.glob('*.py')) + list((P/'scripts').glob('*')) + list((P/'tests').glob('*.py'))
source = [f for f in source if f.is_file()] + [P/'pixi.toml', P/'pixi.lock', P/'results/m4_numerical_study.csv']
before = {str(f): sha(f) for f in sorted(set(frozen+source))}
head = run_git('rev-parse', 'HEAD')
dump(O/'preservation_before.json', {'HEAD': head, 'sha256': before})

package_specs = []
for g, r in ((24, 0), (24, 1), (32, 0), (48, 0)):
    for lateral in ('fixed', 'relaxed_free'):
        package_specs.append(('baseline', A/'binary_gyroid_20261002', g, r, lateral))
for label in ('c048', 'c060'):
    for g in (32, 48):
        package_specs.append((label, A/f'geometry_interface_20261003_r2/{label}/G{g}', g, 0, 'fixed'))
for lateral in ('fixed', 'relaxed_free'):
    package_specs.append(('relocated', A/'mesh_quality_20261002', 24, 0, lateral))
refs, artifacts, mesh_audits = [], {}, {}
for label, pkg, g, r, lateral in package_specs:
    case = f'binary_gyroid_G{g}_R{r}_C3D10_{lateral}'
    ex_path = pkg/(case+'.expected.json')
    ex = load(ex_path)
    ac_path, di_path = pkg/'work'/(case+'.acceptance.json'), pkg/'work'/(case+'.diagnostics.json')
    ac, di = load(ac_path), load(di_path)
    inp, mesh = pkg/(case+'.inp'), pkg/(case+'.mesh.npz')
    assert sha(inp) == ex['input_sha256'] == ac['input_sha256']
    assert sha(mesh) == ex['mesh_sha256']
    assert ex['E_s'] == 10 and ex['nu'] == .3 and ex['gross_volume'] == 1
    assert ex['lateral'] == lateral and ac['status'] == 'ok' and all(ac['checks'].values())
    assert di['errors'] == 0
    sta = pkg/'work'/(case+'.sta')
    assert 'THE ANALYSIS HAS COMPLETED SUCCESSFULLY' in sta.read_text(errors='replace')
    for f in (ex_path, ac_path, di_path, inp, mesh, sta, pkg/'work'/(case+'.dat'), pkg/'work'/(case+'.msg')):
        artifacts[str(f)] = sha(f)
    measured = ac['measured']
    uz, force, energy = measured['macro_displacements'][2], measured['macro_RF'][2], measured['ALLSE']
    k_rf, e_energy = force/uz, 2*energy/(uz**2)
    assert abs(uz+.01) < 1e-8 and abs(k_rf-e_energy)/k_rf < 1e-6
    rec = {'label': label, 'G': g, 'refinement': r, 'lateral': lateral,
           'package': str(pkg), 'c': ex['c'], 'elements': ex['elements'],
           'K_reaction': k_rf, 'K_energy': e_energy, 'mesh_volume': measured['volume_solid'],
           'distorted_fraction': di['distorted_elements']/ex['elements'],
           'minimum_corner_mapping_detJ': ex['geometry']['min_detJ'],
           'periodic_error': measured['max_periodic_error'], 'checks_passed': True}
    # Inspect actual G48 inputs, not just their prepared manifests. Reuse fixed mesh for relaxed.
    if g == 48:
        with np.load(mesh) as saved:
            points, cells = saved['points'], saved['cells']
        rec['actual_input_audit'] = inp_audit(inp, points, cells, ex)
        mesh_key = ex['mesh_sha256']
        if mesh_key not in mesh_audits:
            edges = np.array(((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)))
            error = 0.
            for offset in range(0, len(cells), 16384):
                xyz = points[cells[offset:offset+16384]]
                error = max(error, float(np.max(np.abs(xyz[:, 4:]-(xyz[:, edges[:, 0]]+xyz[:, edges[:, 1]])/2))))
            assert error < 1e-12
            vertices = np.unique(cells[:, :4])
            topology = audit_linear(points[vertices], np.searchsorted(vertices, cells[:, :4]), ex['c'])
            assert topology['periodic_connected_components'] == 1 and topology['min_detJ'] > 0
            assert abs(topology['volume']-measured['volume_solid'])/measured['volume_solid'] < 1e-6
            mesh_audits[mesh_key] = {'topology': topology, 'midside_midpoint_error': error,
                                    'mapping': 'straight-sided C3D10: constant positive geometric Jacobian, within floating precision'}
            print('mesh audited: '+label+' G48', flush=True)
    refs.append(rec)
    print('reference verified: '+label+' '+case, flush=True)

def reference(label, g=48, lateral='fixed', r=0):
    return next(x for x in refs if (x['label'], x['G'], x['lateral'], x['refinement']) == (label, g, lateral, r))


comparisons = []
inputs = [
    ('c048', 'fixed', 'geometry_interface_20261003_r2/step2/c048_N64_analytic.json', 'geometry_interface_20261003_r2/step2/c048_N48_analytic.json'),
    ('baseline', 'fixed', 'projection_effects_20261002/N64_beta40_emin4_fixed_c.json', 'near_term_20261003/step1/N48_fixed.json'),
    ('c060', 'fixed', 'geometry_interface_20261003_r2/step2/c060_N64_analytic.json', 'geometry_interface_20261003_r2/step2/c060_N48_analytic.json'),
    ('baseline', 'relaxed_free', 'near_term_20261003/step1/N64_relaxed_free.json', 'near_term_20261003/step1/N48_relaxed_free.json')]
for label, lateral, n64, n48 in inputs:
    b = load(P/'validation'/n64)
    b48 = load(P/'validation'/n48)
    assert b['status'] == b48['status'] == 'ok' and all(b['checks'].values()) and all(b48['checks'].values())
    assert b['calibration'] is None and b['model']['c'] == reference(label, lateral=lateral)['c']
    assert b['beta'] == 40 and b['emin_ratio'] == 1e-4
    assert b['model']['E_s'] == 10 and b['model']['nu'] == .3 and b['model']['eps_z'] == -.01
    assert b['model']['cell_size'] == 1 and b['model']['periodic_axes'] == [0, 1]
    assert b['model']['interpolation_power'] == 1 and b['lateral'] == lateral
    k, k48 = b['Fz_top']/-.01, b48['Fz_top']/-.01
    assert abs(k-2*b['U_internal']/.01**2)/k < 1e-10
    ref, coarse = reference(label, lateral=lateral), reference(label, 32, lateral)
    vf_path = 'abaqus_binary/binary_volume_reference.json' if label == 'baseline' else f'geometry_interface_20261003_r2/step2/{label}.binary_volume.json'
    sample = load(P/'validation'/vf_path)
    comparisons.append({'label': label, 'c': b['model']['c'], 'lateral': lateral,
                        'background_record': str(P/'validation'/n64), 'K_background': k,
                        'K_reference': ref['K_reaction'],
                        'difference': abs(k-ref['K_reaction'])/ref['K_reaction'],
                        'background_grid_indicator': abs(k48-k)/k,
                        'reference_G32_G48_indicator': abs(coarse['K_reaction']-ref['K_reaction'])/ref['K_reaction'],
                        'vf_smooth': b['vf_int'], 'vf_binary_same_gauss': b['vf_binary_ref'],
                        'vf_fitted_mesh': ref['mesh_volume'], 'analytic_binary_volume_sampling': sample})
quality = {lat: load(P/f'validation/abaqus_mesh_quality/baseline_binary_gyroid_G48_R0_C3D10_{lat}.quality.json')
           for lat in ('fixed', 'relaxed_free')}
quality_compact = {lat: {k: q[k] for k in ('Abaqus_distorted', 'low_quality', 'local_stress_convergence_claim')}
                   for lat, q in quality.items()}
sensitivities = {}
for lat in ('fixed', 'relaxed_free'):
    original, refined, moved = reference('baseline', 24, lat), reference('baseline', 24, lat, 1), reference('relocated', 24, lat)
    sensitivities[lat] = {'fixed_geometry_FE_refinement': abs(original['K_reaction']-refined['K_reaction'])/refined['K_reaction'],
                         'G24_quality_relocation': abs(original['K_reaction']-moved['K_reaction'])/original['K_reaction'],
                         'quality_scope': 'G24 sensitivity, not a direct G48 repaired-mesh bound'}

after_changed = [f for f, h in before.items() if sha(Path(f)) != h]
artifact_changed = [f for f, h in artifacts.items() if sha(Path(f)) != h]
assert not after_changed and not artifact_changed and run_git('rev-parse', 'HEAD') == head
assert all(x['difference'] <= .02 and x['background_grid_indicator'] <= .01 and x['reference_G32_G48_indicator'] <= .01 for x in comparisons)
assert all(s['fixed_geometry_FE_refinement'] <= .01 for s in sensitivities.values())
report = {'stage': 'round4_step1_completed', 'decision': 'qualified_global_reference_usable_for_step2',
          'no_targeted_new_solve_needed': True,
          'reason': 'Original definitions/hashes/completion and actual G48 constraints/topology pass; existing independent response and quality sensitivity sufficient for bounded global-stiffness screening',
          'comparisons': comparisons, 'reference_packages': refs,
          'raw_G48_mesh_audits': mesh_audits, 'quality_energy_volume_evidence': quality_compact,
          'reference_sensitivities': sensitivities,
          'limitations': ['No rigorous true-error bound', 'No local stress certification',
                          'G32-to-G48 varies geometric boundary and FE mesh together',
                          'Quality repair performed only at G24; tiny positive G48 Jacobians remain',
                          'Two off-baseline anchors lack their own fixed-geometry refinement/quality repair',
                          'Independent fitted-solid reference is an approximation of the analytic domain',
                          'No validation of other families, thin walls, large strain or old shell/platen problem'],
          'operations': {'new_FEM_solves': 0, 'new_Abaqus_jobs': 0, 'new_ODB_extractions': 0,
                         'new_training_or_gradient_solves': 0, 'numerical_source_changes': 0,
                         'mesh_and_stored_result_audit': True},
          'elapsed_s': time.monotonic()-start}
dump(O/'audit.json', report)
dump(O/'verification.json', {'passed': True, 'HEAD': head, 'frozen_files': len(frozen),
                            'protected_source_environment_and_CSV_files': len(set(source)),
                            'original_artifacts': len(artifacts), 'source_frozen_or_artifact_changes': [],
                            'artifact_sha256': artifacts, 'audit_sha256': sha(O/'audit.json'),
                            'plan_sha256': sha(O/'plan.json'), 'audit_program_sha256': sha(O/'audit.py')})
print(json.dumps({'decision': report['decision'], 'comparisons': [{k: x[k] for k in ('c', 'lateral', 'difference', 'background_grid_indicator', 'reference_G32_G48_indicator')} for x in comparisons],
                  'source_and_frozen_unchanged': True, 'original_packages': len(refs), 'new_solves': 0}, indent=2), flush=True)

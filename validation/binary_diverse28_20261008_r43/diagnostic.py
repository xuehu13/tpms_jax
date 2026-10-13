"""One frozen diverse_28 occupancy diagnostic, reusing r30/r34 operators."""
from pathlib import Path
import os, sys, time, json, hashlib
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
R = Path('/home/xuehu/projects/tpms_jax')
D = R/'validation/binary_diverse28_20261008_r43'
sys.path.insert(0, str(R))

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def write(p, value): p.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def freeze():
    for name, digest in read(D/'frozen_before.json').items():
        assert sha(R/name) == digest, name
    if (D/'launch_freeze.json').exists():
        for name, digest in read(D/'launch_freeze.json').items():
            assert sha(D/name) == digest, name
def replace(s, old, new, count=1):
    assert s.count(old) == count, (old, s.count(old))
    return s.replace(old, new)
def geometry():
    import numpy as np, basix
    from jax_fem.basis import get_elements
    cfg = read(D/'protocol.json'); c = np.load(R/cfg['cache'])
    family, cell, _, _, degree, order = get_elements('HEX27')
    el = basix.create_element(family, cell, degree)
    origins = 2*np.indices((32,)*3).reshape(3, -1).T
    local = np.rint(el.points[order]*2).astype(int)
    ijk = origins[:, None, :] + local[None, :, :]
    cells = (ijk[:, :, 0]*65 + ijk[:, :, 1])*65 + ijk[:, :, 2]
    return cfg, c, el, order, origins, cells
def gradients(el, order, ref):
    return el.tabulate(1, ref)[1:, :, :, 0].transpose(1, 2, 0)[:, order, :]*32
def unit_energy(q, ids, cells, which, grad, H):
    import numpy as np
    from hyperelastic_fem import MU, KAPPA
    v = H + np.einsum('cni,qnj->cqij', q[ids[cells[which]]], grad, optimize=True)
    e = .5*(v+v.swapaxes(-1, -2)); lam = KAPPA-2*MU/3
    return .5*lam*np.trace(e, axis1=-2, axis2=-1)**2 + MU*np.sum(e*e, axis=(-1, -2))

def prepare():
    import numpy as np
    tick = time.perf_counter(); cfg, c, el, order, origins, cells = geometry()
    st = np.load(R/cfg['smooth_baseline']/'linear_state.npz')
    assert read(R/cfg['smooth_baseline']/'jax_result.json')['status'] == 'ok'
    g = gradients(el, order, c['physical_quad_points'][0]*32)
    uc = np.zeros(32768)
    for pos in range(0, 32768, 256):
        which = np.arange(pos, min(pos+256, 32768))
        u = unit_energy(st['q'], st['class_ids'], cells, which, g, st['H'])
        uc[which] = np.sum((1e-4+(1-1e-4)*c['rho'][which])*u*c['JxW'][which]*1000, axis=1)
    assert abs(uc.sum()/cfg['original_smooth_energy_N_mm']-1) <= cfg['original_energy_relative_reproduction_gate']
    candidate = np.flatnonzero(np.any((c['rho'] >= .01)&(c['rho'] <= .99), axis=1))
    ranked = candidate[np.lexsort((candidate, -uc[candidate]))]
    goal = cfg['selection_candidate_energy_fraction_target']*uc[candidate].sum()
    count = int(np.searchsorted(np.cumsum(uc[ranked]), goal))+1
    selected = np.sort(ranked[:count]); np.save(D/'selected_cells.npy', selected)
    row = {'selection_rule': cfg['selection_rule'], 'selection_sha256': sha(D/'selected_cells.npy'),
           'candidate_cells': len(candidate), 'selected_cells': len(selected),
           'candidate_fraction_full_energy': float(uc[candidate].sum()/uc.sum()),
           'selected_fraction_candidate_energy': float(uc[selected].sum()/uc[candidate].sum()),
           'selected_fraction_full_energy': float(uc[selected].sum()/uc.sum()),
           'original_selected_smooth_energy_N_mm': float(uc[selected].sum()),
           'original_full_smooth_energy_N_mm': float(uc.sum()),
           'outside_selection_densely_checked': False,
           'wall_seconds': time.perf_counter()-tick}
    write(D/'selection.json', row); print(json.dumps(row), flush=True)

def execute(source, out):
    out.mkdir(exist_ok=False)
    write(out/'frozen_before.json', read(D/'frozen_before.json'))
    write(out/'source_adaptation.json', {'driver_sha256': sha(Path(__file__)),
          'compiled_source_sha256': hashlib.sha256(source.encode()).hexdigest(),
          'production_changed': False})
    exec(compile(source, str(out/'adapted_source.py'), 'exec'),
         {'__file__': str(out/'adapted_source.py'), '__name__': '__main__'})

def binary27():
    cfg = read(D/'protocol.json')
    assert sha(D/'selected_cells.npy') == read(D/'selection.json')['selection_sha256']
    source = (R/'validation/initial_tangent_20261008_r30/diagnostic.py').read_text()
    source = replace(source, 'validation/geometry_transfer_20261006_r15/gauss_field.npz', cfg['cache'])
    source = replace(source, "rho=cache['rho']; qp=cache['physical_quad_points']",
                     "rho=(cache['distance']*10<=.25).astype(float); qp=cache['physical_quad_points']")
    source = replace(source, "'case':'diverse_04'", "'case':'diverse_28 binary Gauss occupancy'")
    execute(source, D/'original27')

def check():
    import numpy as np
    from scipy.special import expit
    from surface_distance import PeriodicSurfaceDistance
    tick = time.perf_counter(); cfg, c, el, order, origins, cells = geometry()
    sel = np.load(D/'selected_cells.npy')
    assert sha(D/'selected_cells.npy') == read(D/'selection.json')['selection_sha256']
    states = {'smooth': np.load(R/cfg['smooth_baseline']/'linear_state.npz'),
              'binary': np.load(D/'original27/linear_state.npz')}
    rows = {'smooth': read(R/cfg['smooth_baseline']/'jax_result.json'),
            'binary': read(D/'original27/jax_result.json')}
    assert all(r['status'] == 'ok' for r in rows.values())
    assert np.array_equal(states['smooth']['class_ids'], states['binary']['class_ids'])
    assert np.array_equal(states['smooth']['H'], states['binary']['H'])
    ids, H = states['smooth']['class_ids'], states['smooth']['H']
    eta = cfg['eta']; ell = .05/(2*np.log(9))
    def totals(which, g, weights, distance, phi_s):
        phi_b = (distance*10 <= .25).astype(float)
        out = {}
        for name, st in states.items():
            unit = unit_energy(st['q'], ids, cells, which, g, H)
            out[name+'_state_floor_energy_N_mm'] = float(np.sum(eta*unit*weights))
            for field, phi in [('smooth', phi_s), ('binary', phi_b)]:
                out[name+'_state_'+field+'_occ_energy_N_mm'] = float(np.sum((eta+(1-eta)*phi)*unit*weights))
        for name, phi in [('smooth', phi_s), ('binary', phi_b)]:
            out[name+'_volume_mm3'] = float(np.sum(phi*weights))
            out[name+'_distance_moment_mm5'] = float(np.sum(phi*weights*(distance*10)**2))
        return out
    g0 = gradients(el, order, c['physical_quad_points'][0]*32)
    original = totals(sel, g0, c['JxW'][sel]*1000, c['distance'][sel], c['rho'][sel])
    # Independently reproduce each full original energy before dense comparison.
    full = {name: 0. for name in states}
    for pos in range(0, 32768, 256):
        which = np.arange(pos, min(pos+256, 32768))
        v = totals(which, g0, c['JxW'][which]*1000, c['distance'][which], c['rho'][which])
        for name in full: full[name] += v[name+'_state_'+name+'_occ_energy_N_mm']
    checks = {name+'_original_energy_reproduced': abs(full[name]/rows[name]['energy_N_mm']-1) <= cfg['original_energy_relative_reproduction_gate'] for name in full}
    assert all(checks.values()), checks
    geometry_cache = np.load(R/cfg['surface_geometry_cache'])
    surface = PeriodicSurfaceDistance(geometry_cache['surface_vertices'], geometry_cache['surface_triangles'])
    distance_error = float(np.max(abs(surface.query(c['physical_quad_points'][sel])-c['distance'][sel])))
    checks['original_distance_reproduction_le_1e-12'] = distance_error <= 1e-12
    write(D/'geometry_input_validation.json', {'original_distance_max_abs_normalized':distance_error,
          'source':cfg['surface_geometry_cache'], 'checks':{'distance_reproduction':distance_error<=1e-12}})
    assert checks['original_distance_reproduction_le_1e-12'], 'Archived surface does not reproduce original cache'
    levels = []
    for axis_count in cfg['dense_axis_levels']:
        z, w = np.polynomial.legendre.leggauss(axis_count); z = (z+1)/2; w /= 2
        ref = np.array([[x,y,z_] for x in z for y in z for z_ in z])
        weights = np.array([x*y*z_ for x in w for y in w for z_ in w])/32**3*1000
        g = gradients(el, order, ref); acc = {}; last = time.perf_counter()
        for pos in range(0, len(sel), 32):
            assert time.perf_counter()-tick < cfg['per_major_attempt_budget_seconds'], 'Frozen fixed-field budget exceeded'
            which = sel[pos:pos+32]
            distance = surface.query(origins[which,None,:]/64+ref[None,:,:]/32)
            values = totals(which, g, weights, distance, expit((.25-distance*10)/ell))
            for key, value in values.items(): acc[key] = acc.get(key, 0.)+value
            if time.perf_counter()-last > 10:
                print(json.dumps({'stage':'fixed_field_check','axis_count':axis_count,'cells':pos+len(which),'seconds':time.perf_counter()-tick}),flush=True); last=time.perf_counter()
        levels.append({'axis_count':axis_count, **acc}); print(json.dumps(levels[-1]),flush=True)
    relative = {key: levels[0][key]/levels[1][key]-1 for key in original}
    for key, error in relative.items():
        if 'floor_energy' in key:
            checks[key+'_exact'] = max(abs(levels[0][key]/original[key]-1), abs(levels[1][key]/original[key]-1)) <= cfg['constant_floor_energy_relative_gate']
        else: checks[key+'_8_12_stable'] = abs(error) <= cfg['dense_energy_volume_distance_moment_relative_gate']
    checks['budget'] = time.perf_counter()-tick < cfg['per_major_attempt_budget_seconds']
    result = {'status': 'ok' if all(checks.values()) else 'not_accepted', 'checks':checks,
              'original_selected': original, 'full_original_energies_N_mm': full,
              'levels':levels, 'relative_8_over_12_minus_one':relative,
              'partial_confirm_allowed':all(checks.values()), 'outside_selected_not_checked':True,
              'wall_seconds':time.perf_counter()-tick}
    write(D/'fixed_check.json', result); print(json.dumps(result,indent=2),flush=True)

def mixed(field):
    cfg = read(D/'protocol.json'); fixed = read(D/'fixed_check.json')
    assert fixed['partial_confirm_allowed'], 'Frozen gate failed; no additional integration/solve'
    sel = read(D/'selection.json'); qrdir = D/('fixed_'+field); qrdir.mkdir(exist_ok=False)
    key = field+'_state_'+field+'_occ_energy_N_mm'
    write(qrdir/'result.json', {'status':fixed['status'], 'full_original_energy_N_mm':fixed['full_original_energies_N_mm'][field],
        'selected_original':{'total_energy_N_mm':fixed['original_selected'][key]},
        'levels':[{'total_energy_N_mm':fixed['levels'][-1][key]}]})
    out = D/(field+'_mixed12'); out.mkdir(exist_ok=False)
    write(out/'frozen_before.json', read(D/'frozen_before.json'))
    write(out/'protocol.json', {'shell_stiffness_N_per_mm':cfg['shell_stiffness_N_per_mm'],
        'selected_cells_sha256':sel['selection_sha256'], 'total_budget_seconds':900.,
        'maximum_iterations':6000, 'dense_points_per_axis':12})
    source = (R/'validation/local_reequilibrium_20261008_r34/diagnostic.py').read_text()
    source = replace(source, "Q=R/'validation/local_quadrature_20261008_r33'", "Q=D.parent\nQR=D.parent/'fixed_"+field+"'")
    source = replace(source, "(Q/'result.json')", "(QR/'result.json')", count=2)
    source = replace(source, 'validation/geometry_transfer_20261006_r15/gauss_field.npz', cfg['cache'])
    source = replace(source, "surface=PeriodicSurfaceDistance(c['surface_vertices'],c['surface_triangles'])",
        "geometry_cache=np.load(R/'"+cfg['surface_geometry_cache']+"')\nsurface=PeriodicSurfaceDistance(geometry_cache['surface_vertices'],geometry_cache['surface_triangles'])")
    baseline = cfg['smooth_baseline'] if field == 'smooth' else 'validation/binary_diverse28_20261008_r43/original27'
    source = replace(source, 'validation/initial_tangent_20261008_r30/linear_state.npz', baseline+'/linear_state.npz')
    source = replace(source, 'validation/initial_tangent_20261008_r30/jax_result.json', baseline+'/jax_result.json')
    source = replace(source, "'case':'diverse_04 N32 initial static; only frozen r33 cells integrated densely'", "'case':'diverse_28 "+field+" selected12/rest27 initial static'")
    if field == 'binary':
        source = replace(source, "p=make_density_hyperelastic_problem(N,eta=ETA,rho_quad=c['rho']", "rho0=(c['distance']*L<=.25).astype(float)\np=make_density_hyperelastic_problem(N,eta=ETA,rho_quad=rho0")
        source = replace(source, "rho=expit((.25-surface.query(points)*L)/ELL)", "rho=(surface.query(points)*L<=.25).astype(float)")
        source = replace(source, "c['rho'][which]", "rho0[which]")
    # Reuse all frozen operator and PCG gates; rename only provenance labels.
    source = source.replace('r33_', 'r43_fixed_')
    write(out/'source_adaptation.json', {'compiled_source_sha256':hashlib.sha256(source.encode()).hexdigest(),
          'source':'frozen r34 operator/solver; own r43 selection/cache and fixed-field reconstruction', 'production_changed':False})
    exec(compile(source,str(out/'adapted_source.py'),'exec'),{'__file__':str(out/'adapted_source.py'),'__name__':'__main__'})

if __name__ == '__main__':
    freeze()
    action = sys.argv[1]
    if action == 'prepare': prepare()
    elif action == 'binary27': binary27()
    elif action == 'check': check()
    elif action in ['smooth_mixed12', 'binary_mixed12']: mixed(action.split('_')[0])
    else: raise ValueError(action)
    freeze()

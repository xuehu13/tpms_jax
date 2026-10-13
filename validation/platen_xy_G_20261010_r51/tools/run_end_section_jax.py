"""Bounded local static contact test, using JAX-FEM solver and the shared NH kernel.

Conforming local solid, not the G background/XY model or a new dynamic solver.
"""
from pathlib import Path
import argparse, hashlib, json, os, sys, time
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
R = Path('/home/xuehu/projects/tpms_jax')
sys.path.insert(0, str(R))
import numpy as np
import jax
import jax.numpy as jnp
import basix
from jax_fem.basis import get_elements
from jax_fem.generate_mesh import Mesh
from jax_fem.solver import solver
from hyperelastic_fem import DensityHyperelasticity
from jax_fem import logger
logger.setLevel('WARNING')
jax.config.update('jax_enable_x64', True)
D = R / 'validation/platen_xy_G_20261010_r51'
B = D / 'diagnostics/end_section_benchmark_20261010'

def write(p, value):
    p.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

class EndSection(DensityHyperelasticity):
    def get_surface_maps(self):
        def normal_residual(u, x, plane_z, kn):
            penetration = jnp.maximum(x[2] + u[2] - plane_z, 0.)
            return jnp.array([0., 0., kn * penetration])
        return [normal_residual]

def build(cfg, case):
    nx, ny, nz = cfg['nx_ny_nz']
    levels = np.array([2 * nx + 1, 2 * ny + 1, 2 * nz + 1])
    indices = np.indices(levels).reshape(3, -1).T
    H = cfg['height_mm']; slope = cfg['midsurface_dx_dz']
    width = cfg['normal_thickness_mm'] * np.sqrt(1 + slope ** 2)
    z = indices[:, 2] / (2 * nz) * H
    points = np.c_[5 + slope * (z - H / 2) + width * (indices[:, 0] / (2 * nx) - .5), indices[:, 1] / (2 * ny) * cfg['width_mm'], z]
    family, cell, _, _, degree, order = get_elements('HEX27')
    local = np.rint(2 * basix.create_element(family, cell, degree).points[order]).astype(int)
    origins = 2 * np.indices((nx, ny, nz)).reshape(3, -1).T
    locations = origins[:, None, :] + local[None, :, :]
    cells = (locations[..., 0] * levels[1] + locations[..., 1]) * levels[2] + locations[..., 2]
    bottom = lambda x: jnp.isclose(x[2], 0., atol=1e-10)
    top = lambda x: jnp.isclose(x[2], H, atol=1e-10)
    all_nodes = lambda x: jnp.array(True)
    zero = lambda x: 0.
    p = EndSection(Mesh(points, cells), vec=3, dim=3, ele_type='HEX27', quadrature_order=4,
        dirichlet_bc_info=[[bottom, bottom, all_nodes], [0, 2, 1], [zero, zero, zero]], location_fns=[top])
    p.material_model = 'objective_void'
    p.element_degree = 2
    p.set_params(jnp.zeros((3, 3)), np.ones((p.fe.num_cells, p.fe.num_quads)), cfg['eta'])
    face_cells = np.asarray(p.boundary_inds_list[0])[:, 0]
    if case == 'nodal':
        # Use exactly the 23 C3D20R top-node positions. Duplicate element-edge
        # occurrences share the same total area, so no double-counted springs.
        top_local = np.flatnonzero(local[:, 2] == 2)
        ids = cells[face_cells][:, top_local]
        old = np.load(B / 'input/solid_mesh.npz')
        abq_top_indices = old['indices'][old['top'] - 1]
        enabled = set(map(tuple, abq_top_indices))
        active = np.array([[tuple(indices[n]) in enabled for n in face] for face in ids])
        unique, counts = np.unique(ids[active], return_counts=True)
        assert len(unique) == 23
        occurrences = dict(zip(unique.tolist(), counts.tolist()))
        area = width * cfg['width_mm']
        weights = np.array([[area / 23 / occurrences[n] if ok else 0. for n, ok in zip(face, flags)] for face, flags in zip(ids, active)])
        shapes = np.zeros((len(face_cells), 9, 27))
        for q, n in enumerate(top_local): shapes[:, q, n] = 1.
        p.selected_face_shape_vals[0] = jnp.asarray(shapes)
        p.physical_surface_quad_points[0] = jnp.asarray(points[ids])
        p.nanson_scale[0] = jnp.asarray(weights[:, None, :])
    else:
        assert case == 'gauss'
    p.internal_vars_surfaces = [[jnp.full(np.shape(p.nanson_scale[0])[::2], H), jnp.full(np.shape(p.nanson_scale[0])[::2], cfg['kn_MPa_per_mm'])]]
    # x-z surface quadrature weights are reference-area weights, as in G.
    area = np.asarray(p.nanson_scale[0])[:, 0, :]
    assert np.isclose(area.sum(), width * cfg['width_mm'], atol=1e-12)
    fixed = np.zeros((len(points), 3), dtype=bool)
    for nodes, comp in zip(p.fe.node_inds_list, p.fe.vec_inds_list): fixed[nodes, comp] = True
    return p, points, indices, cells, face_cells, fixed

def observe(p, points, indices, cells, face_cells, fixed, u, closure, elapsed, iterations):
    shapes = jnp.asarray(p.selected_face_shape_vals[0])
    uface = jnp.einsum('fqn,fni->fqi', shapes, u[cells[face_cells]])
    xface = jnp.asarray(p.physical_surface_quad_points[0])
    weights = jnp.asarray(p.nanson_scale[0])[:, 0, :]
    plane = 10. - closure
    pen = jnp.maximum(xface[..., 2] + uface[..., 2] - plane, 0.)
    kn = p.internal_vars_surfaces[0][1]
    loads = kn * weights * pen
    force = float(jnp.sum(loads))
    F = jnp.eye(3) + p.fe.sol_to_grad(u)
    J = np.asarray(jnp.linalg.det(F))
    W = jax.vmap(p.material_energy)(F.reshape(-1, 3, 3), p.stiffness_scale.ravel()).reshape(p.stiffness_scale.shape)
    energy = float(jnp.sum(W * p.fe.JxW))
    residual = np.asarray(p.compute_residual([u])[0])
    bottom = np.isclose(points[:, 2], 0.)
    base_force = float(residual[bottom, 2].sum())
    free_res = float(np.linalg.norm(residual[~fixed]))
    top = np.flatnonzero((indices[:, 2] == 80) & (indices[:, 1] == 0))
    left = top[np.argmin(indices[top, 0])]; right = top[np.argmax(indices[top, 0])]
    mid = top[np.argmin(abs(indices[top, 0] - 4))]
    current = points + np.asarray(u)
    section = current[right] - current[left]
    lever = float(jnp.sum(loads * (xface[..., 0] + uface[..., 0] - current[mid, 0])) / force) if force > 1e-20 else 0.
    top_all = indices[:, 2] == 80
    row = {'closure_mm': closure, 'compression': closure / 10., 'force_N': force,
        'bottom_force_N': base_force, 'free_residual_l2_N': free_res,
        'J_min': float(J.min()), 'internal_energy_N_mm': energy,
        'contact_energy_N_mm': float(.5 * jnp.sum(loads * pen)),
        'top_cross_section_rotation_rad': float(np.arctan2(section[2], section[0])),
        'tip_mid_U1_mm': float(u[mid, 0]), 'tip_mid_U3_mm': float(u[mid, 2]),
        'force_lever_from_current_mid_mm': lever,
        'max_enabled_quadrature_penetration_mm': float(jnp.max(jnp.where(weights > 0, pen, 0.))),
        'max_physical_top_node_penetration_mm': float(max(0., np.max(current[top_all, 2] - plane))),
        'wall_seconds': elapsed, 'Newton_corrections': iterations}
    if not np.isfinite(J).all() or J.min() <= 0 or not np.isfinite(energy) or free_res > 2e-9:
        raise ValueError('Invalid physical Jacobian, energy or equilibrium residual')
    return row

def main():
    ap = argparse.ArgumentParser();ap.add_argument('--case', choices=['nodal','gauss'], required=True);a = ap.parse_args()
    cfg = json.loads((B / 'input/jax_protocol.json').read_text())
    out = B / 'jax' / a.case;out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic(); rows = [];states = []; status = 'running'
    write(out / 'input.json', {**cfg, 'case': a.case, 'tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    try:
        p, points, indices, cells, face_cells, fixed = build(cfg, a.case)
        u = jnp.zeros_like(jnp.asarray(points))
        np.savez_compressed(out / 'mesh_contact.npz', xyz_mm=points, cells=cells, indices=indices,
            face_cells=face_cells, shapes=np.asarray(p.selected_face_shape_vals[0]),
            contact_points_mm=np.asarray(p.physical_surface_quad_points[0]), area_weights_mm2=np.asarray(p.nanson_scale[0])[:,0,:])
        rows.append(observe(p, points, indices, cells, face_cells, fixed, u, 0., time.monotonic()-start, 0));states.append(np.asarray(u))
        for closure in np.linspace(0., cfg['target_closure_mm'], cfg['load_increments'] + 1)[1:]:
            if time.monotonic() - start > cfg['wall_limit_s']:raise TimeoutError('Local solve wall budget reached')
            p.internal_vars_surfaces[0][0] = jnp.full_like(p.internal_vars_surfaces[0][0], 10. - closure)
            p.trial_calls = 0
            sol = solver(p, solver_options={'newton': {'initial_guess': [u], 'tol': cfg['residual_tol_N'],
                'rel_tol': 1e-10, 'line_search_flag': True, 'linear': {'spsolve_solver': {}}}})
            trial = sol[0]
            row = observe(p, points, indices, cells, face_cells, fixed, trial, float(closure), time.monotonic()-start, p.trial_calls-1)
            u = trial;rows.append(row);states.append(np.asarray(u))
            write(out/'progress.json', {'accepted_increments':len(rows)-1,'last':row})
            if (len(rows)-1) % 5 == 0:print(json.dumps(row), flush=True)
        status = 'complete_local_static'
    except Exception as error:
        status = type(error).__name__;write(out/'failure.json', {'type':status, 'message':str(error)})
    if states:np.savez_compressed(out/'field.npz',xyz_mm=points,indices=indices,cells=cells,u_mm=states[-1],path_u_mm=np.array(states))
    result = {'case':a.case,'status':status,'complete':status=='complete_local_static','wall_seconds':time.monotonic()-start,
        'path':rows,'scope':'Conforming fully solid local static test; no G, no background occupancy validation, no dynamics/HRZ or design AD.',
        'force_definition':'Sum of all upper platen normal contact forces, compression positive.',
        'new_G_path':False,'full20':False,'new_design_AD':False}
    write(out/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k!='path'},indent=2),flush=True)
    if not result['complete']:raise SystemExit(1)
if __name__=='__main__':main()

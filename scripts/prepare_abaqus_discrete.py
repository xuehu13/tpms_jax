"""Independently integrate M4 Gauss-point material into linear user elements.

For linear small-strain elasticity only. Abaqus imports the local matrices,
assembles them and solves the independently written total-displacement PBCs.
No UEL/USDFLD subroutine or compiler is needed. Native S/E fields are absent.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import norm as sparse_norm
from scipy.special import expit

from abaqus_element_comparison import quadrature, elasticity_matrix
from prepare_uniform_baseline import build_model, validate_model

DIRECT_REFERENCE_OPTIONS = {"newton": {"tol": 1e-12, "rel_tol": 1e-12,
                                      "linear": {"spsolve_solver": {}}}}


def density_numpy(points, c=0.541062, beta=20.0):
    x, y, z = np.moveaxis(2*np.pi*points, -1, 0)
    g = np.sin(x)*np.cos(y)+np.sin(y)*np.cos(z)+np.sin(z)*np.cos(x)
    return expit(beta*(g+c))-expit(beta*(g-c))


def linear_matrix_lines(matrix):
    """Upper triangle by COLUMN, four comma-separated entries per line.

    Start a new line for each column. The installed Abaqus 2026 keyword
    reader rejects concatenated F20 fields and accepts comma-separated
    data; preserve full double precision within its 256-character limit.
    """
    if matrix.shape != (24,24) or not np.all(np.isfinite(matrix)):
        raise ValueError("Expected finite 24x24 stiffness")
    if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-14):
        raise ValueError("Expected symmetric stiffness")
    lines = []
    for col in range(24):
        for start in range(0, col+1, 4):
            line = ", ".join(f"{v:.17E}" for v in matrix[start:min(start+4,col+1),col])
            if len(line) > 256:
                raise ValueError("Matrix record exceeds keyword line limit")
            lines.append(line)
    return lines


def prepare(directory, n=4, lateral="fixed", beta=20.0, emin_ratio=1e-3):
    if n < 2 or n**3 >= 10000:
        raise ValueError("Pilot requires 2 <= N and N**3 < 10000 (one type per element)")
    if lateral not in ("fixed", "relaxed_free"):
        raise ValueError("Supported lateral conditions: fixed or relaxed_free")
    if not np.isfinite(beta) or beta <= 0 or not 0 < emin_ratio <= 1:
        raise ValueError("Require beta > 0 and 0 < E_min/E_s <= 1")
    directory.mkdir(parents=True, exist_ok=True)
    model = build_model(n, lateral)
    topology = validate_model(n, model)
    nodes, cells, controls, equations, bcs = model
    points = np.array(list(nodes.values()))
    connectivity = np.array(cells)-1
    D_unit = elasticity_matrix(E=1.0)
    local, rho_values, qp, wq = [], [], [], []
    for cell in connectivity:
        B, weights, positions = quadrature(points[cell])
        rho = density_numpy(positions, beta=beta)
        modulus = 10*(emin_ratio+rho*(1-emin_ratio))
        local.append(np.einsum("qia,ij,qjb,q->ab", B, D_unit, B, weights*modulus))
        rho_values.append(rho); qp.append(positions); wq.append(weights)
    local, rho_values, qp, wq = map(np.array, (local, rho_values, qp, wq))
    cell_dofs = (3*connectivity[:,:,None]+np.arange(3)).reshape(-1,24)
    K_independent = coo_matrix((local.ravel(),
        (np.repeat(cell_dofs,24,axis=1).ravel(), np.tile(cell_dofs,(1,24)).ravel())),
        shape=(3*len(nodes),3*len(nodes))).tocsr()

    # Solve the real M4 path and compare its tangent after explicit node mapping.
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from scripts.m4_numerical_study import solve_case
    from density_fem import avg_stress
    problem, solutions, H, ex, ey, _, _ = solve_case(
        n, beta, emin_ratio, lateral, solver_options=DIRECT_REFERENCE_OPTIONS)
    actual_points = np.asarray(problem.fe.points)
    integer = np.rint(actual_points*n).astype(int)
    actual_to_export = ((integer[:,0]*(n+1)+integer[:,1])*(n+1)+integer[:,2])
    if len(set(actual_to_export.tolist())) != len(nodes) or not np.allclose(actual_points, points[actual_to_export], rtol=0, atol=1e-12):
        raise ValueError("JAX node coordinates cannot be mapped to exported mesh")
    dof_map = (3*actual_to_export[:,None]+np.arange(3)).ravel()
    problem.newton_update([jnp.zeros_like(solutions[0])])
    K_jax = coo_matrix((problem.V,(dof_map[problem.I],dof_map[problem.J])),shape=K_independent.shape).tocsr()
    matrix_error = float(sparse_norm(K_jax-K_independent)/sparse_norm(K_jax))
    true_qp = np.asarray(problem.physical_quad_points)
    density_error = float(np.max(np.abs(density_numpy(true_qp,beta=beta)-np.asarray(problem.rho))))
    if matrix_error > 1e-11 or density_error > 1e-12:
        raise ValueError("Independent material/quadrature assembler differs from M4")
    u = np.empty_like(actual_points)
    u[actual_to_export] = np.asarray(solutions[0])+actual_points @ np.asarray(H).T
    r_jax = np.asarray(problem.compute_residual(solutions)[0])
    top = np.isclose(actual_points[:,2],1)
    Fz = float(r_jax[top,2].sum())
    sigma = np.asarray(avg_stress(problem, solutions))
    energy = float(0.5*u.ravel() @ K_independent @ u.ravel())
    macro_energy = float(0.5*np.sum(sigma*np.asarray(H)))
    reduced_residual = float(np.max(np.abs(problem.P_mat.T @ r_jax.ravel())))
    checks = {"matrix_matches_m4": matrix_error < 1e-11, "density_matches_m4": density_error < 1e-12,
              "m4_reduced_residual": reduced_residual < 1e-8,
              "m4_force_balance": abs(float(r_jax[:,2].sum())) < 1e-8,
              "m4_reaction_stress": abs(Fz-sigma[2,2]) < 1e-8,
              "m4_energy_identity": abs(energy-macro_energy) < 1e-8}
    if not all(checks.values()):
        raise ValueError("M4 reference failed consistency checks")
    job = f"discrete_gyroid_N{n}_"+lateral
    lines = ["*HEADING", "M4 linear Gauss-point material, independently integrated matrices",
             "** E_q=E_min+rho_q*(10-E_min), nu=0.3; full 2x2x2 integration",
             "*NODE, NSET=PHYSICAL"]
    lines += [f"{label}, "+", ".join(f"{v:.17g}" for v in xyz) for label,xyz in nodes.items()]
    for name, label in zip(("QX","QY","QZ"),controls):
        lines += [f"*NODE, NSET={name}", f"{label}, 0., 0., 0."]
    for label, (cell, matrix) in enumerate(zip(cells, local), 1):
        lines += [f"*USER ELEMENT, TYPE=U{label}, NODES=8, LINEAR, COORDINATES=3",
                  "1, 2, 3", "*MATRIX, TYPE=STIFFNESS"]
        lines += linear_matrix_lines(matrix)
        lines += [f"*ELEMENT, TYPE=U{label}, ELSET=SOLID", f"{label}, "+", ".join(map(str,cell))]
    lines += ["*UEL PROPERTY, ELSET=SOLID"]
    for terms in equations:
        lines += ["*EQUATION", str(len(terms)), ", ".join(f"{label}, {comp}, {coef:.17g}" for label,comp,coef in terms)]
    lines += ["*STEP, NAME=COMPRESSION, NLGEOM=NO", "*STATIC", "1., 1.", "*BOUNDARY"]
    lines += [f"{label}, {comp}, {comp}, {value:.17g}" for label,comp,value in bcs]
    lines += ["*OUTPUT, FIELD, FREQUENCY=1", "*NODE OUTPUT", "U, RF", "*END STEP"]
    inp = directory/(job+".inp")
    inp.write_text("\n".join(lines)+"\n", encoding="ascii")
    expected = {"case":job, "N":n, "beta":beta, "c":0.541062, "E_s":10.0,
        "emin_ratio":emin_ratio, "nu":0.3, "lateral":lateral, "topology":topology,
        "relative_acceptance_tolerance":1e-6, "matrix_relative_difference":matrix_error,
        "reference_solver_options":DIRECT_REFERENCE_OPTIONS,
        "density_max_difference":density_error, "reference_checks":{k:bool(v) for k,v in checks.items()},
        "reduced_residual":reduced_residual, "Fz":Fz, "energy":energy,
        "sigma":sigma.tolist(), "eps":[float(ex),float(ey),-0.01],
        "vf_int":float(np.sum(rho_values*wq)/wq.sum()),
        "input_sha256":hashlib.sha256(inp.read_bytes()).hexdigest(),
        "nodes":{str(label):{"xyz":points[label-1].tolist(), "u":u[label-1].tolist()} for label in nodes}}
    (directory/(job+".expected.json")).write_text(json.dumps(expected,indent=2,allow_nan=False)+"\n")
    np.savez_compressed(directory/(job+".quadrature.npz"), coordinates=points, cells=connectivity,
                        points=qp, weights=wq, rho=rho_values, local_stiffness=local)
    print(json.dumps({k:v for k,v in expected.items() if k != "nodes"},indent=2),flush=True)
    return expected


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--lateral", choices=["fixed","relaxed_free"], default="fixed")
    p.add_argument("--beta", type=float, default=20.0)
    p.add_argument("--emin-ratio", type=float, default=1e-3)
    a = p.parse_args()
    prepare(a.output, a.n, a.lateral, a.beta, a.emin_ratio)

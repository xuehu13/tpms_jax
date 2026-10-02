"""Prepare C3D8 matrix jobs; compare actual Abaqus and installed M4 tangents.

Preparation needs only Python/NumPy. Comparison runs in the project's Pixi
environment, with actual Abaqus MATRIX INPUT exports copied into --directory.
No Abaqus jobs are submitted by this script.
"""
import argparse
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import sys

import numpy as np

SIGNS = np.array([[-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
                  [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]], dtype=float)
E, NU = 10.0, 0.3


def geometries():
    regular = (SIGNS + 1.0)/2
    distorted = regular.copy()
    distorted[6] += [0.18, -0.08, 0.12]
    distorted[4] += [-0.04, 0.03, 0.02]
    return {"regular": regular, "distorted": distorted}


def quadrature(xyz):
    """Independent trilinear HEX8 derivatives; engineering shear convention."""
    matrices, weights, positions = [], [], []
    for natural in itertools.product((-1/np.sqrt(3), 1/np.sqrt(3)), repeat=3):
        natural = np.array(natural)
        factors = 1 + SIGNS*natural
        deriv = np.empty((8, 3))
        for direction in range(3):
            other = [d for d in range(3) if d != direction]
            deriv[:, direction] = SIGNS[:, direction]*np.prod(factors[:, other], axis=1)/8
        jacobian = xyz.T @ deriv
        weight = float(np.linalg.det(jacobian))
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("Nonpositive/nonfinite HEX8 quadrature Jacobian")
        gradients = deriv @ np.linalg.inv(jacobian)
        B = np.zeros((6, 24))
        for a, (dx, dy, dz) in enumerate(gradients):
            B[:, 3*a:3*a+3] = [[dx,0,0], [0,dy,0], [0,0,dz],
                                [dy,dx,0], [0,dz,dy], [dz,0,dx]]
        matrices.append(B)
        weights.append(weight)
        positions.append((np.prod(factors, axis=1)/8) @ xyz)
    return np.array(matrices), np.array(weights), np.array(positions)


def elasticity_matrix(E=E, nu=NU):
    lam, mu = E*nu/((1+nu)*(1-2*nu)), E/(2*(1+nu))
    D = np.zeros((6,6))
    D[:3,:3] = lam
    D[np.arange(3),np.arange(3)] += 2*mu
    D[3:,3:] = mu*np.eye(3)
    return D


def independent_matrices(xyz):
    B, weights, positions = quadrature(xyz)
    D = elasticity_matrix()
    mean_volume_row = np.einsum("q,qj->j", weights, B[:,:3,:].sum(axis=1))/weights.sum()
    Bbar = B.copy()
    Bbar[:,:3,:] += (mean_volume_row - B[:,:3,:].sum(axis=1))[:,None,:]/3
    full = np.einsum("qia,ij,qjb,q->ab", B, D, B, weights)
    bbar = np.einsum("qia,ij,qjb,q->ab", Bbar, D, Bbar, weights)
    return full, bbar, weights, positions


def read_matrix(path, labels=range(1,9)):
    """Read symmetric assembled MATRIX INPUT, preserving the explicit labels.

    Both triangular and full symmetric files are accepted. Duplicates and
    inconsistent mirrored entries are rejected rather than summed twice.
    """
    label_index = {int(label): i for i, label in enumerate(labels)}
    entries = {}
    for line_number, raw in enumerate(Path(path).read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("**"):
            continue
        fields = [s.strip() for s in line.split(",")]
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected five MATRIX INPUT columns")
        rn, rd, cn, cd = map(int, fields[:4])
        if rn not in label_index or cn not in label_index or rd not in (1,2,3) or cd not in (1,2,3):
            raise ValueError(f"{path}:{line_number}: unknown node/DOF")
        ij = (3*label_index[rn]+rd-1, 3*label_index[cn]+cd-1)
        if ij in entries:
            raise ValueError(f"{path}:{line_number}: duplicate matrix entry")
        value = float(fields[4].replace("D", "E").replace("d", "e"))
        if not np.isfinite(value):
            raise ValueError(f"{path}:{line_number}: nonfinite matrix entry")
        entries[ij] = value
    if not entries:
        raise ValueError(f"{path}: empty matrix")
    matrix = np.zeros((3*len(label_index),)*2)
    for (i,j), value in entries.items():
        mirror = entries.get((j,i), value)
        if not np.isclose(mirror, value, rtol=1e-12, atol=1e-14):
            raise ValueError(f"{path}: inconsistent symmetric entries")
        matrix[i,j] = matrix[j,i] = value
    return matrix


def prepare(directory):
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"E": E, "nu": NU, "node_order": "Abaqus/meshio canonical HEX8",
                "constraints": "none; complete unconstrained 24x24 tangent", "cases": {}}
    for name, xyz in geometries().items():
        _, _, weights, _ = independent_matrices(xyz)
        job = "c3d8_" + name
        text = ["*HEADING", "Unconstrained C3D8 tangent comparison: " + name,
                "*NODE"]
        text += [f"{a}, " + ", ".join(f"{v:.17g}" for v in point) for a, point in enumerate(xyz, 1)]
        text += ["*ELEMENT, TYPE=C3D8, ELSET=SOLID", "1, 1, 2, 3, 4, 5, 6, 7, 8",
                 "*MATERIAL, NAME=BASE", "*ELASTIC", f"{E:.17g}, {NU:.17g}",
                 "*SOLID SECTION, ELSET=SOLID, MATERIAL=BASE", ",",
                 "*STEP, NAME=STIFFNESS", "*MATRIX GENERATE, STIFFNESS",
                 "*MATRIX OUTPUT, STIFFNESS, FORMAT=MATRIX INPUT", "*END STEP"]
        path = directory/(job+".inp")
        path.write_text("\n".join(text)+"\n", encoding="ascii")
        manifest["cases"][name] = {"job": job, "coordinates": xyz.tolist(),
                                   "volume": float(weights.sum()), "min_detJ": float(weights.min()),
                                   "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (directory/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


def runtime_matrix(xyz):
    # Import the project's real M4 implementation, not a substitute NumPy law.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax_fem.generate_mesh import Mesh
    from density_fem import DensityLinearElasticityPeriodic
    from scipy.sparse import coo_matrix
    problem = DensityLinearElasticityPeriodic(
        Mesh(xyz, np.arange(8)[None,:]), vec=3, dim=3, ele_type="HEX8",
        dirichlet_bc_info=[[],[],[]])
    problem.set_params(jnp.zeros((3,3)), 1.0, E_s=E, E_min=1e-3*E)
    residual = problem.newton_update([jnp.zeros((8,3))])
    matrix = coo_matrix((problem.V, (problem.I, problem.J)), shape=(24,24)).toarray()
    return matrix, np.asarray(problem.physical_quad_points)[0], np.asarray(problem.JxW)[0,0], float(np.linalg.norm(residual[0]))


def relative_norm(observed, reference):
    return float(np.linalg.norm(observed-reference)/np.linalg.norm(reference))


def matrix_quality(matrix, xyz):
    norm = np.linalg.norm(matrix)
    eigenvalues = np.linalg.eigvalsh((matrix+matrix.T)/2)
    modes = []
    centered = xyz-xyz.mean(axis=0)
    for direction in np.eye(3):
        modes += [np.tile(direction, (8,1)).ravel(), np.cross(direction, centered).ravel()]
    residual = max(np.linalg.norm(matrix @ v)/(norm*np.linalg.norm(v)) for v in modes)
    return {"symmetric_relative_error": relative_norm(matrix.T, matrix),
            "rigid_body_relative_residual": float(residual),
            "null_mode_count": int(np.count_nonzero(np.abs(eigenvalues)<1e-10*norm)),
            "positive_mode_count": int(np.count_nonzero(eigenvalues>1e-10*norm)),
            "minimum_eigenvalue": float(eigenvalues[0])}


def compare(directory, output):
    manifest = json.loads((directory/"manifest.json").read_text())
    if manifest["E"] != E or manifest["nu"] != NU:
        raise ValueError("Material parameters differ from the declared comparison")
    report = {"runtime": {key: importlib.metadata.version(key) for key in ("jax", "jax-fem", "numpy")},
              "tolerance": {"matrix": 1e-10, "rigid_body": 1e-10}, "cases": {}}
    arrays = {}
    for name, xyz in geometries().items():
        case = manifest["cases"][name]
        if not np.array_equal(np.array(case["coordinates"]), xyz):
            raise ValueError("Coordinates differ from generated inputs")
        inp = directory/(case["job"]+".inp")
        if hashlib.sha256(inp.read_bytes()).hexdigest() != case["input_sha256"]:
            raise ValueError("Abaqus input changed since preparation")
        path = directory/(case["job"]+"_STIF1.mtx")
        abaqus = read_matrix(path)
        full, bbar, weights, points = independent_matrices(xyz)
        jax_matrix, runtime_points, runtime_weights, zero_residual = runtime_matrix(xyz)
        # Match spatial positions, never assume JAX and Abaqus quadrature numbering.
        distances = np.linalg.norm(points[:,None,:]-runtime_points[None,:,:], axis=2)
        mapping = np.argmin(distances, axis=1)
        matching = len(set(mapping.tolist())) == 8 and float(distances[np.arange(8),mapping].max()) < 1e-12
        matching = matching and np.allclose(weights, runtime_weights[mapping], rtol=1e-12, atol=1e-14)
        affine_modes = []
        for i in range(3):
            for j in range(3):
                H = np.zeros((3,3)); H[i,j] = 1.0
                affine_modes.append((xyz @ H.T).ravel())
        affine_difference = max(np.linalg.norm((abaqus-jax_matrix) @ v)/
                                (np.linalg.norm(jax_matrix)*np.linalg.norm(v)) for v in affine_modes)
        nonaffine = np.zeros((8,3))
        nonaffine[:,0] = (xyz[:,0]-0.5)*(xyz[:,1]-0.5)
        v = nonaffine.ravel()
        energy_jax, energy_abaqus = float(0.5*v @ jax_matrix @ v), float(0.5*v @ abaqus @ v)
        quality = {"jax": matrix_quality(jax_matrix, xyz), "abaqus": matrix_quality(abaqus, xyz)}
        errors = {"jax_vs_full": relative_norm(jax_matrix, full),
                  "abaqus_vs_bbar": relative_norm(abaqus, bbar),
                  "abaqus_vs_jax": relative_norm(abaqus, jax_matrix),
                  "affine_action_difference": float(affine_difference)}
        checks = {"quadrature_matches": bool(matching), "zero_state_residual": zero_residual < 1e-12,
                  "jax_matches_full_integration": errors["jax_vs_full"] < 1e-10,
                  "abaqus_matches_bbar": errors["abaqus_vs_bbar"] < 1e-10,
                  "affine_action_agrees": affine_difference < 1e-10,
                  "formulation_difference_detected": errors["abaqus_vs_jax"] > 1e-3}
        for engine, q in quality.items():
            checks[engine+"_quality"] = q["symmetric_relative_error"] < 1e-10 and q["rigid_body_relative_residual"] < 1e-10 and q["null_mode_count"] == 6 and q["positive_mode_count"] == 18
        report["cases"][name] = {"matrix_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "volume": float(weights.sum()), "min_detJ": float(weights.min()), "errors": errors,
            "quality": quality, "nonaffine_mode": {"definition": "ux=(x-0.5)*(y-0.5), uy=uz=0",
                "jax_energy": energy_jax, "abaqus_energy": energy_abaqus,
                "relative_energy_difference": abs(energy_abaqus-energy_jax)/abs(energy_jax)},
            "checks": {key: bool(value) for key, value in checks.items()}, "checks_pass": bool(all(checks.values())),
            "same_discrete_operator": bool(errors["abaqus_vs_jax"] < 1e-10)}
        arrays.update({name+"_"+key: value for key,value in
                       {"abaqus": abaqus, "jax": jax_matrix, "full": full, "bbar": bbar}.items()})
    report["checks_pass"] = all(case["checks_pass"] for case in report["cases"].values())
    report["interpretation"] = "Homogeneous rho=1 only. Passing checks confirms the formulation difference; it does not certify C3D8 as the same M4 discrete operator."
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    np.savez(output.with_suffix(".matrices.npz"), **arrays)
    print(json.dumps(report, indent=2))
    return report["checks_pass"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--output", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("--directory", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output)
    else:
        sys.exit(0 if compare(args.directory, args.output) else 1)

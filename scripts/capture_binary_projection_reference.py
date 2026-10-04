"""Capture one M4 projection reference without changing the original M4 CSV."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.m4_numerical_study import C_CALIB, E_S, EPS_Z, evaluate_case
from fem import NU
import numpy as np
import jax


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=32)
    parser.add_argument("--beta", type=float, default=20.)
    parser.add_argument("--emin-ratio", type=float, default=.001)
    parser.add_argument("--lateral", choices=("fixed", "relaxed_free"), default="relaxed_free")
    parser.add_argument("--solver", choices=("default", "petsc"), default="default")
    geometry = parser.add_mutually_exclusive_group()
    geometry.add_argument("--c", type=float, default=C_CALIB)
    geometry.add_argument("--target-vf", type=float)
    parser.add_argument("--family", choices=("gyroid", "primitive"), default="gyroid")
    parser.add_argument("--field-out", type=Path, help="Optional uniform-grid total displacement for mode comparison")
    parser.add_argument("--voxel-M", type=int, help="Sample the continuous Gyroid at periodic cell centres")
    parser.add_argument("--field-kind", choices=("occupancy", "implicit"), default="occupancy")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.N < 2 or not all(map(math.isfinite, (args.beta, args.emin_ratio, args.c))) or args.beta <= 0 or args.c < 0 or not 0 < args.emin_ratio < 1:
        parser.error("Expected finite beta>0, c>=0, 0<emin-ratio<1 and N>=2")
    if args.target_vf is not None and (not math.isfinite(args.target_vf) or not 0 < args.target_vf < 1):
        parser.error("Expected finite 0<target-vf<1")
    if args.family == "primitive" and (args.voxel_M is not None or args.target_vf is not None):
        parser.error("Primitive currently supports direct analytic input without volume calibration")
    if args.field_out is not None and args.field_out.exists():
        parser.error("Field output already exists")
    if args.out.exists():
        parser.error("Output already exists; choose a new path")
    solver_options = None
    petsc_options = None
    if args.solver == "petsc":
        from petsc4py import PETSc
        petsc_options = {"ksp_rtol": 1e-11, "ksp_atol": 1e-13, "ksp_max_it": 5000,
                         "ksp_error_if_not_converged": True}
        for key, value in petsc_options.items():
            PETSc.Options()[key] = value
        solver_options = {"petsc_solver": {"ksp_type": "cg", "pc_type": "gamg"}}
    rho_quad = None
    voxel_record = None
    if args.family == "primitive":
        from geometry import primitive, project_field
        def rho_quad(problem):
            problem.projection_c, problem.projection_calibration = args.c, None
            problem.geometry_field = primitive
            return project_field(primitive(problem.physical_quad_points), args.c, args.beta)
    if args.voxel_M is not None:
        if args.voxel_M < 2 or args.target_vf is not None:
            parser.error("voxel-M>=2; no volume calibration on array route")
        from voxel_field import sample_gyroid, sample_implicit, periodic_trilinear
        from geometry import density, project_field
        values = sample_implicit(args.voxel_M) if args.field_kind == "implicit" else sample_gyroid(args.voxel_M, args.c, args.beta)
        input_path = args.out.with_suffix('.input.npy')
        if input_path.exists():
            parser.error("Array input file already exists")
        input_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(input_path, np.asarray(values))
        voxel_record = {"M": args.voxel_M, "axis_order": "xyz", "sampling": "(i+.5)/M",
                        "map": "periodic_trilinear", "post_projection": args.field_kind == "implicit", "field_kind": args.field_kind,
                        "input_min": float(values.min()), "input_max": float(values.max()),
                        "input_file": str(input_path), "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest()}
        def rho_quad(problem):
            problem.projection_c, problem.projection_calibration = args.c, None
            projected = periodic_trilinear(values, problem.physical_quad_points)
            if args.field_kind == "implicit":
                projected = project_field(projected, args.c, args.beta)
            reference = density(problem.physical_quad_points, args.c, args.beta)
            weights = np.asarray(problem.JxW)[:, 0, :]
            error = np.asarray(projected-reference)
            voxel_record.update(weighted_L1_field_error=float(np.sum(np.abs(error)*weights)/weights.sum()),
                                maximum_field_error=float(np.abs(error).max()),
                                projected_volume_difference=float(np.sum(error*weights)/weights.sum()),
                                projected_min=float(projected.min()), projected_max=float(projected.max()),
                                analytic_transition_fraction=float(np.mean((np.asarray(reference)>.05)&(np.asarray(reference)<.95))),
                                mapped_transition_fraction=float(np.mean((np.asarray(projected)>.05)&(np.asarray(projected)<.95))))
            return projected
    row = evaluate_case(args.N, args.beta, args.emin_ratio, args.lateral,
                        solver_options=solver_options, c=args.c, target_vf=args.target_vf,
                        include_problem=True, include_solution=True, rho_quad=rho_quad)
    if "_problem" in row:
        problem = row.pop("_problem")
        sol_list = row.pop("_sol_list")
        points = np.asarray(problem.fe.points)
        w = np.asarray(sol_list[0])
        maximum = 0.0
        pair_count = 0
        for axis in (0, 1):
            others = [d for d in range(3) if d != axis]
            low = np.flatnonzero(np.isclose(points[:, axis], 0., atol=1e-10))
            high = np.flatnonzero(np.isclose(points[:, axis], 1., atol=1e-10))
            low = low[np.lexsort((points[low, others[1]], points[low, others[0]]))]
            high = high[np.lexsort((points[high, others[1]], points[high, others[0]]))]
            if len(low) != len(high) or not np.allclose(points[low][:, others], points[high][:, others], atol=1e-12, rtol=0):
                raise ValueError("Periodic face coordinate pairing failed")
            maximum = max(maximum, float(np.max(np.abs(w[high] - w[low]))))
            pair_count += len(low)
        loaded = np.isclose(points[:, 2], 0., atol=1e-10) | np.isclose(points[:, 2], 1., atol=1e-10)
        row["max_periodic_fluctuation_error"] = maximum
        row["periodic_face_pair_count"] = pair_count
        row["max_loaded_face_fluctuation_error"] = float(np.max(np.abs(w[loaded, 2])))
        row["checks"]["periodic_fluctuation<=1e-10"] = maximum <= 1e-10
        row["checks"]["loaded_face_fluctuation<=1e-10"] = row["max_loaded_face_fluctuation_error"] <= 1e-10
        row["status"] = "ok" if all(row["checks"].values()) else "check_failed"
        row["macro_equilibrium_solves_expected"] = 1 if args.lateral == "fixed" else 4
        if args.field_out is not None:
            indices = np.rint(points*args.N).astype(int)
            grid = np.empty((args.N+1, args.N+1, args.N+1, 3))
            H = np.diag([row["eps_x"], row["eps_y"], EPS_Z])
            grid[indices[:,0], indices[:,1], indices[:,2]] = w + points@H.T
            args.field_out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.field_out, total_u=grid, N=args.N, H=H)
            row["displacement_field"] = {"path": str(args.field_out), "sha256": hashlib.sha256(args.field_out.read_bytes()).hexdigest()}
        del problem, sol_list, points, w
    if row["status"] == "ok" and args.lateral == "relaxed_free":
        row["checks"]["zero_lateral_stress<=1e-08"] = max(abs(row["sigma_xx"]), abs(row["sigma_yy"])) <= 1e-8
        row["status"] = "ok" if all(row["checks"].values()) else "check_failed"
    row["solver"] = "Existing M4 default solver; global quantities only" if solver_options is None else "PETSc CG/GAMG; same FEM operator and checks"
    row["solver_options"] = solver_options
    row["petsc_options"] = petsc_options
    row["calibration"] = row.pop("_projection_calibration", None)
    row["model"] = {"c": row.pop("_projection_c", args.c), "E_s": E_S, "nu": NU, "eps_z": EPS_Z,
                    "cell_size": 1., "periodic_axes": [0, 1], "interpolation_power": 1, "family": args.family}
    row["input_representation"] = {"type": "analytic_Gauss"} if voxel_record is None else {"type": "signed_implicit_array" if args.field_kind == "implicit" else "continuous_voxel_array", **voxel_record}
    row["runtime"] = {"jax_version": jax.__version__, "devices": [str(d) for d in jax.devices()],
                      "peak_host_rss_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    names = ("scripts/capture_binary_projection_reference.py", "scripts/m4_numerical_study.py",
             "fem.py", "density_fem.py", "geometry.py", "pbc.py")
    row["source_sha256"] = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
    if voxel_record is not None:
        row["source_sha256"]["voxel_field.py"] = hashlib.sha256((ROOT/"voxel_field.py").read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        stream.write(json.dumps(row, indent=2, allow_nan=row["status"] != "ok")+"\n")
    print(json.dumps(row, indent=2), flush=True)
    if row["status"] != "ok" or not all(row["checks"].values()):
        raise SystemExit("M4 consistency failed; failed result retained in "+str(args.out))

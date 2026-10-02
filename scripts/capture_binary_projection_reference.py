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
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.N < 2 or not all(map(math.isfinite, (args.beta, args.emin_ratio, args.c))) or args.beta <= 0 or args.c < 0 or not 0 < args.emin_ratio < 1:
        parser.error("Expected finite beta>0, c>=0, 0<emin-ratio<1 and N>=2")
    if args.target_vf is not None and (not math.isfinite(args.target_vf) or not 0 < args.target_vf < 1):
        parser.error("Expected finite 0<target-vf<1")
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
    row = evaluate_case(args.N, args.beta, args.emin_ratio, args.lateral,
                        solver_options=solver_options, c=args.c, target_vf=args.target_vf)
    if row["status"] == "ok" and args.lateral == "relaxed_free":
        row["checks"]["zero_lateral_stress<=1e-08"] = max(abs(row["sigma_xx"]), abs(row["sigma_yy"])) <= 1e-8
        row["status"] = "ok" if all(row["checks"].values()) else "check_failed"
    row["solver"] = "Existing M4 default solver; global quantities only" if solver_options is None else "PETSc CG/GAMG; same FEM operator and checks"
    row["solver_options"] = solver_options
    row["petsc_options"] = petsc_options
    row["calibration"] = row.pop("_projection_calibration", None)
    row["model"] = {"c": row.pop("_projection_c", args.c), "E_s": E_S, "nu": NU, "eps_z": EPS_Z,
                    "cell_size": 1., "periodic_axes": [0, 1], "interpolation_power": 1}
    row["runtime"] = {"jax_version": jax.__version__, "devices": [str(d) for d in jax.devices()],
                      "peak_host_rss_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    names = ("scripts/capture_binary_projection_reference.py", "scripts/m4_numerical_study.py",
             "fem.py", "density_fem.py", "geometry.py", "pbc.py")
    row["source_sha256"] = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        stream.write(json.dumps(row, indent=2, allow_nan=row["status"] != "ok")+"\n")
    print(json.dumps(row, indent=2), flush=True)
    if row["status"] != "ok" or not all(row["checks"].values()):
        raise SystemExit("M4 consistency failed; failed result retained in "+str(args.out))

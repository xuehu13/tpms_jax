"""Capture one M4 projection reference without changing the original M4 CSV."""
import argparse
import hashlib
import json
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
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.N < 2 or args.beta <= 0 or not 0 < args.emin_ratio < 1:
        parser.error("Expected N>=2, beta>0 and 0<emin-ratio<1")
    if args.out.exists():
        parser.error("Output already exists; choose a new path")
    row = evaluate_case(args.N, args.beta, args.emin_ratio, args.lateral)
    if row["status"] == "ok" and args.lateral == "relaxed_free":
        row["checks"]["zero_lateral_stress<=1e-08"] = max(abs(row["sigma_xx"]), abs(row["sigma_yy"])) <= 1e-8
        row["status"] = "ok" if all(row["checks"].values()) else "check_failed"
    row["solver"] = "Existing M4 default solver; global quantities only"
    row["model"] = {"c": C_CALIB, "E_s": E_S, "nu": NU, "eps_z": EPS_Z,
                    "cell_size": 1., "periodic_axes": [0, 1], "interpolation_power": 1}
    row["runtime"] = {"jax_version": jax.__version__, "devices": [str(d) for d in jax.devices()],
                      "peak_host_rss_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    names = ("scripts/capture_binary_projection_reference.py", "scripts/m4_numerical_study.py",
             "fem.py", "density_fem.py", "geometry.py", "pbc.py")
    row["source_sha256"] = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        stream.write(json.dumps(row, indent=2, allow_nan=False)+"\n")
    print(json.dumps(row, indent=2), flush=True)
    if row["status"] != "ok" or not all(row["checks"].values()):
        raise SystemExit("M4 consistency failed; failed result retained in "+str(args.out))

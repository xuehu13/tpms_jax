"""Capture installed runtime and compare preserved M4 CSV evidence."""
import csv
import hashlib
import importlib.metadata as metadata
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax_fem

root = Path(__file__).resolve().parents[1]
out = root / "validation" / "m4_review"
out.mkdir(parents=True, exist_ok=True)
packages = ["jax", "jaxlib", "jax-fem", "numpy", "scipy", "fenics-basix",
            "meshio", "gmsh", "pytest", "petsc4py"]
versions = {}
for name in packages:
    try:
        versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        versions[name] = None
source_root = Path(jax_fem.__file__).parent
hashes = {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
          for name in ["basis.py", "fe.py", "problem.py", "solver.py"]}
runtime = {
    "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    "code_base_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
    "note": "Captured before commit; code_file_sha256 identifies tested modifications.",
    "python": sys.version, "executable": sys.executable,
    "platform": platform.platform(), "packages": versions,
    "devices": [str(d) for d in jax.devices()],
    "jax_fem_source": str(source_root), "jax_fem_source_sha256": hashes,
    "code_file_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                         for name in ["density_fem.py", "scripts/m4_numerical_study.py",
                                      "tests/test_m4_checks.py"]},
    "pixi_lock_sha256": hashlib.sha256((root / "pixi.lock").read_bytes()).hexdigest(),
}
(out / "environment.json").write_text(json.dumps(runtime, indent=2) + "\n")

def read_cases(path):
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    keys = ["study", "N", "beta", "emin_ratio", "lateral"]
    result = {tuple(row[k] for k in keys): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate case keys")
    return result

before = read_cases(out / "original_a71c3b1.csv")
after = read_cases(out / "fixed.csv")
if set(before) != set(after) or len(after) != 10:
    raise ValueError("Expected the same ten cases")
fields = ["vf_int", "vf_binary_ref", "Fz_top", "Fz_bottom", "sigma_xx",
          "sigma_yy", "sigma_zz", "U_internal", "U_macro", "eps_x", "eps_y"]
comparison = []
for key, row in after.items():
    if row["status"] != "ok" or "FAIL" in row["checks"]:
        raise ValueError("Fixed case failed checks: " + str(key))
    for field in fields:
        a, b = float(before[key][field]), float(row[field])
        if not (math.isfinite(a) and math.isfinite(b)):
            raise ValueError("Nonfinite physical output")
        absolute = abs(b - a)
        relative = absolute / abs(a) if a else None
        if absolute > 1e-10 + 1e-6 * abs(a):
            raise ValueError("Physical output drift: " + str(key) + " " + field)
        comparison.append(dict(zip(["study", "N", "beta", "emin_ratio", "lateral"], key),
                               field=field, absolute_difference=absolute,
                               relative_difference=relative))
with (out / "comparison.csv").open("w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(comparison[0]))
    writer.writeheader()
    writer.writerows(comparison)
summary = {
    "cases": len(after), "all_status_ok": True,
    "comparison_gate": "absolute difference <= 1e-10 + 1e-6*abs(original)",
    "max_Fz_relative_difference": max(r["relative_difference"] for r in comparison
                                      if r["field"] == "Fz_top"),
    "max_energy_relative_difference": max(r["relative_difference"] for r in comparison
                                          if r["field"] == "U_internal"),
    "original_max_reduced_residual": max(float(r["red_res"]) for r in before.values()),
    "fixed_max_reduced_residual": max(float(r["red_res"]) for r in after.values()),
    "artifacts_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in out.iterdir() if p.suffix in (".csv", ".txt")},
}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))

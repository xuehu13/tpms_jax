"""Summarize this fixed-layout grid study; reuse earlier data, submit no jobs."""
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def accepted(row):
    if row["status"] != "ok" or not all(row["checks"].values()):
        raise ValueError("Physical acceptance failed")
    if row["lateral"] == "relaxed_free" and max(abs(row["sigma_xx"]), abs(row["sigma_yy"])) > 1e-8:
        raise ValueError("Lateral equilibrium failed")


def verify_sources(row):
    for name, recorded in row.get("source_sha256", {}).items():
        if sha(ROOT/name) == recorded:
            continue
        for revision in ("c13b1c7b228615cf0ddf80eb1aef666ca193c22d", "38f650cb86d5bee684a76b162ef060ef8368a16f"):
            content = subprocess.check_output(["git", "show", revision+":"+name], cwd=ROOT)
            if hashlib.sha256(content).hexdigest() == recorded:
                break
        else:
            raise ValueError("Untraceable source: "+name)


plan = load(OUT/"plan.json")
with (ROOT/"validation/m4_review/fixed.csv").open(newline="") as stream:
    fixed = [r for r in csv.DictReader(stream) if r["N"] == "32" and r["beta"] == "20.0" and r["emin_ratio"] == "0.001"]
if len(fixed) != 1 or fixed[0]["status"] != "ok" or not all(c.endswith(":pass") for c in fixed[0]["checks"].split(";")):
    raise ValueError("Invalid reused fixed reference")
keys = ("Fz_top", "U_internal", "vf_int", "eps_x", "eps_y")
rows = {(32, "fixed"): {k: float(fixed[0][k]) for k in keys}}
rows[(32, "relaxed_free")] = load(ROOT/"validation/abaqus_binary/projection_N32_relaxed_free.json")
accepted(rows[(32, "relaxed_free")]); verify_sources(rows[(32, "relaxed_free")])
for n in (48, 64):
    for lateral in ("fixed", "relaxed_free"):
        row = load(OUT/(f"projection_N{n}_{lateral}.json"))
        accepted(row); verify_sources(row)
        if row["N"] != n or row["lateral"] != lateral or row["beta"] != 20 or row["emin_ratio"] != .001:
            raise ValueError("Compared different projected models")
        if any(row["model"][k] != plan["model"][k] for k in row["model"]):
            raise ValueError("Model definition differs from plan")
        rows[(n, lateral)] = row
cross = load(OUT/"projection_N48_fixed_petsc.json")
accepted(cross); verify_sources(cross)
crosscheck = {k: abs(cross[k]-rows[(48, "fixed")][k])/abs(rows[(48, "fixed")][k]) for k in ("Fz_top", "U_internal")}
if not all(abs(cross[k]-rows[(48, "fixed")][k]) <= 1e-10+1e-6*abs(rows[(48, "fixed")][k]) for k in crosscheck):
    raise ValueError("Linear solver changed response")
binary = load(ROOT/"validation/abaqus_binary/summary.json")
binary_rows = {r["lateral"]: r for r in binary["cases"] if r["geometry_N"] == 48 and r["refinement"] == 0}
indicators, comparison = [], []
for lateral in ("fixed", "relaxed_free"):
    if binary_rows[lateral]["physical_consistency"] != "pass":
        raise ValueError("Binary reference not accepted")
    for a, b in ((32, 48), (48, 64)):
        indicators.append({"lateral": lateral, "from": a, "to": b,
                           "relative_Fz_change": abs(rows[(a, lateral)]["Fz_top"]-rows[(b, lateral)]["Fz_top"])/abs(rows[(b, lateral)]["Fz_top"]),
                           "relative_energy_change": abs(rows[(a, lateral)]["U_internal"]-rows[(b, lateral)]["U_internal"])/rows[(b, lateral)]["U_internal"]})
    ref = binary_rows[lateral]
    for n in (32, 48, 64):
        row = rows[(n, lateral)]
        comparison.append({"N": n, "lateral": lateral, **{k: row[k] for k in keys},
                           "binary_geometry_N": 48, "binary_Fz": ref["Fz"], "binary_volume_fraction": ref["volume"],
                           "projection_excess_relative_to_binary": abs(row["Fz_top"])/abs(ref["Fz"])-1})
last = [r for r in indicators if r["to"] == 64]
if "109 passed" not in (OUT/"full_tests.txt").read_text():
    raise ValueError("Full regression did not pass")
summary = {"new_projected_cases": 4, "new_solver_crosschecks": 1, "new_abaqus_jobs": 0,
           "full_tests_passed": 109,
           "solver_crosscheck_relative_difference": crosscheck, "grid_indicators": indicators,
           "preferred_grid_screening_passed": all(r["relative_Fz_change"] <= plan["predeclared_screening"]["preferred_relative_grid_Fz_change"] and r["relative_energy_change"] <= plan["predeclared_screening"]["preferred_relative_grid_energy_change"] for r in last),
           "maximum_grid_screening_passed": all(r["relative_Fz_change"] <= plan["predeclared_screening"]["maximum_relative_grid_Fz_change"] and r["relative_energy_change"] <= plan["predeclared_screening"]["maximum_relative_grid_energy_change"] for r in last),
           "projection_binary_comparison": comparison, "finite_resolution_model_difference": True,
           "rigorous_error_bound": False, "pure_beta_error_claim": False, "gradient_verified": False}
(OUT/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")

fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
for ax, lateral in zip(axes, ("fixed", "relaxed_free")):
    ax.plot((32, 48, 64), [abs(rows[(n, lateral)]["Fz_top"])/.01 for n in (32, 48, 64)], "o-", label="Projected: beta=20, Emin/Es=0.001")
    ax.axhline(abs(binary_rows[lateral]["Fz"])/.01, color="gray", ls="--", label="Binary C3D10: G48 reference")
    ax.set(xticks=(32, 48, 64), xlabel="Projected HEX8 grid N", ylabel="Apparent axial modulus (consistent units)",
           title="Fixed lateral" if lateral == "fixed" else "Free lateral")
    ax.grid(alpha=.2); ax.legend(fontsize=7)
fig.savefig(OUT/"projection_grid.png", dpi=180)
plt.close(fig)

names = ("scripts/capture_binary_projection_reference.py", "scripts/m4_numerical_study.py",
         "density_fem.py", "fem.py", "geometry.py", "pbc.py", "tests/test_m4_checks.py")
installed = ROOT/".pixi/envs/default/lib/python3.13/site-packages/jax_fem"
inputs = [ROOT/"validation/m4_review/fixed.csv", ROOT/"validation/abaqus_binary/projection_N32_relaxed_free.json",
          ROOT/"validation/abaqus_binary/summary.json"]
artifacts = [p for p in OUT.iterdir() if p.is_file() and p.name not in ("source_manifest.json", "README.md")]
manifest = {"parent_commit": plan["parent_commit"], "N48_capture_commit": "c13b1c7b228615cf0ddf80eb1aef666ca193c22d",
            "current_sources_sha256": {n: sha(ROOT/n) for n in names},
            "installed_jax_fem_sha256": {n: sha(installed/n) for n in ("solver.py", "problem.py", "fe.py")},
            "package_versions": {n: importlib.metadata.version(n) for n in ("jax", "jax-fem", "numpy", "scipy", "petsc4py")},
            "reused_inputs_sha256": {str(p.relative_to(ROOT)): sha(p) for p in inputs},
            "artifacts_sha256": {p.name: sha(p) for p in artifacts}}
(OUT/"source_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
print(json.dumps(summary, indent=2), flush=True)

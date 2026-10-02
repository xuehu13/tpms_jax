"""Reproduce this small fixed-layout parameter study from saved results."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
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
        raise ValueError("Physical acceptance missing")


def verify_sources(row):
    for name, recorded in row.get("source_sha256", {}).items():
        if sha(ROOT/name) == recorded:
            continue
        for revision in ("a8b20e2834d77126907f5db283de92f17f344964", "c13b1c7b228615cf0ddf80eb1aef666ca193c22d"):
            content = subprocess.check_output(["git", "show", revision+":"+name], cwd=ROOT)
            if hashlib.sha256(content).hexdigest() == recorded:
                break
        else:
            raise ValueError("Untraceable calculation source: "+name)


plan = load(OUT/"plan.json")
base = load(ROOT/"validation/projection_grid_20261002/projection_N64_fixed.json")
accepted(base); verify_sources(base)
if base["vf_int"] != plan["target_projected_vf"]:
    raise ValueError("Target was not copied from planned baseline")
cases = {}
for spec in plan["new_cases"]:
    row = load(OUT/(spec["name"]+".json"))
    accepted(row); verify_sources(row)
    if any(row[k] != spec[k] for k in ("N", "beta", "emin_ratio")) or row["lateral"] != "fixed":
        raise ValueError("Case parameters differ from plan")
    for key in ("E_s", "nu", "eps_z", "cell_size", "periodic_axes", "interpolation_power"):
        if row["model"][key] != plan["model"][key]:
            raise ValueError("Compared different model definitions")
    if row["solver_options"] != base["solver_options"] or any(row["petsc_options"][key] != value for key, value in plan["solver"].items() if key != "type"):
        raise ValueError("Solver differs from predeclared settings")
    if spec.get("match_target_vf"):
        if abs(row["vf_int"]-plan["target_projected_vf"]) > 1e-10 or row["calibration"]["target"] != plan["target_projected_vf"]:
            raise ValueError("Volume-matched case did not match")
    elif row["model"]["c"] != plan["model"]["c_fixed"]:
        raise ValueError("Fixed-c case changed geometry definition")
    cases[spec["name"]] = row

def named(beta, emin=3, matched=False, n=64):
    return cases[f"N{n}_beta{beta}_emin{emin}_"+("matched_vf" if matched else "fixed_c")]

fixed = {10: named(10), 20: base, 40: named(40)}
matched = {10: named(10, matched=True), 20: base, 40: named(40, matched=True)}
binary = load(ROOT/"validation/abaqus_binary/summary.json")
reference = next(r for r in binary["cases"] if r["case"] == "binary_gyroid_G48_R0_C3D10_fixed")
if reference["physical_consistency"] != "pass":
    raise ValueError("Binary reference not accepted")
grid40 = {key: abs(named(40, n=48)[key]-fixed[40][key])/abs(fixed[40][key]) for key in ("Fz_top", "U_internal")}
if max(grid40.values()) > plan["grid_screening"]["maximum_Fz_and_energy_relative_change"]:
    raise ValueError("Beta40 resolution exceeds predeclared maximum")
with (ROOT/"validation/m4_review/fixed.csv").open(newline="") as stream:
    old32 = {int(float(r["beta"])): r for r in csv.DictReader(stream) if r["N"] == "32" and r["emin_ratio"] == "0.001"}
span10 = abs(float(old32[10]["Fz_top"])-fixed[10]["Fz_top"])/abs(fixed[10]["Fz_top"])
table = [{"series": series, "beta": b, "c": r["model"]["c"], "vf": r["vf_int"], "Fz": r["Fz_top"],
          "U": r["U_internal"], "excess_to_original_binary_reference": abs(r["Fz_top"])/abs(reference["Fz"])-1}
         for series, entries in (("fixed_c", fixed), ("matched_projected_vf", matched)) for b, r in entries.items()]
floor = [{"beta": b, "Fz_emin3": fixed[b]["Fz_top"], "Fz_emin4": named(b,4)["Fz_top"],
          "relative_reduction_from_emin3": 1-abs(named(b,4)["Fz_top"])/abs(fixed[b]["Fz_top"]),
          "excess_to_binary_at_emin4": abs(named(b,4)["Fz_top"])/abs(reference["Fz"])-1} for b in (20,40)]
interaction = abs(named(40,4)["Fz_top"])-abs(fixed[40]["Fz_top"])-abs(named(20,4)["Fz_top"])+abs(base["Fz_top"])
volume_changes = [{"beta": b, "delta_c": matched[b]["model"]["c"]-fixed[b]["model"]["c"],
                   "relative_Fz_change_same_beta": abs(matched[b]["Fz_top"])/abs(fixed[b]["Fz_top"])-1} for b in (10,40)]
if "117 passed" not in (OUT/"full_tests.txt").read_text():
    raise ValueError("Full regression did not pass")
summary = {"new_fixed_lateral_cases": len(cases), "new_abaqus_jobs": 0, "full_tests_passed": 117,
           "beta40_N48_to_N64_relative_changes": grid40, "beta40_preferred_grid_passed": max(grid40.values()) <= .005,
           "beta10_N32_to_N64_span": span10, "projected_volume_target": plan["target_projected_vf"],
           "projection_series": table, "floor_effects": floor, "volume_matching_effects": volume_changes,
           "beta10_to40_fixed_c_span_relative_to_beta20": (abs(fixed[10]["Fz_top"])-abs(fixed[40]["Fz_top"]))/abs(base["Fz_top"]),
           "beta10_to40_matched_vf_span_relative_to_beta20": (abs(matched[10]["Fz_top"])-abs(matched[40]["Fz_top"]))/abs(base["Fz_top"]),
           "four_corner_interaction_Fz_magnitude": interaction, "interaction_relative_to_base": interaction/abs(base["Fz_top"]),
           "interaction_definition": "R(40,1e-4)-R(40,1e-3)-R(20,1e-4)+R(20,1e-3), R=abs(Fz)",
           "finite_resolution_model_difference": True, "rigorous_error_bound": False,
           "matched_series_has_same_binary_geometry": False, "free_parameter_effects_verified": False, "gradient_verified": False}
(OUT/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")

fig, axes = plt.subplots(1,3,figsize=(14,4),layout="constrained")
for entries,label,marker in ((fixed,"Fixed c","o"),(matched,"Matched projected volume","s")):
    axes[0].plot((10,20,40), [abs(entries[b]["Fz_top"])/.01 for b in (10,20,40)], marker+"-", label=label)
axes[0].axhline(abs(reference["Fz"])/.01,color="gray",ls="--",label="Original binary G48 reference")
axes[0].set(xlabel="Projection beta",ylabel="Apparent axial modulus",title="N64, Emin/Es=0.001, fixed lateral")
axes[0].legend(fontsize=7)
for b in (20,40):
    axes[1].plot((1e-4,1e-3), [abs(named(b,4)["Fz_top"])/.01, abs(fixed[b]["Fz_top"])/.01], "o-", label=f"beta={b}, fixed c")
axes[1].set(xscale="log",xlabel="Emin/Es",ylabel="Apparent axial modulus",title="N64: floor and beta interaction")
axes[1].legend(fontsize=8)
for b in (20,40):
    r48 = load(ROOT/"validation/projection_grid_20261002/projection_N48_fixed.json") if b==20 else named(40,n=48)
    axes[2].plot((32,48,64),[abs(float(old32[b]["Fz_top"]))/ .01,abs(r48["Fz_top"])/.01,abs(fixed[b]["Fz_top"])/.01],"o-",label=f"beta={b}, fixed c")
axes[2].set(xticks=(32,48,64),xlabel="HEX8 grid N",ylabel="Apparent axial modulus",title="Emin/Es=0.001: grid indicators")
axes[2].legend(fontsize=8)
for ax in axes:
    ax.grid(alpha=.2)
fig.savefig(OUT/"projection_effects.png",dpi=180)
plt.close(fig)

source = ("density_fem.py","scripts/m4_numerical_study.py","scripts/capture_binary_projection_reference.py","geometry.py","fem.py","pbc.py","tests/test_projection_volume.py")
inputs = ("validation/m4_review/fixed.csv","validation/projection_grid_20261002/projection_N64_fixed.json","validation/projection_grid_20261002/projection_N48_fixed.json","validation/projection_grid_20261002/source_manifest.json","validation/abaqus_binary/summary.json")
manifest = {"parent_commit":plan["parent_commit"],"current_source_sha256":{p:sha(ROOT/p) for p in source},
            "reused_inputs_sha256":{p:sha(ROOT/p) for p in inputs},
            "artifacts_sha256":{p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name not in ("source_manifest.json","README.md")}}
(OUT/"source_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(json.dumps(summary,indent=2),flush=True)

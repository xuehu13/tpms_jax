"""Archive this two-case sensitivity study and make one scientific figure."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

repo = Path("/home/xuehu/projects/tpms_jax")
old = Path("/mnt/e/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus/binary_gyroid_20261002")
package = Path("/mnt/e/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus/mesh_quality_20261002")
output = repo/"validation/abaqus_mesh_quality"
workspace = Path(sys.argv[1])
def load(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

plan = load(output/"plan.json")
relocation = load(package/"relocation.json")
shutil.copy2(package/"relocation.json", output/"relocation.json")
baseline, relocated, cases = {}, {}, []
for folder, target in (("baseline", baseline), ("relocated", relocated)):
    for path in (package/"diagnostics"/folder).glob("*.quality.json"):
        target[path.stem.replace(".quality", "")] = load(path)
        shutil.copy2(path, output/(folder+"_"+path.name))
for lateral in ("fixed", "relaxed_free"):
    name = "binary_gyroid_G24_R0_C3D10_"+lateral
    original = load(old/"work"/(name+".acceptance.json"))
    current = load(package/"work"/(name+".acceptance.json"))
    expected = load(package/(name+".expected.json"))
    diagnostic = load(package/"work"/(name+".diagnostics.json"))
    if diagnostic["errors"] != 0:
        raise ValueError("Native solver errors")
    if current["status"] != "ok" or not all(current["checks"].values()):
        raise ValueError("Physical acceptance failed")
    if relocated[name]["Abaqus_distorted"]["elements"] != diagnostic["distorted_elements"]:
        raise ValueError("ODB warning set and native log disagree")
    old_m, new_m = original["measured"], current["measured"]
    fz_difference = abs(new_m["macro_RF"][2]-old_m["macro_RF"][2])/abs(old_m["macro_RF"][2])
    energy_difference = abs(new_m["ALLSE"]-old_m["ALLSE"])/old_m["ALLSE"]
    p99_difference = abs(relocated[name]["mises_volume_quantiles"][2]-baseline[name]["mises_volume_quantiles"][2])/baseline[name]["mises_volume_quantiles"][2]
    checks = {"Fz": fz_difference <= plan["predeclared_screening"]["relative_Fz_change_from_original"],
              "ALLSE": energy_difference <= plan["predeclared_screening"]["relative_ALLSE_change_from_original"],
              "mises_p99": p99_difference <= plan["predeclared_screening"]["relative_volume_weighted_mises_p99_change"],
              "quality_min": relocation["after"]["mean_ratio_min"] > relocation["before"]["mean_ratio_min"],
              "quality_p01": relocation["after"]["mean_ratio_p01"] > relocation["before"]["mean_ratio_p01"],
              "physical_consistency": True}
    cases.append({"case": name, "lateral": lateral, "original_Fz": old_m["macro_RF"][2],
                  "relocated_Fz": new_m["macro_RF"][2], "relative_Fz_change": fz_difference,
                  "relative_ALLSE_change": energy_difference, "relative_p99_change": p99_difference,
                  "checks": checks, "status": "ok" if all(checks.values()) else "screening_failed"})
    for suffix in (".acceptance.json", ".diagnostics.json", ".sta", ".msg", ".console.txt", ".extract.txt"):
        shutil.copy2(package/"work"/(name+suffix), output/(name+suffix))
    shutil.copy2(package/(name+".expected.json"), output/(name+".expected.json"))

tests = (output/"full_tests.txt").read_text()
if "108 passed" not in tests:
    raise ValueError("Complete regression not passed")
summary = {"source_commit": plan["source_commit"], "new_native_analyses": 2, "cases": cases,
           "screening_passed": all(c["status"] == "ok" for c in cases), "full_tests": 108,
           "denominator": "original response magnitude",
           "surface_unchanged": relocation["boundary_coordinates_bitwise_equal"],
           "connectivity_unchanged": relocation["connectivity_bitwise_equal"],
           "volume_change": relocation["after"]["volume"]-relocation["before"]["volume"],
           "quality_before": relocation["before"], "quality_after": relocation["after"],
           "local_stress_convergence_claim": False}
if not summary["screening_passed"]:
    raise ValueError("Predeclared screening failed")
(output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout="constrained")
x = np.arange(2); width = .25
for offset, key, label in ((-1, "relative_Fz_change", "Reaction"),
                           (0, "relative_ALLSE_change", "Energy"),
                           (1, "relative_p99_change", "Mises p99")):
    axes[0].bar(x+offset*width, [100*c[key] for c in cases], width, label=label)
axes[0].set(xticks=x, xticklabels=["Fixed lateral", "Free lateral"],
            ylabel="Change from original (%)", title="Same G24 skin: volume relocation")
axes[0].legend(fontsize=8); axes[0].grid(axis="y", alpha=.2)
names = ["binary_gyroid_G24_R0_C3D10_fixed",
         "binary_gyroid_G24_R0_C3D10_fixed",
         "binary_gyroid_G24_R1_C3D10_fixed"]
records = [baseline[names[0]], relocated[names[1]], baseline[names[2]]]
axes[1].plot(range(3), [r["mises_max"] for r in records], "o-", label="IP maximum")
axes[1].plot(range(3), [r["mises_volume_quantiles"][2] for r in records], "s-", label="Volume-weighted p99")
axes[1].set(xticks=range(3), xticklabels=["Original R0", "Relocated R0", "Original R1"],
            ylabel="von Mises (consistent units)", title="Fixed lateral: same G24 geometry")
axes[1].legend(fontsize=8); axes[1].grid(alpha=.2)
sample = baseline["binary_gyroid_G48_R0_C3D10_fixed"]
for field, label, marker in (("stress_peaks", "20 highest-stress elements", "o"),
                              ("worst_quality", "20 lowest-quality elements", "x")):
    coords = np.array([r["centroid"] for r in sample[field]])
    axes[2].scatter(coords[:, 0], coords[:, 2], s=25, marker=marker, label=label)
axes[2].axhline(.1, color="gray", ls=":", lw=1); axes[2].axhline(.9, color="gray", ls=":", lw=1)
axes[2].set(xlim=(-.025,1.025), ylim=(-.025,1.025), xticks=np.linspace(0,1,6),
            yticks=np.linspace(0,1,6), xlabel="Element centroid x", ylabel="Element centroid z",
            title="G48 fixed: loading faces z=0/1")
axes[2].legend(fontsize=7, loc="center"); axes[2].grid(alpha=.2)
fig.savefig(output/"mesh_sensitivity.png", dpi=180)
plt.close(fig)

source_files = ["scripts/abaqus_mesh_quality.py", "validation/abaqus_mesh_quality/prepare.py",
                "validation/abaqus_mesh_quality/summarize.py", "tests/test_abaqus_mesh_quality.py",
                "scripts/prepare_abaqus_binary.py", "scripts/run_abaqus_binary.ps1",
                "scripts/extract_abaqus_binary.py", "scripts/extract_uniform_baseline.py", "binary_gyroid.py"]
manifest = {"parent_commit": plan["source_commit"],
            "source_sha256": {p: sha(repo/p) for p in source_files},
            "raw_package": str(package), "diagnosed_baseline_package": str(old), "raw_files": {}}
if {r["diagnostic_source_sha256"] for r in list(baseline.values())+list(relocated.values())} != {sha(repo/"scripts/abaqus_mesh_quality.py")}:
    raise ValueError("Diagnostics were not produced by archived source")
for case in cases:
    name = case["case"]
    for path in (package/(name+".inp"), package/(name+".mesh.npz"), package/"work"/(name+".dat"),
                 package/"work"/(name+".odb")):
        manifest["raw_files"][str(path)] = {"sha256": sha(path), "bytes": path.stat().st_size}
(output/"source_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
for name in ("summary.json", "mesh_sensitivity.png", "full_tests.txt"):
    shutil.copy2(output/name, workspace/("mesh_quality_"+name))
print(json.dumps(summary, indent=2), flush=True)

"""Capture completed comparison evidence; no solver execution or Git operations."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ALLOWED_WARNING = "THE *ELEMENT OUTPUT OPTION IS NOT SUPPORTED FOR USER ELEMENTS"


def fingerprint(path):
    return {"path":str(path), "bytes":path.stat().st_size,
            "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}


def inspect_job(root, job):
    work = root/"work"
    if "THE ANALYSIS HAS COMPLETED SUCCESSFULLY" not in (work/(job+".sta")).read_text():
        raise ValueError(job+": analysis did not complete")
    messages = []
    for suffix in (".dat", ".msg"):
        path = work/(job+suffix)
        if path.exists():
            messages += path.read_text(errors="replace").splitlines()
    errors = [s for s in messages if "***ERROR" in s]
    warnings = [s.strip() for s in messages if "***WARNING" in s]
    if errors or any(ALLOWED_WARNING not in s for s in warnings):
        raise ValueError(job+": unresolved or suppressed diagnostic messages")
    return {"completed":True,"errors":len(errors),"warnings":len(warnings),
            "warning_types":sorted(set(warnings))}


def capture(repo, element_root, small_root, large_root):
    element_target = repo/"validation"/"abaqus_element"
    discrete_target = repo/"validation"/"abaqus_discrete"
    discrete_target.mkdir(parents=True,exist_ok=True)
    element_runs, discrete_runs = {}, {}
    for job in ("c3d8_regular","c3d8_distorted"):
        record = inspect_job(element_root, job)
        paths = [element_root/(job+".inp")]+[element_root/"work"/(job+s) for s in ("_STIF1.mtx",".sta",".dat",".msg",".console.txt")]
        record["files"] = {p.name:fingerprint(p) for p in paths}
        element_runs[job] = record
    (element_target/"run_manifest.json").write_text(json.dumps(element_runs,indent=2)+"\n")
    for job, root in (("discrete_gyroid_N4_fixed",small_root),
                      ("discrete_gyroid_N16_fixed",large_root),
                      ("discrete_gyroid_N16_relaxed_free",large_root)):
        record = inspect_job(root, job)
        source = root/(job+".expected.json")
        reference = json.loads(source.read_text())
        report_path = root/"work"/(job+".acceptance.json")
        report = json.loads(report_path.read_text())
        if report["status"] != "ok" or not all(report["checks"].values()):
            raise ValueError(job+": failed acceptance")
        if fingerprint(root/(job+".inp"))["sha256"] != reference["input_sha256"]:
            raise ValueError(job+": input hash mismatch")
        paths = [root/(job+s) for s in (".inp",".expected.json",".quadrature.npz")]
        paths += [root/"work"/(job+s) for s in (".odb",".sta",".dat",".msg",".console.txt",".acceptance.json",".acceptance.displacements.json",".extract.txt")]
        record["files"] = {p.name:fingerprint(p) for p in paths}
        for suffix in (".sta",".acceptance.json",".console.txt",".extract.txt"):
            shutil.copy2(root/"work"/(job+suffix),discrete_target)
        summary = {k:v for k,v in reference.items() if k != "nodes"}
        summary["full_reference"] = fingerprint(source)
        (discrete_target/(job+".reference_summary.json")).write_text(json.dumps(summary,indent=2)+"\n")
        record["reference"] = summary
        record["acceptance"] = report
        discrete_runs[job] = record
        if reference["N"] == 4:
            for suffix in (".inp",".expected.json",".quadrature.npz"):
                shutil.copy2(root/(job+suffix),discrete_target)
    (discrete_target/"run_manifest.json").write_text(json.dumps(discrete_runs,indent=2)+"\n")
    job = "discrete_gyroid_N16_relaxed_free"
    initial_path = large_root/(job+".expected.initial.json")
    initial = json.loads(initial_path.read_text())
    final = discrete_runs[job]["reference"]
    final_full = json.loads((large_root/(job+".expected.json")).read_text())
    if initial["input_sha256"] != final["input_sha256"]:
        raise ValueError("Precision diagnostic changed the Abaqus input")
    initial_summary = {k:v for k,v in initial.items() if k != "nodes"}
    (discrete_target/(job+".reference_initial_summary.json")).write_text(json.dumps(initial_summary,indent=2)+"\n")
    initial_report_path = large_root/"work"/(job+".acceptance.initial.json")
    shutil.copy2(initial_report_path,discrete_target)
    initial_report = json.loads(initial_report_path.read_text())
    diagnostic = {"case":job,"same_input_sha256":initial["input_sha256"],
        "initial_reference":fingerprint(initial_path),
        "default_vs_direct_max_displacement_difference":max(abs(initial["nodes"][k]["u"][i]-final_full["nodes"][k]["u"][i]) for k in initial["nodes"] for i in range(3)),
        "initial_reduced_residual":initial["reduced_residual"],
        "direct_reduced_residual":final["reduced_residual"],
        "initial_abaqus_max_displacement_difference":initial_report["measured"]["max_displacement_error"],
        "final_abaqus_max_displacement_difference":discrete_runs[job]["acceptance"]["measured"]["max_displacement_error"],
        "displacement_acceptance_threshold":1e-8,"threshold_changed":False,
        "abaqus_job_rerun":False}
    (discrete_target/"precision_diagnostic.json").write_text(json.dumps(diagnostic,indent=2)+"\n")
    source_paths = [repo/p for p in ("fem.py","density_fem.py","pbc.py","geometry.py","scripts/m4_numerical_study.py","scripts/abaqus_element_comparison.py","scripts/prepare_abaqus_discrete.py","scripts/extract_abaqus_discrete.py")]
    source_paths += [repo/".pixi"/"envs"/"default"/"lib"/"python3.13"/"site-packages"/"jax_fem"/p for p in ("problem.py","solver.py","fe.py")]
    (discrete_target/"source_manifest.json").write_text(json.dumps({str(p):fingerprint(p) for p in source_paths},indent=2)+"\n")
    shutil.copy2(repo/"results"/"abaqus_discrete_full_tests.txt",discrete_target/"full_tests.txt")
    print(json.dumps({job:{"Fz":v["acceptance"]["measured"]["macro_RF"][2],
        "Fz_relative_difference":v["acceptance"]["relative_Fz_difference"],
        "energy_relative_difference":v["acceptance"]["relative_energy_difference"],
        "displacement_max_difference":v["acceptance"]["measured"]["max_displacement_error"],
        "warnings":v["warnings"]} for job,v in discrete_runs.items()},indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("repo","element-root","small-root","large-root"):
        p.add_argument("--"+name,type=Path,required=True)
    a = p.parse_args()
    capture(a.repo,a.element_root,a.small_root,a.large_root)

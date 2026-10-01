"""Read a completed uniform-baseline ODB; never submit a job.

Example: abaqus python scripts/extract_uniform_baseline.py path/to/model.odb
Requires the prepared expected.json; output is a JSON acceptance report.
"""
import argparse
import json
import math
from pathlib import Path


def data(value):
    try:
        return value.dataDouble
    except Exception:
        return value.data


def region(odb, name, kind):
    containers = [odb.rootAssembly] + list(odb.rootAssembly.instances.values())
    for container in containers:
        mapping = getattr(container, kind)
        if name in mapping:
            return mapping[name]
    raise ValueError("Missing ODB set: " + name)


def point_key(value):
    return (value.instance.name, value.elementLabel, value.integrationPoint)


def extract(odb_path, expected_path, out_path):
    from odbAccess import openOdb
    from abaqusConstants import INTEGRATION_POINT
    manifest = json.loads(expected_path.read_text())
    if odb_path.stem not in manifest["cases"]:
        raise ValueError("ODB name must match a prepared case name")
    expected = manifest["cases"][odb_path.stem]
    odb = openOdb(path=str(odb_path.resolve()), readOnly=True)
    try:
        step = odb.steps["COMPRESSION"]
        frame = step.frames[-1]
        if abs(frame.frameValue-1.0) > 1e-8:
            raise ValueError("ODB does not contain the completed load step")
        solid = region(odb, "SOLID", "elementSets")
        stress_values = frame.fieldOutputs["S"].getSubset(
            region=solid, position=INTEGRATION_POINT).values
        weight_values = frame.fieldOutputs["IVOL"].getSubset(
            region=solid, position=INTEGRATION_POINT).values
        weights = {point_key(v): float(data(v)) for v in weight_values}
        stresses = {point_key(v): list(data(v)) for v in stress_values}
        count = 8*manifest["n"]**3
        if len(weights) != count or len(stresses) != count or set(weights) != set(stresses):
            raise ValueError("Unexpected or mismatched integration-point output")
        if any(w <= 0 or not math.isfinite(w) for w in weights.values()):
            raise ValueError("Invalid integration volume")
        volume = sum(weights.values())
        sigma = [sum(stresses[k][i]*weights[k] for k in weights)/volume
                 for i in range(6)]
        controls, forces = [], []
        for axis,name in enumerate(("QX", "QY", "QZ")):
            node_set = region(odb,name,"nodeSets")
            u = frame.fieldOutputs["U"].getSubset(region=node_set).values
            rf = frame.fieldOutputs["RF"].getSubset(region=node_set).values
            if len(u) != 1 or len(rf) != 1:
                raise ValueError("Expected one node per macro control set")
            controls.append(float(data(u[0])[axis]))
            forces.append(float(data(rf[0])[axis]))
        energies = [r.historyOutputs["ALLSE"].data[-1][1]
                    for r in step.historyRegions.values() if "ALLSE" in r.historyOutputs]
        if len(energies) != 1:
            raise ValueError("Expected unique whole-model ALLSE history")
        energy = float(energies[0])
        measured = {"volume":volume,"sigma":sigma,"macro_displacements":controls,
                    "macro_RF":forces,"ALLSE":energy,"integration_points":count}
        if not all(math.isfinite(x) for x in [volume,energy]+sigma+controls+forces):
            raise ValueError("Nonfinite measured output")
        stress_scale = abs(expected["sigma_diagonal"][2])
        rel = manifest["relative_acceptance_tolerance"]
        checks = {"volume":abs(volume-1) <= 1e-10,
                  "energy":abs(energy-expected["energy"]) <= rel*expected["energy"],
                  "Fz":abs(forces[2]-expected["Fz_top"]) <= rel*stress_scale,
                  "reaction_vs_stress":abs(forces[2]-sigma[2]) <= rel*stress_scale}
        for i in range(3):
            checks["sigma_"+str(i+1)] = abs(sigma[i]-expected["sigma_diagonal"][i]) <= rel*stress_scale
            checks["eps_"+str(i+1)] = abs(controls[i]-expected["eps"][i]) <= rel*0.01
        checks["mean_shear"] = max(abs(x) for x in sigma[3:]) <= rel*stress_scale
        # Uniform affine solution: check the entire physical displacement field,
        # not only volume averages that could conceal a constraint error.
        physical = region(odb,"PHYSICAL","nodeSets")
        u_values = frame.fieldOutputs["U"].getSubset(region=physical).values
        if len(u_values) != (manifest["n"]+1)**3:
            raise ValueError("Incomplete physical displacement output")
        error = 0.0
        for value in u_values:
            xyz = value.instance.getNodeFromLabel(value.nodeLabel).coordinates
            observed = data(value)
            error = max(error, max(abs(observed[i]-xyz[i]*expected["eps"][i]) for i in range(3)))
        checks["affine_displacement"] = error <= rel*0.01
        measured["max_affine_displacement_error"] = error
        if expected["lateral"] != "fixed":
            checks["zero_lateral_stress"] = max(abs(sigma[0]),abs(sigma[1])) <= rel*stress_scale
        report = {"odb":str(odb_path),"case":odb_path.stem,
                  "measured":measured,"expected":expected,"checks":checks,
                  "status":"ok" if all(checks.values()) else "check_failed"}
    finally:
        odb.close()
    out_path.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    return all(checks.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("odb",type=Path)
    parser.add_argument("--expected",type=Path,default=Path("validation/abaqus_uniform/expected.json"))
    parser.add_argument("--out",type=Path)
    args = parser.parse_args()
    out = args.out or args.odb.with_suffix(".acceptance.json")
    raise SystemExit(0 if extract(args.odb,args.expected,out) else 1)

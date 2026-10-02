"""Abaqus Python: compare linear-user-element U, RF and reconstructed energy.

Read-only ODB extraction. Native stress output is unavailable for this route;
stress reconstruction remains external and is not Abaqus constitutive output.
"""
import argparse
import json
import math
from pathlib import Path
from extract_uniform_baseline import data, region


def extract(path, expected_path, output):
    import numpy as np
    from odbAccess import openOdb
    expected = json.loads(expected_path.read_text())
    if path.stem != expected["case"] or not all(expected["reference_checks"].values()):
        raise ValueError("Mismatched case or failed M4 reference")
    quadrature_path = expected_path.parent/(expected["case"]+".quadrature.npz")
    with np.load(quadrature_path) as archive:
        cells = archive["cells"].copy()
        stiffness = archive["local_stiffness"].copy()
    odb = openOdb(path=str(path.resolve()), readOnly=True)
    try:
        step = odb.steps["COMPRESSION"]
        frame = step.frames[-1]
        if abs(frame.frameValue-1) > 1e-8:
            raise ValueError("Incomplete compression step")
        values = frame.fieldOutputs["U"].getSubset(region=region(odb,"PHYSICAL","nodeSets")).values
        observed = {}
        for value in values:
            label = str(value.nodeLabel)
            if label in observed:
                raise ValueError("Duplicate physical node label")
            observed[label] = [float(x) for x in data(value)]
        if set(observed) != set(expected["nodes"]):
            raise ValueError("Incomplete or mismatched physical displacement field")
        error = max(abs(observed[k][i]-expected["nodes"][k]["u"][i]) for k in observed for i in range(3))
        controls, forces = [], []
        for axis, name in enumerate(("QX","QY","QZ")):
            selection = region(odb,name,"nodeSets")
            u = frame.fieldOutputs["U"].getSubset(region=selection).values
            rf = frame.fieldOutputs["RF"].getSubset(region=selection).values
            if len(u) != 1 or len(rf) != 1:
                raise ValueError("Expected one macro control node")
            controls.append(float(data(u[0])[axis]))
            forces.append(float(data(rf[0])[axis]))
        u_array = np.array([observed[str(i+1)] for i in range(len(observed))])
        u_cell = u_array[cells].reshape(-1,24)
        energy = float(0.5*np.einsum("ea,eab,eb->",u_cell,stiffness,u_cell))
        rel = expected["relative_acceptance_tolerance"]
        displacement_scale = max(0.01,max(abs(x) for x in expected["eps"]))
        stress_scale = abs(expected["Fz"])
        macro_work = 0.5*sum(f*u for f,u in zip(forces,controls))
        checks = {"finite":all(math.isfinite(x) for x in [energy,error,macro_work]+controls+forces+[v for row in observed.values() for v in row]),
            "displacement_field":error <= rel*displacement_scale,
            "Fz":abs(forces[2]-expected["Fz"]) <= rel*stress_scale,
            "energy_reconstructed":abs(energy-expected["energy"]) <= rel*abs(expected["energy"]),
            "macro_work":abs(macro_work-energy) <= rel*abs(expected["energy"])}
        for axis in range(3):
            checks["macro_strain_"+str(axis)] = abs(controls[axis]-expected["eps"][axis]) <= rel*displacement_scale
        if expected["lateral"] == "relaxed_free":
            checks["zero_lateral_macro_force"] = max(abs(forces[0]),abs(forces[1])) <= rel*stress_scale
        report = {"case":path.stem, "measured":{"macro_displacements":controls,"macro_RF":forces,
            "energy_reconstructed_from_u":energy,"energy_source":"Independent element matrices and Abaqus U; no native ALLSE claim",
            "macro_work_from_RF_U":macro_work,"max_displacement_error":error},
            "relative_Fz_difference":abs(forces[2]-expected["Fz"])/stress_scale,
            "relative_energy_difference":abs(energy-expected["energy"])/abs(expected["energy"]),
            "checks":{k:bool(v) for k,v in checks.items()},"status":"ok" if all(checks.values()) else "check_failed"}
    finally:
        odb.close()
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    output.with_suffix(".displacements.json").write_text(json.dumps(observed,allow_nan=False)+"\n")
    print(json.dumps(report,indent=2))
    return all(checks.values())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("odb",type=Path)
    p.add_argument("--expected",type=Path,required=True)
    p.add_argument("--out",type=Path)
    a = p.parse_args()
    raise SystemExit(0 if extract(a.odb,a.expected,a.out or a.odb.with_suffix(".acceptance.json")) else 1)

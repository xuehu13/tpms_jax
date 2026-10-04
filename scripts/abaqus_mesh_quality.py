"""Read-only C3D10 quality/energy diagnostics; run with abaqus python.

Reuses the validated bulk reader. These metrics screen mesh sensitivity and
do not establish convergence of a local stress maximum.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from extract_abaqus_binary import read_field
from extract_uniform_baseline import region


def weighted_quantiles(values, weights, probabilities=(.5, .95, .99, .995)):
    """Inverse weighted empirical CDF, with no interpolation."""
    values, weights = np.asarray(values), np.asarray(weights)
    if values.shape != weights.shape or not np.all(np.isfinite(values)):
        raise ValueError("Invalid samples")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("Expected positive finite weights")
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    indices = np.searchsorted(cumulative, np.asarray(probabilities)*cumulative[-1])
    return values[order][np.minimum(indices, len(order)-1)].tolist()


def von_mises(stress):
    """S11,S22,S33,S12,S13,S23 in global axes."""
    return np.sqrt(.5*((stress[:, 0]-stress[:, 1])**2+
                   (stress[:, 1]-stress[:, 2])**2+(stress[:, 2]-stress[:, 0])**2)
                   +3*np.sum(stress[:, 3:]**2, axis=1))


def diagnose(odb_path, expected_path, out, nodal_out=None):
    from odbAccess import openOdb
    from abaqusConstants import INTEGRATION_POINT
    manifest = json.loads(expected_path.read_text())
    if manifest["case"] != odb_path.stem:
        raise ValueError("Case name mismatch")
    for suffix, field in ((".inp", "input_sha256"), (".mesh.npz", "mesh_sha256")):
        path = expected_path.with_name(manifest["case"]+suffix)
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[field]:
            raise ValueError("Analyzed file hash mismatch")
    mesh = np.load(expected_path.with_name(manifest["case"]+".mesh.npz"))
    xyz = mesh["points"][mesh["cells"][:, :4]]
    det = np.linalg.det(np.transpose(xyz[:, 1:]-xyz[:, :1], (0, 2, 1)))
    edges = np.array(((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)))
    length2 = np.sum((xyz[:, edges[:, 0]]-xyz[:, edges[:, 1]])**2, axis=(1, 2))
    quality = 12*(det/2)**(2/3)/length2
    if np.any(det <= 0) or not np.all(np.isfinite(quality)):
        raise ValueError("Invalid linear geometry")
    if not np.isclose(quality.min(), manifest["geometry"]["mean_ratio_min"], rtol=1e-10):
        raise ValueError("Quality definition differs from mesh audit")
    centroids = xyz.mean(axis=1)
    odb = openOdb(path=str(odb_path.resolve()), readOnly=True)
    try:
        frame = odb.steps["COMPRESSION"].frames[-1]
        if abs(frame.frameValue-1) > 1e-8:
            raise ValueError("Incomplete load")
        solid = region(odb, "SOLID", "elementSets")
        fields = {}
        field_instance = None
        for name in ("S", "E", "IVOL"):
            field = frame.fieldOutputs[name]
            if name != "IVOL" and tuple(field.componentLabels) != tuple(
                    name+suffix for suffix in ("11", "22", "33", "12", "13", "23")):
                raise ValueError("Unexpected tensor component convention")
            instance, labels, ips, values = read_field(
                field.getSubset(region=solid, position=INTEGRATION_POINT))
            if field_instance is not None and instance != field_instance:
                raise ValueError("Fields belong to different instances")
            field_instance = instance
            if not np.array_equal(labels, np.repeat(np.arange(1, len(xyz)+1), 4)):
                raise ValueError("Element labels mismatch")
            if not np.array_equal(ips, np.tile(np.arange(1, 5), len(xyz))):
                raise ValueError("Integration point labels mismatch")
            fields[name] = values
        stress, strain, weight = fields["S"], fields["E"], fields["IVOL"].reshape(-1)
        if stress.shape != strain.shape or stress.shape[1] != 6:
            raise ValueError("Tensor fields mismatch")
        if not np.all(np.isfinite(stress)) or not np.all(np.isfinite(strain)):
            raise ValueError("Nonfinite field")
        energy = .5*np.einsum("qi,qi->q", stress, strain)*weight
        if not np.all(np.isfinite(weight)) or np.any(weight <= 0) or np.min(energy) < -1e-12*np.sum(np.abs(energy)):
            raise ValueError("Invalid volume or elastic energy")
        mises = von_mises(stress)
        volume_e = weight.reshape(-1, 4).sum(axis=1)
        energy_e = energy.reshape(-1, 4).sum(axis=1)
        vm_e = mises.reshape(-1, 4).max(axis=1)
        total_volume, total_energy = weight.sum(), energy.sum()
        acceptance_path = odb_path.with_suffix(".acceptance.json")
        acceptance = json.loads(acceptance_path.read_text(encoding="utf-8-sig"))
        if acceptance["status"] != "ok" or not all(acceptance["checks"].values()):
            raise ValueError("Physical acceptance missing or failed")
        if acceptance["input_sha256"] != manifest["input_sha256"]:
            raise ValueError("Accepted input differs from diagnosed input")
        if abs(total_energy-acceptance["measured"]["ALLSE"]) > 1e-6*total_energy:
            raise ValueError("Diagnostic energy differs from accepted ALLSE")
        distorted = None
        distortion_set = None
        for owner in (odb.rootAssembly, odb.rootAssembly.instances[instance]):
            for name, elem_set in owner.elementSets.items():
                if "distort" in name.lower():
                    ids = []
                    for block in elem_set.elements:
                        ids.extend([block.label] if hasattr(block, "label") else
                                   [element.label for element in block])
                    ids = np.unique(ids)
                    if np.any(ids < 1) or np.any(ids > len(xyz)):
                        raise ValueError("Invalid distortion warning labels")
                    distorted = np.isin(np.arange(1, len(xyz)+1), ids)
                    distortion_set = name
                    break
        def shares(mask):
            return {"elements": int(mask.sum()), "element_fraction": float(mask.mean()),
                    "volume_fraction": float(volume_e[mask].sum()/total_volume),
                    "energy_fraction": float(energy_e[mask].sum()/total_energy)}
        def peaks(indices):
            return [{"element": int(i+1), "centroid": centroids[i].tolist(),
                     "quality": float(quality[i]), "max_IP_mises": float(vm_e[i]),
                     "energy_fraction": float(energy_e[i]/total_energy)}
                    for i in indices]
        probabilities = [.5, .95, .99, .995]
        middle = (centroids[:, 2] >= .1) & (centroids[:, 2] <= .9)
        middle_ip = np.repeat(middle, 4)
        result = {"case": odb_path.stem, "status": "ok",
                  "diagnostic_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "input_sha256": manifest["input_sha256"], "mesh_sha256": manifest["mesh_sha256"],
                  "elements": len(xyz), "volume": float(total_volume), "IP_energy": float(total_energy),
                  "mises_mean": float(mises@weight/total_volume),
                  "mises_max": float(mises.max()), "quantile_probabilities": probabilities,
                  "mises_volume_quantiles": weighted_quantiles(mises, weight),
                  "quality_min": float(quality.min()), "quality_count_p01": float(np.quantile(quality, .01)),
                  "low_quality": {str(q): shares(quality < q) for q in (.05, .1, .2)},
                  "Abaqus_distortion_set": distortion_set,
                  "Abaqus_distorted": None if distorted is None else shares(distorted),
                  "loading_band": shares(~middle),
                  "middle_height_mises_quantiles": weighted_quantiles(mises[middle_ip], weight[middle_ip]),
                  "region_definition": "Element centroid z in [0.1,0.9]; approximate region, not exact IP cut",
                  "stress_peaks": peaks(np.argsort(vm_e)[-20:][::-1]),
                  "worst_quality": peaks(np.argsort(quality)[:20]),
                  "local_stress_convergence_claim": False}
        if nodal_out is not None:
            physical = region(odb, "PHYSICAL", "nodeSets")
            _, labels, _, u = read_field(frame.fieldOutputs["U"].getSubset(region=physical), nodal=True)
            if not np.array_equal(labels, np.arange(1,len(mesh["points"])+1)):
                raise ValueError("Incomplete nodal field")
            take = np.unique(np.linspace(0,len(labels)-1,min(4096,len(labels))).astype(int))
            nodal_out.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(nodal_out, points=mesh["points"][take], u=u[take], labels=labels[take])
            result["nodal_sample"] = {"points":len(take), "path":str(nodal_out), "sha256":hashlib.sha256(nodal_out.read_bytes()).hexdigest()}
    finally:
        odb.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: result[k] for k in ("case", "mises_max", "mises_volume_quantiles",
                    "low_quality", "Abaqus_distorted")}, indent=2), flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("odb", type=Path)
    p.add_argument("--expected", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--nodal-out", type=Path)
    a = p.parse_args()
    diagnose(a.odb, a.expected, a.out, a.nodal_out)

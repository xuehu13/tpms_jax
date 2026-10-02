"""Two volume-only relocation cases on the already analyzed G24 skin.

Task-specific reproduction script, not another general mesh framework.
"""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import gmsh
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/"scripts")]
from binary_gyroid import audit_linear, boundary_faces
from prepare_abaqus_binary import prepare


def prepare_study(source, output):
    case = "binary_gyroid_G24_R0_C3D10_fixed"
    expected = json.loads((source/(case+".expected.json")).read_text())
    path = source/(case+".mesh.npz")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected["mesh_sha256"]:
        raise ValueError("Analyzed source mesh hash mismatch")
    raw = np.load(path)
    points, cells = raw["points"][:expected["geometry"]["nodes"]], raw["cells"][:, :4]
    before = audit_linear(points, cells)
    boundary = boundary_faces(cells)
    surface_ids = np.unique(boundary)
    interior_ids = np.setdiff1d(np.arange(len(points)), surface_ids)
    gmsh.initialize()
    gmsh.logger.start()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("fixed_skin_quality_check")
        surface = gmsh.model.addDiscreteEntity(2)
        volume = gmsh.model.addDiscreteEntity(3, boundary=[surface])
        gmsh.model.mesh.addNodes(2, surface, (surface_ids+1).tolist(), points[surface_ids].ravel().tolist())
        gmsh.model.mesh.addElementsByType(surface, 2, list(range(1, len(boundary)+1)), (boundary+1).ravel().tolist())
        gmsh.model.mesh.addNodes(3, volume, (interior_ids+1).tolist(), points[interior_ids].ravel().tolist())
        gmsh.model.mesh.addElementsByType(volume, 4, list(range(len(boundary)+1, len(boundary)+1+len(cells))), (cells+1).ravel().tolist())
        gmsh.option.setNumber("Mesh.OptimizeThreshold", .5)
        gmsh.model.mesh.optimize("Relocate3D", force=True, niter=10, dimTags=[(3, volume)])
        tags, coordinates, _ = gmsh.model.mesh.getNodes()
        order = np.argsort(tags)
        if not np.array_equal(np.array(tags)[order], np.arange(1, len(points)+1)):
            raise ValueError("Optimizer changed nodes")
        xyz = np.array(coordinates).reshape(-1, 3)[order]
        types, _, connectivity = gmsh.model.mesh.getElements(3, volume)
        tets = np.asarray(connectivity[0], dtype=int).reshape(-1, 4)-1
        if list(types) != [4] or not np.array_equal(tets, cells):
            raise ValueError("Relocation changed connectivity")
        if not np.array_equal(xyz[surface_ids], points[surface_ids]):
            raise ValueError("Relocation changed prescribed boundary")
        after = audit_linear(xyz, tets)
        if abs(after["volume"]-before["volume"]) > 1e-12:
            raise ValueError("Relocation changed volume")
        messages = gmsh.logger.get()
        if any(m.startswith("Error") for m in messages):
            raise ValueError("Gmsh error")
    finally:
        gmsh.logger.stop()
        gmsh.finalize()
    output.mkdir(parents=True, exist_ok=True)
    for lateral in ("fixed", "relaxed_free"):
        if (output/(case.replace("fixed", lateral)+".inp")).exists():
            raise ValueError("Output input already exists; use a new directory")
    cache = output/"G24_relocated.linear_cache.npz"
    np.savez_compressed(cache, points=xyz, cells=tets, geometry_N=24, c=expected["c"], fe_refinement=0)
    record = {"method": "Relocate3D", "niter": 10, "threshold": .5,
              "source_mesh_sha256": expected["mesh_sha256"], "before": before, "after": after,
              "boundary_coordinates_bitwise_equal": True, "connectivity_bitwise_equal": True,
              "max_node_movement": float(np.linalg.norm(xyz-points, axis=1).max()),
              "gmsh_version": gmsh.__version__, "gmsh_messages": messages}
    cases = []
    with (output/"prepare.console.txt").open("w") as log, contextlib.redirect_stdout(log):
        for lateral in ("fixed", "relaxed_free"):
            manifest = prepare(output, n=24, lateral=lateral, cached_mesh=cache)
            manifest["meshing"]["quality_study"] = {k: v for k, v in record.items()
                                                    if k not in ("before", "after", "gmsh_messages")}
            (output/(manifest["case"]+".expected.json")).write_text(json.dumps(manifest, indent=2)+"\n")
            cases.append(manifest["case"])
    fixed, free = [np.load(output/(c+".mesh.npz")) for c in cases]
    if not all(np.array_equal(fixed[k], free[k]) for k in ("points", "cells")):
        raise ValueError("Fixed/free meshes differ")
    (output/"relocation.json").write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps({"cases": cases, "before": before, "after": after}, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-package", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    prepare_study(a.source_package, a.output)

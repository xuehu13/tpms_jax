"""Matrix parsing and checks against the installed M4 tangent."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "abaqus_element_comparison", Path(__file__).parents[1]/"scripts"/"abaqus_element_comparison.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("full", [False, True])
def test_parser_maps_labels_without_doubling_symmetric_entries(tmp_path, full):
    path = tmp_path/"example.mtx"
    entries = "** stiffness\n20,1,20,1,2.0\n20,1,10,3,-1.25D+00\n10,3,10,3,3.0\n"
    if full:
        entries += "10,3,20,1,-1.25\n"
    path.write_text(entries)
    result = module.read_matrix(path, labels=[10,20])
    assert result[3,3] == 2 and result[2,2] == 3
    assert result[3,2] == result[2,3] == -1.25
    assert np.count_nonzero(result) == 4


@pytest.mark.parametrize("entries", [
    "1,1,1,1,2\n1,1,1,1,2\n",  # duplicate
    "1,1,2,1,2\n2,1,1,1,3\n",  # inconsistent symmetry
    "9,1,1,1,2\n", "1,4,1,1,2\n",  # unknown label/DOF
    "1,1,1,1,NaN\n", "", "1,1,2\n",
])
def test_parser_rejects_invalid_evidence(tmp_path, entries):
    path = tmp_path/"invalid.mtx"
    path.write_text(entries)
    with pytest.raises(ValueError):
        module.read_matrix(path)


@pytest.mark.parametrize("name", ["regular", "distorted"])
def test_installed_m4_tangent_and_affine_subspace(name):
    xyz = module.geometries()[name]
    full, bbar, weights, _ = module.independent_matrices(xyz)
    actual, _, runtime_weights, residual = module.runtime_matrix(xyz)
    assert module.relative_norm(actual, full) < 1e-12
    assert weights.sum() == pytest.approx(runtime_weights.sum(), rel=1e-12)
    assert residual < 1e-12
    for i in range(3):
        for j in range(3):
            H = np.zeros((3,3)); H[i,j] = 1
            v = (xyz @ H.T).ravel()
            assert np.linalg.norm((full-bbar)@v) < 1e-12
    for matrix in (full, bbar, actual):
        quality = module.matrix_quality(matrix, xyz)
        assert quality["null_mode_count"] == 6
        assert quality["positive_mode_count"] == 18
        assert quality["rigid_body_relative_residual"] < 1e-12
    # The two rules agree on affine fields but remain different operators.
    assert module.relative_norm(full, bbar) > 0.01
    eigenvalues = np.linalg.eigvalsh(full-bbar)
    assert eigenvalues.min() > -1e-12
    assert eigenvalues.max() > 0.1


def test_inverted_geometry_is_rejected():
    xyz = module.geometries()["regular"]*np.array([-1,1,1])
    with pytest.raises(ValueError, match="Jacobian"):
        module.independent_matrices(xyz)

"""Check the installed keyword matrix protocol and independent density formula."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

script_path = Path(__file__).parents[1]/"scripts"
sys.path.insert(0, str(script_path))
spec = importlib.util.spec_from_file_location("prepare_abaqus_discrete",script_path/"prepare_abaqus_discrete.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_matrix_protocol_preserves_columns_and_signed_entries():
    # Every upper-triangle entry is distinct, so row/column transposes and
    # incorrect continuation between columns cannot pass this round trip.
    matrix = np.arange(24*24,dtype=float).reshape(24,24)+0.125
    matrix = np.triu(matrix)+np.triu(matrix,1).T
    matrix[::2,:] *= -1
    matrix = (matrix+matrix.T)/2
    lines = iter(module.linear_matrix_lines(matrix))
    reconstructed = np.zeros_like(matrix)
    for col in range(24):
        values = []
        while len(values) < col+1:
            line = next(lines)
            assert len(line) <= 256
            fields = line.split(",")
            assert len(fields) <= 4
            values.extend(float(field) for field in fields)
        assert len(values) == col+1
        reconstructed[:col+1,col] = values
        reconstructed[col,:col+1] = values
    assert list(lines) == []
    assert np.array_equal(matrix,reconstructed)


def test_density_matches_jax_at_arbitrary_and_transition_points():
    import jax
    jax.config.update("jax_enable_x64",True)
    from geometry import density
    points = np.r_[np.random.default_rng(1234).random((80,3)), [[0,0,0],[0.1,0.4,0.9]]]
    for beta in (10,20,40):
        assert np.max(np.abs(module.density_numpy(points,beta=beta)-np.asarray(density(points,0.541062,beta)))) < 1e-14


@pytest.mark.parametrize("n,beta,emin", [(1,20,1e-3),(22,20,1e-3),(4,-1,1e-3),(4,20,0),(4,float("nan"),1e-3)])
def test_invalid_case_rejected_before_solver_or_job(tmp_path,n,beta,emin):
    with pytest.raises(ValueError):
        module.prepare(tmp_path,n=n,beta=beta,emin_ratio=emin)
    assert list(tmp_path.iterdir()) == []


def test_direct_reference_satisfies_free_lateral_equilibrium():
    import jax
    jax.config.update("jax_enable_x64",True)
    from scripts.m4_numerical_study import solve_case
    from density_fem import avg_stress
    problem, solutions, H, ex, ey, _, _ = solve_case(
        2,20,1e-3,"relaxed_free",solver_options=module.DIRECT_REFERENCE_OPTIONS)
    stress = np.asarray(avg_stress(problem,solutions))
    residual = np.asarray(problem.compute_residual(solutions)[0])
    assert max(abs(stress[0,0]),abs(stress[1,1])) < 1e-12
    assert np.max(np.abs(problem.P_mat.T @ residual.ravel())) < 1e-12
    assert H[0,0] == ex and H[1,1] == ey
    assert abs(residual[:,2].sum()) < 1e-12

"""Physical-point volume calibration and its fixed/free forward use."""
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from density_fem import calibrate_density_c
from geometry import density
from scripts.m4_numerical_study import evaluate_case

jax.config.update("jax_enable_x64", True)


def test_calibration_uses_volume_weights_not_point_count():
    points = jnp.array([[[0., 0., 0.], [.25, 0., 0.]]])
    problem = SimpleNamespace(physical_quad_points=points, JxW=jnp.array([[[.2, .8]]]))
    record = calibrate_density_c(problem, 10., .35)
    rho = np.asarray(density(points, record["c"], 10.))
    assert abs(float((rho*np.array([[.2, .8]])).sum())-.35) <= 1e-10
    assert abs(rho.mean()-.35) > .01


@pytest.mark.parametrize("lateral", ["fixed", "relaxed_free"])
def test_matched_density_uses_single_actual_problem(lateral, monkeypatch):
    import density_fem
    original = density_fem.make_density_problem
    builds = []
    def counted(*args, **kwargs):
        result = original(*args, **kwargs)
        builds.append(result)
        return result
    monkeypatch.setattr(density_fem, "make_density_problem", counted)
    monkeypatch.setattr("scripts.m4_numerical_study.make_density_problem", counted)
    row = evaluate_case(8, 10., .001, lateral, target_vf=.35, include_problem=True)
    assert row["status"] == "ok" and all(row["checks"].values())
    assert len(builds) == 1 and builds[0] is row["_problem"]
    assert abs(row["vf_int"]-.35) <= 1e-10
    problem = row["_problem"]
    np.testing.assert_allclose(problem.rho, density(problem.physical_quad_points, row["_projection_c"], 10.), rtol=0, atol=0)
    if lateral == "relaxed_free":
        assert max(abs(row["sigma_xx"]), abs(row["sigma_yy"])) <= 1e-8


def test_explicit_c_changes_density_and_binary_reference_consistently():
    row = evaluate_case(8, 20., .001, "fixed", c=.6, include_problem=True)
    assert row["status"] == "ok" and row["_projection_c"] == .6
    problem = row["_problem"]
    np.testing.assert_allclose(problem.rho, density(problem.physical_quad_points, .6, 20.), rtol=0, atol=0)
    from geometry import gyroid
    weights = np.asarray(problem.JxW)[:, 0, :]
    expected = float(((np.abs(gyroid(problem.physical_quad_points)) <= .6)*weights).sum()/weights.sum())
    assert row["vf_binary_ref"] == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize("beta,target", [(0., .35), (float("nan"), .35), (20., 0.), (20., 1.)])
def test_calibration_rejects_invalid_parameters(beta, target):
    with pytest.raises(ValueError):
        calibrate_density_c(None, beta, target)

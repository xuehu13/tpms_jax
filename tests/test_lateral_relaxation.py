"""M3-B tests: automatic lateral relaxation to zero average lateral stress."""
import numpy as onp
import pytest
import jax
import jax.numpy as jnp

from fem import solve
from density_fem import avg_stress, make_density_problem, solve_lateral_relaxation
from geometry import density
from pbc import xy_compression_fixed_dofs

jax.config.update("jax_enable_x64", True)

C_CALIB = 0.541062
BETA = 20.0


def test_uniform_relaxation_matches_analytic():
    for n in (4, 8):
        out = solve_lateral_relaxation(
            n, n, n, rho_quad=1.0, eps_z=-0.01,
            fixed_dofs=lambda pts: xy_compression_fixed_dofs(pts, n, n, n))
        assert out["eps_x"] == pytest.approx(0.003, rel=1e-9)
        assert out["eps_y"] == pytest.approx(0.003, rel=1e-9)
        s = out["sigma_avg"]
        assert abs(s[0, 0]) <= 1e-12 and abs(s[1, 1]) <= 1e-12
        assert s[2, 2] == pytest.approx(-0.1, rel=1e-10)
        r = onp.asarray(out["problem"].compute_residual(out["sol_list"])[0])
        pts = onp.asarray(out["problem"].fe.points)
        top = onp.where(onp.isclose(pts[:, 2], 1.0, atol=1e-8))[0]
        Fz = float(r[top, 2].sum())
        assert Fz == pytest.approx(-0.1, rel=1e-10)
        eps_zz = -0.01
        U = 0.5 * float(s[2, 2] * eps_zz)
        assert U == pytest.approx(5.0e-4, rel=1e-10)


def test_same_instance_reuse_consistency():
    """Re-solving the base load on the same instance must reproduce the
    original average stress (guards against stale internal_vars/jit state)."""
    n = 4
    out = solve_lateral_relaxation(
        n, n, n, rho_quad=1.0, eps_z=-0.01,
        fixed_dofs=lambda pts: xy_compression_fixed_dofs(pts, n, n, n))
    problem = out["problem"]
    H0 = onp.diag(onp.array([0.0, 0.0, -0.01]))
    problem.set_params(H0, 1.0, 10.0, 1e-2)
    sol_again = solve(problem)
    s_again = avg_stress(problem, sol_again)
    assert onp.allclose(s_again, out["sigma_avg_base"], atol=1e-12)


def test_gyroid_relaxed_vs_fixed():
    n = 8
    H_fixed = onp.diag(onp.array([0.0, 0.0, -0.01]))
    pins = lambda pts: xy_compression_fixed_dofs(pts, n, n, n)

    # A: lateral macro strain fixed to zero
    problem_a = make_density_problem(
        n, n, n, H_macro=jnp.asarray(H_fixed), rho_quad=None,
        periodic_axes=(0, 1), fixed_class=None, fixed_dofs=pins)
    rho = density(problem_a.physical_quad_points, C_CALIB, BETA, 1.0)
    problem_a.set_params(H_fixed, rho)
    sol_a = solve(problem_a)
    s_a = avg_stress(problem_a, sol_a)

    # B: automatic lateral relaxation on the same grid and density
    out = solve_lateral_relaxation(n, n, n, rho_quad=rho, eps_z=-0.01,
                                   fixed_dofs=pins)
    s_b = out["sigma_avg"]

    assert abs(s_b[0, 0]) <= 1e-8 and abs(s_b[1, 1]) <= 1e-8
    assert 0.001 <= out["eps_x"] <= 0.006 and 0.001 <= out["eps_y"] <= 0.006
    # the two lateral strains are NOT assumed equal (the cell is not
    # x/y-symmetric); report the measured split
    assert abs(out["eps_x"] - out["eps_y"]) <= 2e-3
    # relaxation must soften the axial response at fixed eps_z
    assert abs(s_b[2, 2]) < abs(s_a[2, 2])
    # shear averages negligible in both cases
    for s in (s_a, s_b):
        assert max(abs(s[0, 1]), abs(s[0, 2]), abs(s[1, 2])) <= 1e-8
    # same density field -> same FEM volume fraction in both cases
    JxW = onp.asarray(problem_a.JxW)[:, 0, :]
    vf = float((onp.asarray(rho) * JxW).sum() / JxW.sum())
    assert 0.3 <= vf <= 0.4

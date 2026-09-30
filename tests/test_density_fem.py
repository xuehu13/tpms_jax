"""M3-A tests: integration-point density in the periodic HEX8 problem."""
import numpy as onp
import pytest
import jax
import jax.numpy as jnp

from density_fem import layered_rho, make_density_problem
from fem import NU, solve
from geometry import density
from pbc import xy_compression_fixed_dofs

jax.config.update("jax_enable_x64", True)

C_CALIB = 0.541062
BETA = 20.0


def build(n, H, rho_quad):
    return make_density_problem(
        n, n, n, H_macro=jnp.asarray(H), rho_quad=rho_quad,
        periodic_axes=(0, 1), fixed_class=None,
        fixed_dofs=lambda pts: xy_compression_fixed_dofs(pts, n, n, n))


def test_quad_points_and_rho_shape():
    n = 4
    problem = build(n, onp.diag(onp.array([0.0, 0.0, -0.01])), None)
    X_q = onp.asarray(problem.physical_quad_points)
    assert X_q.shape == (n**3, 8, 3)
    assert X_q.min() >= 0.0 and X_q.max() <= 1.0
    JxW = onp.asarray(problem.JxW)[:, 0, :]
    assert JxW.shape == (n**3, 8)
    assert JxW.min() > 0.0
    assert JxW.sum() == pytest.approx(1.0, abs=1e-10)
    rho = density(X_q, C_CALIB, BETA, 1.0)
    assert rho.shape == (n**3, 8)
    assert float(rho.min()) >= 0.0 and float(rho.max()) <= 1.0


def test_uniform_rho_full_material_matches_m2():
    n = 4
    H = onp.diag(onp.array([0.003, 0.003, -0.01]))
    problem = build(n, H, jnp.ones((n**3, 8)))  # rho = 1 -> E_q = E_s = 10
    sol_list = solve(problem)
    w = onp.asarray(sol_list[0])
    r = onp.asarray(problem.compute_residual(sol_list)[0])
    pts = onp.asarray(problem.fe.points)
    top = onp.where(onp.isclose(pts[:, 2], 1.0, atol=1e-8))[0]
    Fz = float(r[top, 2].sum())
    assert onp.abs(w).max() <= 1e-12
    assert Fz == pytest.approx(-0.1, abs=1e-10)      # same anchor as M2-C case B
    assert abs(strain_energy_check(problem, sol_list) - 5.0e-4) <= 1e-14


def strain_energy_check(problem, sol_list):
    u_grad = onp.asarray(problem.fe.sol_to_grad(sol_list[0]))
    eps = 0.5 * (u_grad + onp.swapaxes(u_grad, -1, -2)) + onp.asarray(problem.H_macro)
    lam = onp.asarray(problem.internal_vars[1])[..., None, None]
    mu = onp.asarray(problem.internal_vars[2])[..., None, None]
    sigma = lam * onp.trace(eps, axis1=-2, axis2=-1)[..., None, None] * onp.eye(3) \
        + 2.0 * mu * eps
    JxW = onp.asarray(problem.JxW)[:, 0, :]
    return 0.5 * float(onp.sum(sigma * eps * JxW[..., None, None]))


def test_uniform_intermediate_density_scaling():
    n = 4
    H = onp.diag(onp.array([0.003, 0.003, -0.01]))
    E_s, ratio = 10.0, 0.5
    E_min = 1e-3 * E_s
    problem = build(n, H, ratio * onp.ones((n**3, 8)))
    sol_list = solve(problem)
    u_grad = onp.asarray(problem.fe.sol_to_grad(sol_list[0]))
    eps = 0.5 * (u_grad + onp.swapaxes(u_grad, -1, -2)) + H
    E_mid = E_min + ratio * (E_s - E_min)
    sigma_zz = float((onp.asarray(problem.internal_vars[1])[..., None, None] * onp.trace(eps, axis1=-2, axis2=-1)[..., None, None]
                      + 2.0 * onp.asarray(problem.internal_vars[2])[..., None, None] * eps)[..., 2, 2].mean())
    assert sigma_zz == pytest.approx(E_mid * -0.01, rel=1e-12)
    U = strain_energy_check(problem, sol_list)
    assert U == pytest.approx(0.5 * E_mid * 0.01**2, rel=1e-12)
    # linear interpolation in rho at p = 1: response scales exactly with E_q
    assert sigma_zz / -0.1 == pytest.approx(E_mid / E_s, rel=1e-12)


def test_two_layer_series_benchmark():
    n = 4  # interface z = 0.5 lies on an element boundary
    H = onp.diag(onp.array([0.0, 0.0, -0.01]))
    problem = build(n, H, None)
    rho = layered_rho(problem, 1.0, 0.5)
    problem.set_params(H, rho)
    sol_list = solve(problem)
    w = onp.asarray(sol_list[0])
    r = onp.asarray(problem.compute_residual(sol_list)[0])
    pts = onp.asarray(problem.fe.points)
    top = onp.where(onp.isclose(pts[:, 2], 1.0, atol=1e-8))[0]
    Fz = float(r[top, 2].sum())

    M1 = 10.0 * (1.0 - NU) / ((1.0 + NU) * (1.0 - 2.0 * NU))
    E2 = 1e-2 + 0.5 * (10.0 - 1e-2)
    M2 = E2 * (1.0 - NU) / ((1.0 + NU) * (1.0 - 2.0 * NU))
    M_eff = 1.0 / (0.5 / M1 + 0.5 / M2)
    assert Fz == pytest.approx(M_eff * -0.01, rel=1e-10)
    assert strain_energy_check(problem, sol_list) \
        == pytest.approx(0.5 * M_eff * 0.01**2, rel=1e-10)

    # per-layer axial strain follows sigma_zz / M_i; sigma_zz is continuous
    u_grad = onp.asarray(problem.fe.sol_to_grad(sol_list[0]))
    eps = 0.5 * (u_grad + onp.swapaxes(u_grad, -1, -2)) + H
    cell_z = pts[onp.asarray(problem.fe.cells)].mean(axis=1)[:, 2]
    bottom = cell_z < 0.5
    eps_zz_b = float(eps[bottom][..., 2, 2].mean())
    eps_zz_t = float(eps[~bottom][..., 2, 2].mean())
    sigma_zz = M_eff * -0.01
    assert eps_zz_b == pytest.approx(sigma_zz / M1, rel=1e-10)
    assert eps_zz_t == pytest.approx(sigma_zz / M2, rel=1e-10)
    sig_zz_b = float((onp.asarray(problem.internal_vars[1])[bottom][..., None, None]
                      * onp.trace(eps[bottom], axis1=-2, axis2=-1)[..., None, None]
                      * onp.eye(3) + 2.0 * onp.asarray(problem.internal_vars[2])[bottom][..., None, None]
                      * eps[bottom])[..., 2, 2].mean())
    sig_zz_t = float((onp.asarray(problem.internal_vars[1][~bottom])[..., None, None]
                      * onp.trace(eps[~bottom], axis1=-2, axis2=-1)[..., None, None]
                      * onp.eye(3) + 2.0 * onp.asarray(problem.internal_vars[2][~bottom])[..., None, None]
                      * eps[~bottom])[..., 2, 2].mean())
    assert abs(sig_zz_b - sig_zz_t) <= 1e-12, "axial stress continuity"
    assert onp.abs(problem.P_mat.T @ r.reshape(-1)).max() <= 1e-12


def test_gyroid_first_calculation():
    n = 8
    H = onp.diag(onp.array([0.0, 0.0, -0.01]))
    problem = build(n, H, None)
    rho = density(problem.physical_quad_points, C_CALIB, BETA, 1.0)
    problem.set_params(H, rho)
    sol_list = solve(problem)
    rho_o = onp.asarray(rho)
    JxW = onp.asarray(problem.JxW)[:, 0, :]
    assert 0.0 <= rho_o.min() and rho_o.max() <= 1.0
    vf_int = float((rho_o * JxW).sum() / JxW.sum())
    assert 0.2 <= vf_int <= 0.45, "FEM volume fraction in a sane band"

    w = onp.asarray(sol_list[0])
    assert onp.all(onp.isfinite(w))
    r = onp.asarray(problem.compute_residual(sol_list)[0])
    assert onp.abs(problem.P_mat.T @ r.reshape(-1)).max() <= 1e-9

    # periodic pairing of the fluctuation stays exact
    pts = onp.asarray(problem.fe.points)
    idx = onp.round(pts * n).astype(int)
    index_to_node = {(idx[i, 0], idx[i, 1], idx[i, 2]): i for i in range(len(pts))}
    other = {0: (1, 2), 1: (0, 2)}
    for axis in (0, 1):
        a1, a2 = other[axis]
        low = onp.where(idx[:, axis] == 0)[0]
        for node_low in low:
            partner = index_to_node[tuple(
                n if k == axis else idx[node_low, k] for k in range(3))]
            assert onp.array_equal(w[node_low], w[partner])

    # porous cell must be softer than the full-solid reference (-0.1)
    r_top = float(r[onp.where(onp.isclose(pts[:, 2], 1.0, atol=1e-8))[0], 2].sum())
    assert -0.05 < r_top < -0.005

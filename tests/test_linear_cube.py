"""M2-A tests: full-solid HEX8 linear elasticity benchmarks on the unit cube.

Both benchmarks are LINEAR displacement problems reproduced exactly by
trilinear HEX8 on a uniform grid, so discretization error is zero and the
residual errors below come only from the iterative linear solve (observed
max ~3e-7 at n=8 across all quantities). Tolerances are set 40-1000x above
the observed values.
"""
import numpy as onp
import pytest

import jax

jax.config.update("jax_enable_x64", True)

from fem import (E, NU, LAMBDA, MU, affine_bc, face_nodes, hex_jacobian_check,
                 internal_force, make_cube_problem, solve, strain_energy,
                 uniaxial_bc)


def boundary_mask(problem):
    pts = onp.asarray(problem.fe.points)
    return (onp.isclose(pts, 0.0, atol=1e-8) | onp.isclose(pts, 1.0, atol=1e-8)).any(axis=1)


def test_hex8_mesh_jacobian_and_volume():
    n = 4
    problem = make_cube_problem(n, n, n, affine_bc(0.01, 0.02, -0.005))
    num_nodes = (n + 1) ** 3
    num_cells = n**3
    assert onp.asarray(problem.fe.points).shape == (num_nodes, 3)
    assert onp.asarray(problem.fe.cells).shape == (num_cells, 8)
    pts = onp.asarray(problem.fe.points)
    assert pts.min() == pytest.approx(0.0, abs=1e-12)
    assert pts.max() == pytest.approx(1.0, abs=1e-12)
    min_det, total_vol = hex_jacobian_check(problem)
    assert min_det > 0.0, "non-positive Jacobian"
    assert total_vol == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize("n", [4, 8])
def test_benchmark_affine_displacement(n):
    ex, ey, ez = 0.01, 0.02, -0.005
    problem = make_cube_problem(n, n, n, affine_bc(ex, ey, ez))
    sol_list = solve(problem)

    u = onp.asarray(sol_list[0])
    pts = onp.asarray(problem.fe.points)
    u_exact = onp.stack([ex * pts[:, 0], ey * pts[:, 1], ez * pts[:, 2]], axis=1)
    assert onp.abs(u - u_exact).max() <= 1e-6

    u_grad = onp.asarray(problem.fe.sol_to_grad(sol_list[0]))
    eps = 0.5 * (u_grad + onp.swapaxes(u_grad, -1, -2))
    eps_exact = onp.array([[[[ex, 0.0, 0.0], [0.0, ey, 0.0], [0.0, 0.0, ez]]]])
    assert onp.abs(eps - eps_exact).max() <= 1e-8

    sigma_exact = (LAMBDA * (ex + ey + ez) * onp.eye(3)[None, None]
                   + 2.0 * MU * eps_exact)
    sigma = LAMBDA * onp.trace(eps, axis1=-2, axis2=-1)[..., None, None] * onp.eye(3) \
        + 2.0 * MU * eps
    assert onp.abs(sigma - sigma_exact).max() <= 1e-5

    U = strain_energy(problem, sol_list)
    U_exact = 0.5 * LAMBDA * (ex + ey + ez) ** 2 + MU * (ex**2 + ey**2 + ez**2)
    assert abs(U - U_exact) / U_exact <= 1e-10

    r = internal_force(problem, sol_list)
    free = ~boundary_mask(problem)
    assert onp.abs(r[free]).max() <= 1e-9, "equilibrium residual at free DOFs"

    # reaction through the +x face equals sigma_xx * A (A = 1)
    rx = float(r[face_nodes(problem, 0, 1.0), 0].sum())
    sigma_xx = LAMBDA * (ex + ey + ez) + 2.0 * MU * ex
    assert rx == pytest.approx(sigma_xx, abs=1e-5)


@pytest.mark.parametrize("n", [4, 8])
def test_benchmark_uniaxial_compression(n):
    delta = -0.01  # top uz, with L = 1
    problem = make_cube_problem(n, n, n, uniaxial_bc(delta))
    sol_list = solve(problem)

    u = onp.asarray(sol_list[0])
    pts = onp.asarray(problem.fe.points)
    u_exact = onp.stack([-NU * delta * pts[:, 0], -NU * delta * pts[:, 1],
                         delta * pts[:, 2]], axis=1)
    assert onp.abs(u - u_exact).max() <= 1e-6

    r = internal_force(problem, sol_list)
    top = face_nodes(problem, 2, 1.0)
    bottom = face_nodes(problem, 2, 0.0)
    Fz_top = float(r[top, 2].sum())
    Fz_bottom = float(r[bottom, 2].sum())
    # sigma_zz * A with sigma_zz = E * eps_zz = E * delta / L
    assert Fz_top == pytest.approx(E * delta, abs=1e-6)
    assert Fz_top + Fz_bottom == pytest.approx(0.0, abs=1e-9), "force balance"

    U = strain_energy(problem, sol_list)
    U_exact = 0.5 * E * delta**2
    assert abs(U - U_exact) / U_exact <= 1e-10

    u_grad = onp.asarray(problem.fe.sol_to_grad(sol_list[0]))
    eps = 0.5 * (u_grad + onp.swapaxes(u_grad, -1, -2))
    sigma = LAMBDA * onp.trace(eps, axis1=-2, axis2=-1)[..., None, None] * onp.eye(3) \
        + 2.0 * MU * eps
    assert onp.abs(eps[..., 2, 2] - delta).max() <= 1e-8
    assert onp.abs(sigma[..., 0, 0]).max() <= 1e-5, "lateral stress must vanish"
    assert onp.abs(sigma[..., 1, 1]).max() <= 1e-5

    # lateral Poisson expansion at the x = 1 face: u_x = -NU * delta
    ux_side = u[face_nodes(problem, 0, 1.0), 0]
    assert onp.abs(ux_side - (-NU * delta)).max() <= 1e-6

    free = ~boundary_mask(problem)
    assert onp.abs(r[free]).max() <= 1e-9

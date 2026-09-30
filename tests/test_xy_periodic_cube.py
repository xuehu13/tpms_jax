"""M2-C tests: XY-periodic solid cube under axial compression.

Both compression benchmarks prescribe a uniform H_macro so the analytic
solution is w = 0, u = H X -- reproduced exactly by trilinear HEX8; the
residual errors come only from the iterative linear solve. Tolerances keep
40x+ margin over the observed ~1e-11 values.
"""
import numpy as onp
import pytest
import jax
import jax.numpy as jnp

from fem import E, LAMBDA, MU, solve
from pbc import make_periodic_problem, xy_compression_fixed_dofs

jax.config.update("jax_enable_x64", True)

A_SINE = 0.001


def build_xy(n, H=None, sine=False):
    return make_periodic_problem(
        n, n, n,
        H_macro=jnp.zeros((3, 3)) if H is None else jnp.asarray(H),
        sine_force_amplitude=A_SINE if sine else None,
        periodic_axes=(0, 1), fixed_class=None,
        fixed_dofs=lambda pts: xy_compression_fixed_dofs(pts, n, n, n))


def test_xy_p_mat_structure_and_independent_z_faces():
    n = 4
    problem = build_xy(n)
    P = problem.P_mat
    num_nodes = (n + 1) ** 3
    num_classes = n * n * (n + 1)  # 80 for n=4
    # excluded groups: w_z groups of the two surfaces (Nx*Ny classes each,
    # because in-plane partners share a class) + the corner w_x, w_y groups
    n_excluded_groups = 2 * n * n + 2
    assert P.shape == (3 * num_nodes, 3 * num_classes - n_excluded_groups)
    # z faces are NOT periodic: bottom/top copies of one (x, y) location
    # belong to different classes
    pts = onp.asarray(problem.fe.points)
    bottom = onp.where(onp.isclose(pts[:, 2], 0.0, atol=1e-8))[0]
    top = onp.where(onp.isclose(pts[:, 2], 1.0, atol=1e-8))[0]
    i_low = onp.round(pts[bottom, :2] * n).astype(int)
    i_high = onp.round(pts[top, :2] * n).astype(int)
    assert onp.array_equal(i_low, i_high), "same in-plane ordering assumed"
    assert not onp.array_equal(problem.class_ids[bottom], problem.class_ids[top])
    # pinned rows: w_z on both surfaces (2*(n+1)^2) + two corner lateral DOFs
    nnz_rows, nnz_cols = onp.nonzero(onp.asarray(P.todense()))
    # zero rows = w_z of both surfaces + all dofs of the bottom corner class
    # (the 4 mesh corner nodes (0/1,0/1,0) are one physical periodic point)
    idx = onp.round(pts * n).astype(int)
    corner_class_nodes = onp.where((idx[:, 0] % n == 0) & (idx[:, 1] % n == 0)
                                   & (idx[:, 2] == 0))[0]
    zero_rows = onp.concatenate([3 * onp.concatenate([bottom, top]) + 2,
                                 3 * corner_class_nodes + 0,
                                 3 * corner_class_nodes + 1])
    assert set(onp.arange(P.shape[0]).tolist()) - set(nnz_rows.tolist())         == set(zero_rows.tolist())


def test_xy_periodic_pairing_and_free_z():
    n = 4
    problem = build_xy(n, sine=True)
    sol_list = solve(problem)
    w = onp.asarray(sol_list[0])
    pts = onp.asarray(problem.fe.points)
    other = {0: (1, 2), 1: (0, 2)}
    idx = onp.round(pts * n).astype(int)
    index_to_node = {(idx[i, 0], idx[i, 1], idx[i, 2]): i for i in range(len(pts))}
    for axis in (0, 1):
        a1, a2 = other[axis]
        low = onp.where(idx[:, axis] == 0)[0]
        for node_low in low:
            p_idx = [n if k == axis else idx[node_low, k] for k in range(3)]
            partner = index_to_node[tuple(p_idx)]
            assert problem.class_ids[partner] == problem.class_ids[node_low]
            assert onp.array_equal(w[node_low], w[partner]), "x/y periodicity broken"
    # z faces are flat: w_z pinned to zero there, everything else solved
    bottom = onp.where(onp.isclose(pts[:, 2], 0.0, atol=1e-8))[0]
    top = onp.where(onp.isclose(pts[:, 2], 1.0, atol=1e-8))[0]
    assert onp.abs(w[bottom, 2]).max() == 0.0
    assert onp.abs(w[top, 2]).max() == 0.0
    r = onp.asarray(problem.compute_residual(sol_list)[0]).reshape(-1)
    assert onp.abs(problem.P_mat.T @ r).max() <= 1e-14


@pytest.mark.parametrize("case", ["zero_lateral", "uniaxial_stress"])
@pytest.mark.parametrize("n", [4, 8])
def test_xy_compression_benchmarks(case, n):
    if case == "zero_lateral":
        H = onp.diag(onp.array([0.0, 0.0, -0.01]))
        sigma_diag = onp.array([LAMBDA * -0.01, LAMBDA * -0.01,
                                (LAMBDA + 2 * MU) * -0.01])
        U_exact = 0.5 * (LAMBDA + 2 * MU) * 0.01**2  # uniaxial STRAIN state
    else:
        H = onp.diag(onp.array([0.003, 0.003, -0.01]))
        sigma_diag = onp.array([0.0, 0.0, E * -0.01])
        U_exact = 0.5 * E * 0.01**2

    problem = build_xy(n, H=H)
    sol_list = solve(problem)
    w = onp.asarray(sol_list[0])
    assert onp.abs(w).max() <= 1e-12, "fluctuation must vanish for uniform H"

    u_grad = onp.asarray(problem.fe.sol_to_grad(sol_list[0]))
    eps = 0.5 * (u_grad + onp.swapaxes(u_grad, -1, -2)) + H
    sigma = LAMBDA * onp.trace(eps, axis1=-2, axis2=-1)[..., None, None] * onp.eye(3) \
        + 2.0 * MU * eps
    assert onp.abs(eps - H.reshape(1, 1, 3, 3)).max() <= 1e-12
    assert onp.abs(sigma - sigma_diag.reshape(1, 1, 3, 1)
                   * onp.eye(3).reshape(1, 1, 3, 3)).max() <= 1e-12

    JxW = onp.asarray(problem.JxW)[:, 0, :]
    U = 0.5 * float(onp.sum(sigma * eps * JxW[..., None, None]))
    assert abs(U - U_exact) / U_exact <= 1e-12

    r = onp.asarray(problem.compute_residual(sol_list)[0])  # (num_nodes, 3)
    assert onp.abs(problem.P_mat.T @ r.reshape(-1)).max() <= 1e-12, "reduced equilibrium"

    # axial reactions through the pinned w_z DOFs (sigma_zz * A, A = 1)
    pts = onp.asarray(problem.fe.points)
    top = onp.where(onp.isclose(pts[:, 2], 1.0, atol=1e-8))[0]
    bottom = onp.where(onp.isclose(pts[:, 2], 0.0, atol=1e-8))[0]
    Fz_top = float(r[top, 2].sum())
    Fz_bottom = float(r[bottom, 2].sum())
    assert Fz_top == pytest.approx(float(sigma_diag[2]), abs=1e-10)
    assert Fz_top + Fz_bottom == pytest.approx(0.0, abs=1e-10)

    # total displacement on the loaded faces: u_z = H_zz * z with w_z = 0
    assert onp.abs(w[top, 2]).max() == 0.0 and onp.abs(w[bottom, 2]).max() == 0.0


def test_xy_sine_fluctuation_convergence():
    A = 0.001
    errs = {}
    for n in (4, 8):
        problem = build_xy(n, sine=True)
        sol_list = solve(problem)
        w = onp.asarray(sol_list[0])
        pts = onp.asarray(problem.fe.points)
        w_exact = onp.stack([A * onp.sin(2.0 * onp.pi * pts[:, 0]),
                             onp.zeros(len(pts)), onp.zeros(len(pts))], axis=1)
        errs[n] = onp.abs(w - w_exact).max()
        assert onp.abs(w).max() == pytest.approx(A, rel=0.05)
        assert errs[n] <= {4: 5e-5, 8: 5e-6}[n]
        r = onp.asarray(problem.compute_residual(sol_list)[0]).reshape(-1)
        assert onp.abs(problem.P_mat.T @ r).max() <= 1e-14
    assert errs[8] < errs[4], f"error must decrease on refinement: {errs}"

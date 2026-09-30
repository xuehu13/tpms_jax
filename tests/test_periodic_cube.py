"""M2-B tests: full XYZ periodic boundary conditions on the solid unit cube."""
import numpy as onp
import pytest
import scipy.sparse
import jax
import jax.numpy as jnp

from fem import LAMBDA, MU, solve
from pbc import make_periodic_problem

jax.config.update("jax_enable_x64", True)

# observed sine-fluctuation nodal errors: 4.8e-6 (n=4), 2.7e-7 (n=8)
SINE_TOL = {4: 5e-5, 8: 5e-6}


def test_p_mat_structure():
    n = 4
    problem = make_periodic_problem(n, n, n, H_macro=jnp.zeros((3, 3)))
    P = problem.P_mat
    num_nodes = (n + 1) ** 3
    assert isinstance(P, scipy.sparse.csr_array)
    assert P.shape == (3 * num_nodes, 3 * (n**3 - 1))
    nnz_rows, nnz_cols = onp.nonzero(onp.asarray(P.todense()))
    n_fixed_nodes = int((problem.class_ids == problem.fixed_class_id).sum())
    assert len(nnz_rows) == P.shape[0] - 3 * n_fixed_nodes
    col_counts = onp.bincount(nnz_cols, minlength=P.shape[1])
    assert (col_counts >= 1).all(), "every reduced dof must be used"
    # classes on the periodic boundary own 2 (face), 4 (edge) or 8 (corner)
    # member nodes; consistency of the row budget
    assert len(nnz_rows) + 3 * n_fixed_nodes == P.shape[0]
    row_counts = onp.asarray((P != 0).sum(axis=1)).reshape(-1)
    assert row_counts.max() <= 1
    corner_nodes = onp.where(problem.class_ids == problem.fixed_class_id)[0]
    for comp in range(3):
        assert P[3 * corner_nodes + comp].nnz == 0


def test_periodic_face_pairing_identity():
    """Partner nodes share one reduced DOF, so their w values match exactly."""
    n = 4
    problem = make_periodic_problem(n, n, n, H_macro=jnp.zeros((3, 3)),
                                    sine_force_amplitude=0.001)
    sol_list = solve(problem)
    w = onp.asarray(sol_list[0])
    pts = onp.asarray(problem.fe.points)

    other = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
    idx = onp.round(pts * n).astype(int)
    index_to_node = {(idx[i, 0], idx[i, 1], idx[i, 2]): i for i in range(len(pts))}
    for axis in range(3):
        a1, a2 = other[axis]
        low = onp.where(idx[:, axis] == 0)[0]
        for node_low in low:
            partner_idx = [n if k == axis else idx[node_low, k] for k in range(3)]
            partner = index_to_node[tuple(partner_idx)]
            assert problem.class_ids[partner] == problem.class_ids[node_low]
            assert onp.array_equal(w[node_low], w[partner]), "periodic values differ"


@pytest.mark.parametrize("n", [4, 8])
def test_uniform_macro_strain_recovers_w_zero(n):
    H = onp.diag(onp.array([0.01, 0.02, -0.005]))
    problem = make_periodic_problem(n, n, n, H_macro=jnp.asarray(H))
    sol_list = solve(problem)

    w = onp.asarray(sol_list[0])
    assert onp.abs(w).max() <= 1e-12, "fluctuation must vanish for uniform H"

    r = onp.asarray(problem.compute_residual(sol_list)[0]).reshape(-1)
    assert onp.abs(problem.P_mat.T @ r).max() <= 1e-12, "reduced equilibrium residual"

    u_grad = onp.asarray(problem.fe.sol_to_grad(sol_list[0]))
    eps = 0.5 * (u_grad + onp.swapaxes(u_grad, -1, -2)) + H
    eps_exact = onp.diag(H).reshape(1, 1, 3, 1) * onp.eye(3).reshape(1, 1, 3, 3)
    sigma = LAMBDA * onp.trace(eps, axis1=-2, axis2=-1)[..., None, None] * onp.eye(3) \
        + 2.0 * MU * eps
    sigma_exact = LAMBDA * onp.trace(H) * onp.eye(3) + 2.0 * MU * H
    assert onp.abs(eps - eps_exact).max() <= 1e-12
    assert onp.abs(sigma - sigma_exact[None, None]).max() <= 1e-12
    # independent numeric anchors for the isotropic formula (review finding:
    # sigma_zz = lambda*0.025 + 2*mu*(-0.005) = +0.105769..., not -0.086538)
    assert sigma_exact[0, 0] == pytest.approx(0.2211538461538462, abs=1e-12)
    assert sigma_exact[1, 1] == pytest.approx(0.2980769230769231, abs=1e-12)
    assert sigma_exact[2, 2] == pytest.approx(0.1057692307692308, abs=1e-12)

    JxW = onp.asarray(problem.JxW)[:, 0, :]
    U = 0.5 * float(onp.sum(sigma * eps * JxW[..., None, None]))
    U_exact = 0.5 * (LAMBDA * onp.trace(H) ** 2 + 2.0 * MU * onp.sum(H * H))
    assert abs(U - U_exact) / U_exact <= 1e-12


def test_sine_fluctuation_convergence():
    A = 0.001
    errs = {}
    for n in (4, 8):
        problem = make_periodic_problem(n, n, n, H_macro=jnp.zeros((3, 3)),
                                        sine_force_amplitude=A)
        sol_list = solve(problem)
        w = onp.asarray(sol_list[0])
        pts = onp.asarray(problem.fe.points)
        w_exact = onp.stack([A * onp.sin(2.0 * onp.pi * pts[:, 0]),
                             onp.zeros(len(pts)), onp.zeros(len(pts))], axis=1)
        errs[n] = onp.abs(w - w_exact).max()
        assert onp.abs(w).max() == pytest.approx(A, rel=0.05), "fluctuation amplitude"
        assert errs[n] <= SINE_TOL[n]
        r = onp.asarray(problem.compute_residual(sol_list)[0]).reshape(-1)
        assert onp.abs(problem.P_mat.T @ r).max() <= 1e-14
    assert errs[8] < errs[4], f"error must decrease on refinement: {errs}"

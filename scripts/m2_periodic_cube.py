"""M2-B run entry: periodic unit-cell verification (uniform H + sine wave).

Run from the project root:  pixi run python scripts/m2_periodic_cube.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as onp
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from fem import LAMBDA, MU, solve
from jax_fem.utils import save_sol
from pbc import make_periodic_problem

A_SINE = 0.001


def main():
    print("== M2-B: full XYZ periodic unit cell ==")

    H = onp.diag(onp.array([0.01, 0.02, -0.005]))
    for n in (4, 8):
        problem = make_periodic_problem(n, n, n, H_macro=jnp.asarray(H))
        sol_list = solve(problem)
        w = onp.asarray(sol_list[0])
        r = onp.asarray(problem.compute_residual(sol_list)[0]).reshape(-1)
        r_red = problem.P_mat.T @ r
        u_grad = onp.asarray(problem.fe.sol_to_grad(sol_list[0]))
        eps = 0.5 * (u_grad + onp.swapaxes(u_grad, -1, -2)) + H
        sigma = LAMBDA * onp.trace(eps, axis1=-2, axis2=-1)[..., None, None] * onp.eye(3) \
            + 2.0 * MU * eps
        sigma_exact = LAMBDA * onp.trace(H) * onp.eye(3) + 2.0 * MU * H
        JxW = onp.asarray(problem.JxW)[:, 0, :]
        U = 0.5 * float(onp.sum(sigma * eps * JxW[..., None, None]))
        U_exact = 0.5 * (LAMBDA * onp.trace(H) ** 2 + 2.0 * MU * onp.sum(H * H))
        print(f"[A] n={n}: full dofs={problem.P_mat.shape[0]}, reduced={problem.P_mat.shape[1]}, "
              f"max|w|={onp.abs(w).max():.3e}, sigma err={onp.abs(sigma - sigma_exact[None, None]).max():.3e}, "
              f"U={U:.10f} (exact {U_exact:.10f}), red res={onp.abs(r_red).max():.3e}")

    print(f"[B] sine fluctuation w_x = {A_SINE} sin(2 pi x)")
    prev = None
    for n in (4, 8, 16):
        problem = make_periodic_problem(n, n, n, H_macro=jnp.zeros((3, 3)),
                                        sine_force_amplitude=A_SINE)
        sol_list = solve(problem)
        w = onp.asarray(sol_list[0])
        pts = onp.asarray(problem.fe.points)
        w_exact = onp.stack([A_SINE * onp.sin(2.0 * onp.pi * pts[:, 0]),
                             onp.zeros(len(pts)), onp.zeros(len(pts))], axis=1)
        err = onp.abs(w - w_exact).max()
        msg = f"[B] n={n:3d}: max abs w={onp.abs(w).max():.6f}, err={err:.3e}"
        if prev:
            msg += f", ratio={prev / err:.1f}"
        print(msg)
        prev = err

    n = 8
    problem = make_periodic_problem(n, n, n, H_macro=jnp.zeros((3, 3)),
                                    sine_force_amplitude=A_SINE)
    sol_list = solve(problem)
    os.makedirs("results", exist_ok=True)
    out = os.path.join("results", f"m2_periodic_sine_N{n}.vtu")
    save_sol(problem.fe, sol_list[0], out,
             point_infos=[("w", sol_list[0])])
    print(f"wrote {out} (nodal periodic fluctuation w)")


if __name__ == "__main__":
    main()

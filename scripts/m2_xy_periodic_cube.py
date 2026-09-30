"""M2-C run entry: XY-periodic solid cube under axial compression.

Run from the project root:  pixi run python scripts/m2_xy_periodic_cube.py
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
from pbc import make_periodic_problem, xy_compression_fixed_dofs

CASES = {
    "A_zero_lateral": (onp.diag(onp.array([0.0, 0.0, -0.01])),
                       onp.array([LAMBDA * -0.01, LAMBDA * -0.01, (LAMBDA + 2 * MU) * -0.01]),
                       0.5 * (LAMBDA + 2 * MU) * 0.01**2),
    "B_uniaxial_stress": (onp.diag(onp.array([0.003, 0.003, -0.01])),
                          onp.array([0.0, 0.0, -0.1]),
                          0.5 * 10.0 * 0.01**2),
}


def main():
    print("== M2-C: XY-periodic solid cube, 1% axial compression ==")
    for name, (H, sigma_diag, U_exact) in CASES.items():
        for n in (4, 8):
            problem = make_periodic_problem(
                n, n, n, H_macro=jnp.asarray(H), periodic_axes=(0, 1),
                fixed_class=None,
                fixed_dofs=lambda p: xy_compression_fixed_dofs(p, n, n, n))
            sol_list = solve(problem)
            w = onp.asarray(sol_list[0])
            r = onp.asarray(problem.compute_residual(sol_list)[0])
            pts = onp.asarray(problem.fe.points)
            top = onp.where(onp.isclose(pts[:, 2], 1.0, atol=1e-8))[0]
            bottom = onp.where(onp.isclose(pts[:, 2], 0.0, atol=1e-8))[0]
            Fz_top = float(r[top, 2].sum())
            Fz_bottom = float(r[bottom, 2].sum())
            u_grad = onp.asarray(problem.fe.sol_to_grad(sol_list[0]))
            eps = 0.5 * (u_grad + onp.swapaxes(u_grad, -1, -2)) + H
            sigma = LAMBDA * onp.trace(eps, axis1=-2, axis2=-1)[..., None, None] * onp.eye(3) \
                + 2.0 * MU * eps
            JxW = onp.asarray(problem.JxW)[:, 0, :]
            U = 0.5 * float(onp.sum(sigma * eps * JxW[..., None, None]))
            print(f"[{name}] n={n}: dofs {problem.P_mat.shape[0]}->{problem.P_mat.shape[1]}, "
                  f"max abs w={onp.abs(w).max():.3e}")
            print(f"    sigma_xx={float(sigma[..., 0, 0].mean()):+.8f} "
                  f"(exact {sigma_diag[0]:+.8f}), "
                  f"sigma_zz={float(sigma[..., 2, 2].mean()):+.8f} "
                  f"(exact {sigma_diag[2]:+.8f})")
            print(f"    U={U:.12f} (exact {U_exact:.12f}), "
                  f"Fz_top={Fz_top:+.8f} (exact {sigma_diag[2]:+.8f}), "
                  f"balance={Fz_top + Fz_bottom:+.2e}, "
                  f"red res={onp.abs(problem.P_mat.T @ r.reshape(-1)).max():.3e}")

    print("\n[sine] XY-periodic sine fluctuation w_x = 0.001 sin(2 pi x)")
    prev = None
    for n in (4, 8, 16):
        problem = make_periodic_problem(
            n, n, n, H_macro=jnp.zeros((3, 3)), sine_force_amplitude=0.001,
            periodic_axes=(0, 1), fixed_class=None,
            fixed_dofs=lambda p: xy_compression_fixed_dofs(p, n, n, n))
        sol_list = solve(problem)
        w = onp.asarray(sol_list[0])
        pts = onp.asarray(problem.fe.points)
        w_exact = onp.stack([0.001 * onp.sin(2.0 * onp.pi * pts[:, 0]),
                             onp.zeros(len(pts)), onp.zeros(len(pts))], axis=1)
        err = onp.abs(w - w_exact).max()
        msg = f"[sine] n={n:3d}: max abs w={onp.abs(w).max():.6f}, err={err:.3e}"
        if prev:
            msg += f", ratio={prev / err:.1f}"
        print(msg)
        prev = err

    n = 8
    H, _, _ = CASES["B_uniaxial_stress"]
    problem = make_periodic_problem(
        n, n, n, H_macro=jnp.asarray(H), periodic_axes=(0, 1), fixed_class=None,
        fixed_dofs=lambda p: xy_compression_fixed_dofs(p, n, n, n))
    sol_list = solve(problem)
    os.makedirs("results", exist_ok=True)
    out = os.path.join("results", f"m2_xy_uniaxial_N{n}.vtu")
    pts = jnp.asarray(problem.fe.points)
    total_u = jnp.asarray(sol_list[0]) + (jnp.asarray(H) @ pts.T).T
    save_sol(problem.fe, sol_list[0], out,
             point_infos=[("w", sol_list[0]), ("u_total", total_u)])
    print(f"wrote {out} (nodal w and total u = H X + w)")


if __name__ == "__main__":
    main()

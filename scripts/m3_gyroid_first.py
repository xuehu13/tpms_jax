"""M3-A run entry: first Gyroid linear-elastic calculation (XY periodic).

Run from the project root:  pixi run python scripts/m3_gyroid_first.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as onp
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from density_fem import make_density_problem
from fem import solve
from geometry import density
from jax_fem.utils import save_sol
from pbc import xy_compression_fixed_dofs

C_CALIB = 0.541062
BETA = 20.0
H = onp.diag(onp.array([0.0, 0.0, -0.01]))
RESOLUTIONS = [4, 8, 16]


def main():
    print("== M3-A: first Gyroid calculation (XY periodic, 1% axial "
          "compression, c=0.541062, beta=20) ==")
    for n in RESOLUTIONS:
        t0 = time.time()
        problem = make_density_problem(
            n, n, n, H_macro=jnp.asarray(H), rho_quad=None,
            periodic_axes=(0, 1), fixed_class=None,
            fixed_dofs=lambda pts: xy_compression_fixed_dofs(pts, n, n, n))
        rho = density(problem.physical_quad_points, C_CALIB, BETA, 1.0)
        problem.set_params(H, rho)
        t1 = time.time()
        sol_list = solve(problem)
        t2 = time.time()

        rho_o = onp.asarray(rho)
        JxW = onp.asarray(problem.JxW)[:, 0, :]
        vf_int = float((rho_o * JxW).sum() / JxW.sum())
        lam = onp.asarray(problem.internal_vars[1])[..., None, None]
        mu = onp.asarray(problem.internal_vars[2])[..., None, None]
        u_grad = onp.asarray(problem.fe.sol_to_grad(sol_list[0]))
        eps = 0.5 * (u_grad + onp.swapaxes(u_grad, -1, -2)) + H
        sigma = lam * onp.trace(eps, axis1=-2, axis2=-1)[..., None, None] * onp.eye(3) \
            + 2.0 * mu * eps
        w = onp.asarray(sol_list[0])
        r = onp.asarray(problem.compute_residual(sol_list)[0])
        pts = onp.asarray(problem.fe.points)
        top = onp.where(onp.isclose(pts[:, 2], 1.0, atol=1e-8))[0]
        Fz = float(r[top, 2].sum())
        r_red = onp.abs(problem.P_mat.T @ r.reshape(-1)).max()
        print(f"n={n:3d}: cells={n**3}, quads={n**3*8}, "
              f"rho range=[{rho_o.min():.4f}, {rho_o.max():.4f}], "
              f"Vf_int={vf_int:.4f}, Fz={Fz:+.6f}, "
              f"sigma_zz_avg={float(sigma[..., 2, 2].mean()):+.6f}, "
              f"U={float(onp.sum(sigma * eps * JxW[..., None, None]) * 0.5):.6e}, "
              f"red res={r_red:.2e}, setup={t1-t0:.1f}s, solve={t2-t1:.1f}s")

        if n == 8:
            sigma_zz_cell = onp.sum(sigma[..., 2, 2] * JxW, axis=1) / onp.sum(JxW, axis=1)
            rho_cell = onp.sum(rho_o * JxW, axis=1) / onp.sum(JxW, axis=1)
            pts_j = jnp.asarray(problem.fe.points)
            total_u = jnp.asarray(sol_list[0]) + (jnp.asarray(H) @ pts_j.T).T
            os.makedirs("results", exist_ok=True)
            out = os.path.join("results", f"m3_gyroid_N{n}.vtu")
            save_sol(problem.fe, sol_list[0], out,
                     point_infos=[("w", sol_list[0]), ("u_total", total_u)],
                     cell_infos=[("rho_cell_avg", jnp.asarray(rho_cell)),
                                 ("sigma_zz_cell_avg", jnp.asarray(sigma_zz_cell))])
            print(f"    wrote {out} (nodal w/u_total; JxW-weighted cell averages of "
                  "the integration-point density and sigma_zz; the computed field "
                  "remains the raw integration-point density)")

if __name__ == "__main__":
    main()

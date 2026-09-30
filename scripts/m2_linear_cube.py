"""M2-A run entry: full-solid HEX8 linear elasticity benchmarks.

Run from the project root:  pixi run python scripts/m2_linear_cube.py
Writes one VTU of the uniaxial case (meshio via jax_fem.utils.save_sol).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as onp
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from fem import (E, LAMBDA, MU, NU, affine_bc, face_nodes, hex_jacobian_check,
                 internal_force, make_cube_problem, solve, strain_energy,
                 uniaxial_bc)
from jax_fem.utils import save_sol

RESOLUTIONS = [4, 8]
DELTA = -0.01


def main():
    print("== M2-A: full-solid HEX8 linear elasticity, unit cube ==")
    print(f"E={E}, nu={NU}, lambda={LAMBDA:.6f}, mu={MU:.6f}")

    for n in RESOLUTIONS:
        print(f"\n--- Benchmark A: affine u=(ex*x, ey*y, ez*z), "
              f"(ex, ey, ez)=(0.01, 0.02, -0.005), n={n} ---")
        ex, ey, ez = 0.01, 0.02, -0.005
        problem = make_cube_problem(n, n, n, affine_bc(ex, ey, ez))
        min_det, vol = hex_jacobian_check(problem)
        sol_list = solve(problem)
        pts = onp.asarray(problem.fe.points)
        u = onp.asarray(sol_list[0])
        u_exact = onp.stack([ex * pts[:, 0], ey * pts[:, 1], ez * pts[:, 2]], axis=1)
        print(f"cells={n**3}, nodes={(n+1)**3}, min detJ={min_det:.3e}, total volume={vol:.6f}")
        print(f"max |u - u_affine|          = {onp.abs(u - u_exact).max():.3e}")
        U = strain_energy(problem, sol_list)
        U_exact = 0.5 * LAMBDA * (ex + ey + ez) ** 2 + MU * (ex**2 + ey**2 + ez**2)
        print(f"strain energy               = {U:.8f} (exact {U_exact:.8f})")
        r = internal_force(problem, sol_list)
        onb = (onp.isclose(pts, 0.0, atol=1e-8) | onp.isclose(pts, 1.0, atol=1e-8)).any(axis=1)
        print(f"max |residual| at free DOFs = {onp.abs(r[~onb]).max():.3e}")
        rx = float(r[face_nodes(problem, 0, 1.0), 0].sum())
        sigma_xx = LAMBDA * (ex + ey + ez) + 2.0 * MU * ex
        print(f"+x face reaction sum        = {rx:+.6f} (sigma_xx exact {sigma_xx:+.6f})")

        print(f"--- Benchmark B: uniaxial compression delta={DELTA}, n={n} ---")
        pb = make_cube_problem(n, n, n, uniaxial_bc(DELTA))
        sol_b = solve(pb)
        pts_b = onp.asarray(pb.fe.points)
        u_b = onp.asarray(sol_b[0])
        u_exact_b = onp.stack([-NU * DELTA * pts_b[:, 0], -NU * DELTA * pts_b[:, 1],
                               DELTA * pts_b[:, 2]], axis=1)
        print(f"max |u - u_uniaxial|        = {onp.abs(u_b - u_exact_b).max():.3e}")
        r_b = internal_force(pb, sol_b)
        Fz_top = float(r_b[face_nodes(pb, 2, 1.0), 2].sum())
        Fz_bottom = float(r_b[face_nodes(pb, 2, 0.0), 2].sum())
        print(f"top reaction Fz             = {Fz_top:+.6f} (exact {E * DELTA:+.6f})")
        print(f"bottom reaction Fz          = {Fz_bottom:+.6f} (balance {Fz_top + Fz_bottom:+.2e})")
        Ub = strain_energy(pb, sol_b)
        Ub_exact = 0.5 * E * DELTA**2
        print(f"strain energy               = {Ub:.8f} (exact {Ub_exact:.8f})")
        ux_side = u_b[face_nodes(pb, 0, 1.0), 0]
        print(f"lateral u_x at x=1 face     = max err {onp.abs(ux_side - (-NU * DELTA)).max():.3e}"
              f" (exact {-NU * DELTA:+.4f})")

    n = RESOLUTIONS[-1]
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    pb = make_cube_problem(n, n, n, uniaxial_bc(DELTA))
    sol_b = solve(pb)
    u_grad = pb.fe.sol_to_grad(sol_b[0])
    eps = 0.5 * (u_grad + jnp.swapaxes(u_grad, -1, -2))
    sigma = LAMBDA * jnp.trace(eps, axis1=-2, axis2=-1)[..., None, None] * jnp.eye(3) \
        + 2.0 * MU * eps
    JxW = pb.JxW[:, 0, :]
    sigma_zz_cell = jnp.sum(sigma[..., 2, 2] * JxW, axis=1) / jnp.sum(JxW, axis=1)
    out = os.path.join(out_dir, f"m2_uniaxial_N{n}.vtu")
    save_sol(pb.fe, sol_b[0], out,
             cell_infos=[("sigma_zz", sigma_zz_cell),
                         ("sigma_zz_exact", jnp.full_like(sigma_zz_cell, E * DELTA))])
    print(f"wrote {out} (uniaxial case, n={n}: nodal u + cell sigma_zz)")


if __name__ == "__main__":
    main()

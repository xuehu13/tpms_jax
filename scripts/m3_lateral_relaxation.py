"""M3-B run entry: automatic lateral relaxation of the periodic unit cell.

Run from the project root:  pixi run python scripts/m3_lateral_relaxation.py
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

from fem import solve
from density_fem import solve_lateral_relaxation
from geometry import density
from pbc import xy_compression_fixed_dofs

C_CALIB = 0.541062
BETA = 20.0


def main():
    print("== M3-B: automatic lateral relaxation to zero average lateral stress ==")

    print("\n[uniform] rho = 1 (E_s = 10), analytic eps_x = eps_y = +0.003")
    for n in (4, 8):
        t0 = time.time()
        out = solve_lateral_relaxation(
            n, n, n, rho_quad=1.0, eps_z=-0.01,
            fixed_dofs=lambda pts: xy_compression_fixed_dofs(pts, n, n, n))
        s = out["sigma_avg"]
        r = onp.asarray(out["problem"].compute_residual(out["sol_list"])[0])
        pts = onp.asarray(out["problem"].fe.points)
        top = onp.where(onp.isclose(pts[:, 2], 1.0, atol=1e-8))[0]
        Fz = float(r[top, 2].sum())
        U = 0.5 * float(s[2, 2] * -0.01)
        print(f"n={n}: eps_x={out['eps_x']:+.10f}, eps_y={out['eps_y']:+.10f}, "
              f"sxx={s[0, 0]:+.2e}, syy={s[1, 1]:+.2e}, szz={s[2, 2]:+.10f}, "
              f"Fz={Fz:+.10f}, U={U:.10f} (exact 5.0e-4), t={time.time()-t0:.1f}s")

    print(f"\n[gyroid] c={C_CALIB}, beta={BETA}: lateral-fixed vs lateral-relaxed")
    H_fixed = onp.diag(onp.array([0.0, 0.0, -0.01]))
    for n in (8, 16):
        t0 = time.time()
        pins = lambda pts: xy_compression_fixed_dofs(pts, n, n, n)
        out = solve_lateral_relaxation(n, n, n, rho_quad=rho_fn, eps_z=-0.01,
                                       fixed_dofs=pins)
        t1 = time.time()

        from density_fem import avg_stress, make_density_problem
        problem_a = make_density_problem(
            n, n, n, H_macro=jnp.asarray(H_fixed), rho_quad=rho_fn,
            periodic_axes=(0, 1), fixed_class=None, fixed_dofs=pins)
        sol_a = solve(problem_a)
        s_fixed = avg_stress(problem_a, sol_a)
        r_a = onp.asarray(problem_a.compute_residual(sol_a)[0])
        pts = onp.asarray(problem_a.fe.points)
        top = onp.where(onp.isclose(pts[:, 2], 1.0, atol=1e-8))[0]
        Fz_fixed = float(r_a[top, 2].sum())

        s_b = out["sigma_avg"]
        r_b = onp.asarray(out["problem"].compute_residual(out["sol_list"])[0])
        Fz_relaxed = float(r_b[top, 2].sum())
        JxW = onp.asarray(problem_a.JxW)[:, 0, :]
        rho_o = onp.asarray(rho)
        vf = float((rho_o * JxW).sum() / JxW.sum())
        s_b_diag = onp.diag(s_b)
        s_a_diag = onp.diag(s_fixed)
        print(f"n={n}: Vf_int={vf:.4f}")
        print(f"  fixed  : sxx={s_a_diag[0]:+.6f} syy={s_a_diag[1]:+.6f} szz={s_a_diag[2]:+.6f}, "
              f"shear max={onp.abs(s_fixed - onp.diag(s_a_diag)).max():.2e}, Fz={Fz_fixed:+.6f}")
        print(f"  relaxed: eps_x={out['eps_x']:+.6f}, eps_y={out['eps_y']:+.6f}, "
              f"sxx={s_b_diag[0]:+.2e} syy={s_b_diag[1]:+.2e} szz={s_b_diag[2]:+.6f}, "
              f"shear max={onp.abs(s_b - onp.diag(s_b_diag)).max():.2e}, Fz={Fz_relaxed:+.6f}")
        print(f"  stiffness |Fz_relaxed|/|Fz_fixed| = {abs(Fz_relaxed)/abs(Fz_fixed):.4f}, "
              f"relaxation total t={time.time()-t0:.1f}s")


def rho_fn(problem):
    return density(problem.physical_quad_points, C_CALIB, BETA, 1.0)


def make_fixed_problem(n, pins):
    from density_fem import make_density_problem
    H_fixed = onp.diag(onp.array([0.0, 0.0, -0.01]))
    return make_density_problem(
        n, n, n, H_macro=jnp.asarray(H_fixed), rho_quad=rho_fn,
        periodic_axes=(0, 1), fixed_class=None, fixed_dofs=pins)


if __name__ == "__main__":
    main()

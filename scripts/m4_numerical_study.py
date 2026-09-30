"""M4-A run entry: numerical sensitivity study for the fixed-grid Gyroid.

Run from the project root:  pixi run python scripts/m4_numerical_study.py
Writes results/m4_numerical_study.csv. Studies: mesh resolution (N, both
lateral conditions), beta projection sharpness, E_min virtual porosity.
"""
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as onp
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from density_fem import make_density_problem, solve_lateral_relaxation
from fem import solve
from geometry import density, gyroid
from pbc import xy_compression_fixed_dofs

C_CALIB = 0.541062
E_S = 10.0
EPS_Z = -0.01
CSV_PATH = os.path.join("results", "m4_numerical_study.csv")
FIELDS = ["study", "N", "beta", "emin_ratio", "lateral", "cells", "quads",
          "n_red", "vf_int", "vf_binary_ref", "Fz_top", "Fz_bottom",
          "sigma_xx", "sigma_yy", "sigma_zz", "U", "eps_x", "eps_y",
          "red_res", "balance", "work_identity_err", "t_build", "t_solve",
          "status"]


def solve_case(N, beta, emin_ratio, lateral):
    """Build once, solve once (fixed) or run the 4-solve relaxation.

    Returns (problem, sol_list, H_used, eps_x, eps_y, t_build, t_solve).
    """
    E_min = emin_ratio * E_S
    H_fixed = onp.diag(onp.array([0.0, 0.0, EPS_Z]))
    pins = lambda pts: xy_compression_fixed_dofs(pts, N, N, N)

    if lateral == "fixed":
        t0 = time.perf_counter()
        problem = make_density_problem(
            N, N, N, H_macro=jnp.asarray(H_fixed), rho_quad=None,
            E_min=E_min, periodic_axes=(0, 1), fixed_class=None,
            fixed_dofs=pins)
        rho = density(problem.physical_quad_points, C_CALIB, beta, 1.0)
        problem.set_params(H_fixed, rho, E_S, E_min)
        t1 = time.perf_counter()
        sol_list = solve(problem)
        t2 = time.perf_counter()
        return problem, sol_list, H_fixed, 0.0, 0.0, t1 - t0, t2 - t1

    # relaxed: 4 solves (H0, Hx, Hy probes + final) on one instance
    t0 = time.perf_counter()
    probe = make_density_problem(
        N, N, N, H_macro=jnp.zeros((3, 3)), rho_quad=None,
        E_min=E_min, periodic_axes=(0, 1), fixed_class=None, fixed_dofs=pins)
    rho = density(probe.physical_quad_points, C_CALIB, beta, 1.0)
    t1 = time.perf_counter()
    out = solve_lateral_relaxation(
        N, N, N, rho_quad=rho, eps_z=EPS_Z, h=0.01, E_s=E_S, E_min=E_min,
        cell_size=1.0, periodic_axes=(0, 1), fixed_class=None,
        fixed_dofs=pins)
    t2 = time.perf_counter()
    return (out["problem"], out["sol_list"], out["H_final"],
            out["eps_x"], out["eps_y"], (t1 - t0), (t2 - t1))


def measure(problem, sol_list, H_used):
    r = onp.asarray(problem.compute_residual(sol_list)[0])
    pts = onp.asarray(problem.fe.points)
    top = onp.where(onp.isclose(pts[:, 2], 1.0, atol=1e-8))[0]
    bottom = onp.where(onp.isclose(pts[:, 2], 0.0, atol=1e-8))[0]
    Fz_top = float(r[top, 2].sum())
    Fz_bottom = float(r[bottom, 2].sum())
    u_grad = onp.asarray(problem.fe.sol_to_grad(sol_list[0]))
    eps = 0.5 * (u_grad + onp.swapaxes(u_grad, -1, -2)) + onp.asarray(problem.H_macro)
    lam = onp.asarray(problem.internal_vars[1])[..., None, None]
    mu = onp.asarray(problem.internal_vars[2])[..., None, None]
    sigma = lam * onp.trace(eps, axis1=-2, axis2=-1)[..., None, None] * onp.eye(3) \
        + 2.0 * mu * eps
    JxW = onp.asarray(problem.JxW)[:, 0, :]
    s_avg = onp.sum(sigma * JxW[..., None, None], axis=(0, 1)) / JxW.sum()
    U = 0.5 * float(onp.sum(sigma * eps * JxW[..., None, None]))
    r_red = onp.abs(problem.P_mat.T @ r.reshape(-1)).max()
    # macro work identity: U == 0.5 * sigma_avg : eps_avg * V (V = 1)
    work = 0.5 * float(onp.sum(s_avg * eps.mean(axis=(0, 1))))
    return Fz_top, Fz_bottom, s_avg, U, r_red, abs(U - work)


def run_row(study, N, beta, emin_ratio, lateral):
    row = dict(study=study, N=N, beta=beta, emin_ratio=emin_ratio,
               lateral=lateral, status="ok")
    try:
        problem, sol_list, H_used, eps_x, eps_y, t_build, t_solve = \
            solve_case(N, beta, emin_ratio, lateral)
        (row["Fz_top"], row["Fz_bottom"], s_avg, row["U"], row["red_res"],
         row["work_identity_err"]) = measure(problem, sol_list, H_used)
        row["cells"] = problem.fe.num_cells
        row["quads"] = problem.fe.num_cells * problem.fe.num_quads
        row["n_red"] = problem.P_mat.shape[1]
        rho_o = onp.asarray(problem.rho)
        JxW = onp.asarray(problem.JxW)[:, 0, :]
        row["vf_int"] = float((rho_o * JxW).sum() / JxW.sum())
        X_q = onp.asarray(problem.physical_quad_points)
        row["vf_binary_ref"] = float(
            onp.mean(onp.abs(onp.asarray(gyroid(X_q))) <= C_CALIB))
        row["sigma_xx"], row["sigma_yy"], row["sigma_zz"] = \
            float(s_avg[0, 0]), float(s_avg[1, 1]), float(s_avg[2, 2])
        row["eps_x"], row["eps_y"] = float(eps_x), float(eps_y)
        row["balance"] = row["Fz_top"] + row["Fz_bottom"]
        row["t_build"], row["t_solve"] = round(t_build, 2), round(t_solve, 2)
    except Exception as exc:  # record the actual failure, do not mask it
        row["status"] = f"FAILED: {type(exc).__name__}: {exc}"
    return row


def main():
    os.makedirs("results", exist_ok=True)
    rows = []

    print("== study 1: mesh resolution (beta=20, E_min/E_s=1e-3) ==")
    prev = {}
    for N in (8, 16, 24, 32):
        for lateral in ("fixed", "relaxed"):
            row = run_row("mesh", N, 20.0, 1e-3, lateral)
            rows.append(row)
            print(f"  N={N:3d} {lateral:8s}: {row['status']}", flush=True)
            if row["status"] != "ok":
                print("    -> stop refining after a failed mesh", flush=True)
                break_outer = True
            else:
                print(f"    Vf={row['vf_int']:.4f} Fz={row['Fz_top']:+.6f} "
                      f"U={row['U']:.4e} red_res={row['red_res']:.1e} "
                      f"t_build={row['t_build']}s t_solve={row['t_solve']}s", flush=True)
                if lateral == "relaxed":
                    prev[N] = row["Fz_top"]

    print("\n== study 2: beta sensitivity (N=16, E_min/E_s=1e-3, lateral fixed) ==")
    for beta in (10.0, 20.0, 40.0):
        row = run_row("beta", 16, beta, 1e-3, "fixed")
        rows.append(row)
        print(f"  beta={beta:5.1f}: Vf_int={row.get('vf_int', float('nan')):.4f} "
              f"Vf_bin={row.get('vf_binary_ref', float('nan')):.4f} "
              f"Fz={row.get('Fz_top', float('nan')):+.6f} "
              f"U={row.get('U', float('nan')):.4e} status={row['status']}", flush=True)

    print("\n== study 3: E_min sensitivity (N=16, beta=20, lateral fixed) ==")
    for ratio in (1e-2, 1e-3, 1e-4):
        row = run_row("emin", 16, 20.0, ratio, "fixed")
        rows.append(row)
        print(f"  E_min/E_s={ratio:.0e}: Fz={row.get('Fz_top', float('nan')):+.6f} "
              f"U={row.get('U', float('nan')):.4e} red_res={row.get('red_res', float('nan')):.1e} "
              f"t_solve={row.get('t_solve')}s status={row['status']}", flush=True)

    with open(CSV_PATH, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {CSV_PATH} ({len(rows)} rows)")

    if prev:
        print("\n== adjacent-mesh change rate of |Fz| (relaxed, vs finest) ==")
        finest = max(prev)
        for N in sorted(prev):
            rate = abs(prev[N] - prev[finest]) / abs(prev[finest])
        print(f"  N={N:3d}: |Fz-Fz_finest|/|Fz_finest| = {rate:.4f} "
              "(adjacent-refinement indicator, NOT a true physical error)")


if __name__ == "__main__":
    main()

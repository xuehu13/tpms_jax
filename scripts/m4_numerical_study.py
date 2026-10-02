"""M4-A: numerical sensitivity study for the fixed-grid Gyroid (revised).

Run from the project root:  pixi run python scripts/m4_numerical_study.py

Studies (lateral macro strain FIXED unless stated otherwise):
  1. N x beta coupling  : N in {16, 24, 32} x beta in {10, 20, 40}
  2. E_min verification : N=32, beta=20, E_min/E_s = 1e-4 vs 1e-3

Every case records the FEM-integrated volume fraction, the binary
reference fraction on the SAME quadrature points, reactions, JxW-weighted
average stresses, strain energy, the macroscopic work identity
U_internal vs 0.5*V*sigma_avg:sym(H_used), reduced residual, force
balance and build/solve timings. status=ok means the case computed AND
passed the numerical consistency checks (finite values, residuals,
balance, reaction consistency, work identity) with tolerances matched to
the observed solver accuracy -- not merely "the solver returned".
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

from density_fem import make_density_problem, solve_lateral_relaxation, calibrate_density_c
from fem import solve
from geometry import density, gyroid
from pbc import xy_compression_fixed_dofs

C_CALIB = 0.541062
E_S = 10.0
EPS_Z = -0.01
CSV_PATH = os.path.join("results", "m4_numerical_study.csv")
FIELDS = ["study", "N", "beta", "emin_ratio", "lateral", "cells", "quads",
          "n_red", "vf_int", "vf_binary_ref", "Fz_top", "Fz_bottom",
          "sigma_xx", "sigma_yy", "sigma_zz", "U_internal", "U_macro",
          "eps_x", "eps_y", "red_res", "balance",
          "reaction_consistency_err", "work_identity_err",
          "t_build", "t_solve", "checks", "status"]

TOL_RES = 1e-8
TOL_BALANCE = 1e-8
TOL_REACTION = 1e-6
TOL_WORK = 1e-8


def rho_fn_factory(beta, c=C_CALIB, target_vf=None):
    """Density on the TRUE Gauss points of whichever problem asks for it."""
    def rho_fn(problem):
        record = None if target_vf is None else calibrate_density_c(problem, beta, target_vf)
        problem.projection_c = c if record is None else record["c"]
        problem.projection_calibration = record
        return density(problem.physical_quad_points, problem.projection_c, beta, 1.0)
    return rho_fn


def solve_case(N, beta, emin_ratio, lateral, solver_options=None,
               c=C_CALIB, target_vf=None):
    """One build + one solve (fixed) or one build + four solves (relaxed).

    The relaxed path creates NO temporary problem: the callable rho_quad is
    evaluated inside solve_lateral_relaxation on the real
    physical_quad_points of the single Problem instance it builds.
    """
    E_min = emin_ratio * E_S
    H_fixed = onp.diag(onp.array([0.0, 0.0, EPS_Z]))
    pins = lambda pts: xy_compression_fixed_dofs(pts, N, N, N)
    if lateral == "fixed":
        t0 = time.perf_counter()
        problem = make_density_problem(
            N, N, N, H_macro=jnp.asarray(H_fixed), rho_quad=rho_fn_factory(beta, c, target_vf),
            E_min=E_min, periodic_axes=(0, 1), fixed_class=None,
            fixed_dofs=pins)
        t1 = time.perf_counter()
        sol_list = solve(problem) if solver_options is None else solve(problem, solver_options)
        t2 = time.perf_counter()
        return problem, sol_list, H_fixed, 0.0, 0.0, t1 - t0, t2 - t1
    extra_options = {} if solver_options is None else {"solver_options": solver_options}
    out = solve_lateral_relaxation(
        N, N, N, rho_quad=rho_fn_factory(beta, c, target_vf), eps_z=EPS_Z, h=0.01,
        E_s=E_S, E_min=E_min, cell_size=1.0, periodic_axes=(0, 1),
        fixed_class=None, fixed_dofs=pins, **extra_options)
    return (out["problem"], out["sol_list"], out["H_final"],
            out["eps_x"], out["eps_y"], out["t_build"], out["t_solve"])


def evaluate_case(N, beta, emin_ratio, lateral,
                  tol_res=TOL_RES, tol_balance=TOL_BALANCE,
                  tol_reaction=TOL_REACTION, tol_work=TOL_WORK,
                  include_problem=False, solver_options=None,
                  c=C_CALIB, target_vf=None):
    """Run one case and apply the numerical consistency checks.

    status=ok requires finite values AND the residual/balance/reaction/
    work-identity checks within tolerances; check_failed means computed
    but at least one check failed; FAILED means an exception.
    """
    row = dict(study="", N=N, beta=beta, emin_ratio=emin_ratio,
               lateral=lateral, status="")
    try:
        options = {} if solver_options is None else {"solver_options": solver_options}
        if c != C_CALIB or target_vf is not None:
            options.update(c=c, target_vf=target_vf)
        problem, sol_list, H_used, eps_x, eps_y, t_build, t_solve = \
            solve_case(N, beta, emin_ratio, lateral, **options)
        r = onp.asarray(problem.compute_residual(sol_list)[0])
        pts = onp.asarray(problem.fe.points)
        top = onp.where(onp.isclose(pts[:, 2], 1.0, atol=1e-8))[0]
        bottom = onp.where(onp.isclose(pts[:, 2], 0.0, atol=1e-8))[0]
        Fz_top = float(r[top, 2].sum())
        Fz_bottom = float(r[bottom, 2].sum())

        H = onp.asarray(H_used)
        u_grad = onp.asarray(problem.fe.sol_to_grad(sol_list[0]))
        eps = 0.5 * (u_grad + onp.swapaxes(u_grad, -1, -2)) + 0.5 * (H + H.T)
        lam = onp.asarray(problem.internal_vars[1])[..., None, None]
        mu = onp.asarray(problem.internal_vars[2])[..., None, None]
        sigma = lam * onp.trace(eps, axis1=-2, axis2=-1)[..., None, None] \
            * onp.eye(3) + 2.0 * mu * eps
        JxW = onp.asarray(problem.JxW)[:, 0, :]
        V = float(JxW.sum())
        s_avg = onp.sum(sigma * JxW[..., None, None], axis=(0, 1)) / V
        U_internal = 0.5 * float(onp.sum(sigma * eps * JxW[..., None, None]))
        # macro work identity with the ACTUAL imposed gradient
        eps_macro = 0.5 * (H + H.T)
        U_macro = 0.5 * V * float(onp.sum(s_avg * eps_macro))

        rho_o = onp.asarray(problem.rho)
        X_q = onp.asarray(problem.physical_quad_points)
        vf_binary_ref = float(onp.sum(
            (onp.abs(onp.asarray(gyroid(X_q))) <= problem.projection_c) * JxW) / V)

        row.update(cells=problem.fe.num_cells,
                   quads=problem.fe.num_cells * problem.fe.num_quads,
                   n_red=problem.P_mat.shape[1],
                   vf_int=float((rho_o * JxW).sum() / V),
                   vf_binary_ref=vf_binary_ref,
                   Fz_top=Fz_top, Fz_bottom=Fz_bottom,
                   sigma_xx=float(s_avg[0, 0]), sigma_yy=float(s_avg[1, 1]),
                   sigma_zz=float(s_avg[2, 2]),
                   U_internal=U_internal, U_macro=U_macro,
                   eps_x=float(eps_x), eps_y=float(eps_y),
                   red_res=float(onp.abs(problem.P_mat.T @ r.reshape(-1)).max()),
                   balance=Fz_top + Fz_bottom,
                   reaction_consistency_err=abs(Fz_top - float(s_avg[2, 2])),
                   work_identity_err=abs(U_internal - U_macro),
                   t_build=round(t_build, 2), t_solve=round(t_solve, 2))

        finite = all(onp.isfinite(v) for v in row.values()
                     if isinstance(v, (int, float)))
        checks = {
            "finite": bool(finite),
            f"red_res<={tol_res:.0e}": row["red_res"] <= tol_res,
            f"balance<={tol_balance:.0e}": abs(row["balance"]) <= tol_balance,
            f"reaction<={tol_reaction:.0e}":
                row["reaction_consistency_err"] <= tol_reaction,
            f"work_identity<={tol_work:.0e}":
                row["work_identity_err"] <= tol_work,
        }
        row["checks"] = checks
        row["status"] = "ok" if all(checks.values()) else "check_failed"
        if include_problem:
            row["_problem"] = problem
        row["_projection_c"] = float(problem.projection_c)
        row["_projection_calibration"] = problem.projection_calibration
    except Exception as exc:
        row["status"] = f"FAILED: {type(exc).__name__}: {exc}"
    return row


def main():
    os.makedirs("results", exist_ok=True)
    rows = []
    failed = False

    print("== study 1: N x beta coupling (lateral macro strain fixed) ==",
          flush=True)
    for N in (16, 24, 32):
        for beta in (10.0, 20.0, 40.0):
            row = evaluate_case(N, beta, 1e-3, "fixed")
            row["study"] = "mesh_beta"
            rows.append(row)
            print(f"  N={N:2d} beta={beta:5.1f}: {row['status']} "
                  f"Vf={row.get('vf_int', float('nan')):.4f} "
                  f"Fz={row.get('Fz_top', float('nan')):+.6f} "
                  f"U={row.get('U_internal', float('nan')):.4e} "
                  f"work_err={row.get('work_identity_err', float('nan')):.1e} "
                  f"t={row.get('t_solve', float('nan'))}s", flush=True)
            if row["status"] != "ok":
                failed = True
                print("    -> stop study after failed case", flush=True)
                break
        if failed:
            break

    print("\n== study 2: E_min verification (N=32, beta=20, fixed) ==",
          flush=True)
    if not failed:
        row = evaluate_case(32, 20.0, 1e-4, "fixed")
        row["study"] = "emin"
        rows.append(row)
        failed = row["status"] != "ok"
        print(f"  E_min/E_s=1e-4: {row['status']} "
              f"Fz={row.get('Fz_top', float('nan')):+.6f} "
              f"U={row.get('U_internal', float('nan')):.4e}", flush=True)
    else:
        print("  skipped after failed coupling case", flush=True)

    csv_rows = []
    for row in rows:
        out = {k: v for k, v in row.items() if not k.startswith("_")}
        if "checks" in out:
            out["checks"] = ";".join(
                f"{k}:{'pass' if v else 'FAIL'}" for k, v in out["checks"].items())
        csv_rows.append(out)
    with open(CSV_PATH, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nwrote {CSV_PATH} ({len(csv_rows)} rows)")

    # --- analysis: separate adjacent rates from vs-finest differences ---
    fz = {(r["N"], r["beta"]): r["Fz_top"] for r in csv_rows
          if r["status"] == "ok" and r["study"] == "mesh_beta"}
    print("\n== beta=20 column: Fz refinement indicators (fixed lateral) ==")
    for a, b in ((16, 24), (24, 32)):
        if (a, 20.0) in fz and (b, 20.0) in fz:
            adj = abs(fz[(b, 20.0)] - fz[(a, 20.0)]) / abs(fz[(b, 20.0)])
            print(f"  adjacent {a}->{b}: |dFz|/|Fz_b| = {adj:.4f}")
    if (32, 20.0) in fz:
        for N in (16, 24, 32):
            if (N, 20.0) not in fz:
                continue
            ref = abs(fz[(N, 20.0)] - fz[(32, 20.0)]) / abs(fz[(32, 20.0)])
            print(f"  vs-finest N={N:2d}: |Fz(N)-Fz(32)|/|Fz(32)| = {ref:.4f}")
    print("(both are discretization indicators, NOT true physical errors)")

    print("\n== beta effect at fixed N (fixed lateral) ==")
    for N in (16, 24, 32):
        if all((N, b) in fz for b in (10.0, 20.0, 40.0)):
            span = abs(fz[(N, 40.0)] - fz[(N, 10.0)]) / abs(fz[(N, 20.0)])
            print(f"  N={N}: |Fz(beta=40)-Fz(beta=10)|/|Fz(beta=20)| = {span:.4f}")

    e_rows = [r for r in csv_rows if r["study"] == "emin" and r["status"] == "ok"]
    if e_rows:
        fz_e4 = e_rows[0]["Fz_top"]
        fz_e3 = fz.get((32, 20.0))
        if fz_e3:
            print(f"\n== E_min N=32 beta=20: 1e-4 vs 1e-3 relative diff of Fz = "
                  f"{abs(fz_e4 - fz_e3)/abs(fz_e3):.4f} ==")
    return not failed


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

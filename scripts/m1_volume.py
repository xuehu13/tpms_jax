"""M1-B run entry: resolution study, beta sensitivity, ParaView export.

Run from the project root:  pixi run python scripts/m1_volume.py
"""
import os
import sys
import time

# 共享显存的笔记本 GPU:按需分配而不是预分配 75% 显存
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from geometry import density, gyroid, solid_mask
from volume import (
    binary_volume_fraction,
    calibrate_c,
    cell_centers,
    smooth_volume_fraction,
    write_vti_cell_data,
)

TARGET = 0.35
RESOLUTIONS = [32, 64, 96, 128]
BETAS = [10.0, 20.0, 40.0, 80.0]
EXPORT_N = 64
EXPORT_BETA = 40.0


def main():
    print(f"Target binary volume fraction: {TARGET}")
    print("== Resolution study (calibrate c on binary geometry) ==")
    print(f"{'N':>5} {'calibrated c':>14} {'rho_binary':>12} {'abs err':>10}")
    c_ref = None
    xyz_ref = None
    rho_ref = None
    for N in RESOLUTIONS:
        xyz = cell_centers(N)
        t0 = time.perf_counter()
        c, rho, err = calibrate_c(TARGET, xyz)
        dt = time.perf_counter() - t0
        print(f"{N:>5} {c:>14.6f} {rho:>12.6f} {err:>10.2e}   ({dt:.1f} s)")
        if N == RESOLUTIONS[-1]:
            c_ref, xyz_ref, rho_ref = c, xyz, rho

    print(f"\n== Beta sensitivity at N={RESOLUTIONS[-1]}, c={c_ref:.6f} (fixed geometry) ==")
    print(f"{'beta':>6} {'rho_smooth':>12} {'rho_binary':>12} {'abs diff':>10} {'rel diff':>10}")
    for beta in BETAS:
        rs = float(smooth_volume_fraction(xyz_ref, c_ref, beta))
        abs_diff = abs(rs - rho_ref)
        rel_diff = abs_diff / rho_ref
        print(f"{beta:>6.0f} {rs:>12.6f} {rho_ref:>12.6f} {abs_diff:>10.2e} {rel_diff:>10.2e}")

    print(f"\n== ParaView export: N={EXPORT_N} cell-center grid, c={c_ref:.6f}, beta={EXPORT_BETA} ==")
    xyz = cell_centers(EXPORT_N)
    fields = {
        "G": (gyroid(xyz), "Float64"),
        "solid": (solid_mask(xyz, c_ref).astype(jnp.uint8), "UInt8"),
        "phi_beta40": (density(xyz, c_ref, EXPORT_BETA), "Float64"),
    }
    os.makedirs("results", exist_ok=True)
    out = os.path.join("results", f"gyroid_cell_center_N{EXPORT_N}.vti")
    write_vti_cell_data(out, EXPORT_N, 1.0, fields)
    print(f"wrote {out} ({EXPORT_N**3} cells, fields: {list(fields)})")


if __name__ == "__main__":
    main()

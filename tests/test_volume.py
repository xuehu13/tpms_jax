"""M1-B tests: regular sampling, volume fractions, c calibration, vti export."""
import xml.etree.ElementTree as ET

import jax
import jax.numpy as jnp
import numpy as onp
import pytest

from geometry import density, gyroid, solid_mask
from volume import (
    binary_volume_fraction,
    calibrate_c,
    cell_centers,
    smooth_volume_fraction,
    write_vti_cell_data,
)

jax.config.update("jax_enable_x64", True)


# ---------- 1./2. cell-center coordinates: count, range, no periodic endpoints ----------
def test_cell_centers_shape_and_range():
    N, L = 12, 1.3
    xyz = cell_centers(N, L)
    assert xyz.shape == (N, N, N, 3)
    flat = onp.asarray(xyz).reshape(-1, 3)
    assert flat.shape[0] == N**3
    step = L / N
    assert flat.min() == pytest.approx(step / 2, rel=1e-12)
    assert flat.max() == pytest.approx(L - step / 2, rel=1e-12)
    # each axis has exactly the N analytical centers
    xs = onp.unique(onp.round(flat[:, 0], 12))
    assert xs.shape[0] == N
    assert onp.allclose(xs, (onp.arange(N) + 0.5) * step)


def test_no_periodic_endpoint_sampling_or_duplicates():
    N, L = 10, 1.0
    flat = onp.asarray(cell_centers(N, L)).reshape(-1, 3)
    assert onp.all(flat > 0.0) and onp.all(flat < L)
    assert not onp.any(onp.isclose(flat, 0.0, atol=1e-12))
    assert not onp.any(onp.isclose(flat, L, atol=1e-12))
    assert onp.unique(flat, axis=0).shape[0] == N**3


# ---------- 3./4. volume fractions stay in [0, 1] ----------
def test_binary_volume_fraction_range():
    xyz = cell_centers(16)
    for c in (0.05, 0.1, 0.2, 0.4, 0.8):
        v = float(binary_volume_fraction(xyz, c))
        assert 0.0 <= v <= 1.0


def test_smooth_volume_fraction_range():
    xyz = cell_centers(16)
    for beta in (5.0, 50.0):
        v = float(smooth_volume_fraction(xyz, 0.15, beta))
        assert 0.0 <= v <= 1.0


# ---------- 5. binary volume fraction monotone non-decreasing in c ----------
def test_binary_volume_fraction_monotone_in_c():
    xyz = cell_centers(16)
    cs = [0.05, 0.1, 0.2, 0.4, 0.8]
    vals = [float(binary_volume_fraction(xyz, c)) for c in cs]
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:])), f"non-monotone: {vals}"


# ---------- 6. calibration for target 0.35 reaches resolution-matched error ----------
def test_calibrate_c_target_035():
    N = 32
    xyz = cell_centers(N)
    target = 0.35
    c, rho, err = calibrate_c(target, xyz)
    assert 0.05 <= c <= 0.8, "calibrated c unreasonably outside geometric range"
    assert err <= 1e-3, "did not reach the declared bisection tolerance"
    # 0.2 pct of target: near critical points of G the binary staircase has
    # steps much taller than 1/N^3, so sub-step errors are not achievable
    assert err <= 2e-3
    assert abs(rho - target) == pytest.approx(err, abs=1e-12)
    # consistency: recomputing rho_binary at the returned c reproduces rho
    assert abs(float(binary_volume_fraction(xyz, c)) - rho) < 1e-12


# ---------- 7. refinement does not produce anomalous jumps ----------
def test_resolution_refinement_stable():
    target = 0.35
    calib = {}
    for N in (32, 64):
        c, rho, err = calibrate_c(target, cell_centers(N))
        calib[N] = (c, rho, err)
        assert abs(rho - target) <= 1e-3
        assert 0.05 <= c <= 0.8
    c32 = calib[32][0]
    c64 = calib[64][0]
    # cell-center sampling has a systematic O(1/N) drift; the band only
    # excludes anomalous jumps (observed N=32 to 64 shift is about 0.023)
    assert abs(c64 - c32) < 0.05, f"calibrated c jumped on refinement: {c32} -> {c64}"


# ---------- 8. exported visualization data matches cell count ----------
def test_vti_export_matches_cell_count(tmp_path):
    N = 8
    xyz = cell_centers(N)
    fields = {
        "G": (gyroid(xyz), "Float64"),
        "solid": (solid_mask(xyz, 0.3).astype(jnp.uint8), "UInt8"),
        "phi_beta20": (density(xyz, 0.3, 20.0), "Float64"),
    }
    path = tmp_path / "probe.vti"
    write_vti_cell_data(str(path), N, 1.0, fields)

    root = ET.parse(str(path)).getroot()
    assert root.get("type") == "ImageData"
    img = root.find("ImageData")
    assert img.get("WholeExtent") == f"0 {N} 0 {N} 0 {N}"
    assert img.get("Spacing") == "0.125 0.125 0.125"
    arrays = {
        a.get("Name"): a.text.split()
        for a in img.find("Piece/CellData").findall("DataArray")
    }
    assert set(arrays) == set(fields)
    for name, (field, _) in fields.items():
        vals = onp.array(arrays[name], dtype=onp.float64)
        ref = onp.asarray(field).reshape(-1)
        assert vals.shape == ref.shape == (N**3,)
        assert onp.allclose(vals, ref, atol=1e-9)

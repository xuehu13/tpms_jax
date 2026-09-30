"""M1-A tests: Gyroid implicit function, binary solid phase, design density."""
import jax
import jax.numpy as jnp
import numpy as onp
import pytest

from geometry import density, gyroid, solid_mask

jax.config.update("jax_enable_x64", True)

KEY = jax.random.PRNGKey(0)


def random_points(n, L=1.0):
    return L * jax.random.uniform(KEY, (n, 3), minval=0.0, maxval=L)


# ---------- 1. known analytic values ----------
def test_gyroid_analytic_values():
    assert float(gyroid(jnp.array([0.0, 0.0, 0.0]))) == pytest.approx(0.0, abs=1e-14)
    # X=Y=Z=pi/2 -> all three terms vanish
    p = jnp.array([0.25, 0.25, 0.25])
    assert float(gyroid(p)) == pytest.approx(0.0, abs=1e-14)
    # (0, L/4, 0): sin(Y)cos(Z) = 1
    p = jnp.array([0.0, 0.25, 0.0])
    assert float(gyroid(p)) == pytest.approx(1.0, abs=1e-14)
    # period scaling: with L=2, x=0.25 -> X = pi/4
    p = jnp.array([0.25, 0.0, 0.0])
    assert float(gyroid(p, L=2.0)) == pytest.approx(onp.sqrt(2.0) / 2.0, abs=1e-14)


# ---------- 2. periodicity in x, y, z ----------
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_gyroid_periodicity(axis):
    L = 1.3  # non-default period
    pts = random_points(512, L=L)
    shift = jnp.zeros_like(pts).at[:, axis].set(L)
    g0 = gyroid(pts, L=L)
    g1 = gyroid(pts + shift, L=L)
    assert onp.allclose(onp.asarray(g0), onp.asarray(g1), atol=1e-9)


# ---------- 3. single point and batched inputs ----------
def test_single_point_and_batch_consistency():
    p = jnp.array([0.1, 0.2, 0.3])
    g_single = gyroid(p)
    assert jnp.asarray(g_single).shape == ()

    batch = random_points(64)
    g_batch = gyroid(batch)
    assert g_batch.shape == (64,)

    # the batched value at a repeated row must equal the single-point value
    row = jnp.repeat(p[None, :], 5, axis=0)
    assert onp.allclose(onp.asarray(gyroid(row)), onp.asarray(g_single).repeat(5), atol=1e-14)

    # arbitrary batch shape (..., 3)
    grid = jnp.zeros((4, 5, 3))
    assert gyroid(grid).shape == (4, 5)

    # same for mask and density
    assert jnp.asarray(solid_mask(p, c=0.3)).shape == ()
    assert solid_mask(batch, c=0.3).shape == (64,)
    d = density(batch, c=0.3, beta=20.0)
    assert d.shape == (64,)
    assert onp.asarray(d).dtype == onp.float64


# ---------- 4. binary solid phase consistent with |G| <= c ----------
def test_solid_mask_definition():
    pts = random_points(2048)
    c = 0.4
    mask = solid_mask(pts, c)
    assert mask.dtype == jnp.bool_
    assert onp.array_equal(onp.asarray(mask), onp.abs(onp.asarray(gyroid(pts))) <= c)

    # hand-checked points: |G| = 1 at (0, L/4, 0)
    boundary_touch = jnp.array([0.0, 0.25, 0.0])
    assert bool(solid_mask(boundary_touch, c=1.0))      # inclusive boundary
    assert not bool(solid_mask(boundary_touch, c=0.5))
    assert bool(solid_mask(jnp.array([0.0, 0.0, 0.0]), c=1e-3))  # G=0 is solid


# ---------- 5. density always in [0, 1] ----------
def test_density_range():
    pts = random_points(8192)
    for c, beta in [(0.1, 8.0), (0.2, 40.0), (0.05, 200.0)]:
        d = onp.asarray(density(pts, c, beta))
        assert d.min() >= -1e-12 and d.max() <= 1.0 + 1e-12


# ---------- 6. density non-decreasing in c ----------
def test_density_monotone_in_c():
    pts = random_points(1024)
    cs = [0.05, 0.1, 0.15, 0.2, 0.25]
    vals = onp.stack([onp.asarray(density(pts, c, beta=20.0)) for c in cs])
    diffs = onp.diff(vals, axis=0)
    assert diffs.min() >= -1e-12, f"density decreased in c: min diff {diffs.min():.3e}"


# ---------- 7. beta -> binary limit away from the boundary ----------
def test_density_converges_to_binary_with_beta():
    c = 0.15
    margin = 0.05
    pts = random_points(4096)
    g = onp.abs(onp.asarray(gyroid(pts)))

    inside = g <= (c - margin)    # deep inside the sheet
    outside = g >= (c + margin)   # deep inside the pore
    assert inside.sum() > 0 and outside.sum() > 0

    # interior density grows toward 1 with beta and is essentially binary at large beta
    d_small = onp.asarray(density(pts[inside], c, beta=5.0))
    d_large = onp.asarray(density(pts[inside], c, beta=200.0))
    assert d_large.min() >= 0.99
    assert d_large.mean() > d_small.mean()

    # exterior density decays toward 0
    d_out_small = onp.asarray(density(pts[outside], c, beta=5.0))
    d_out_large = onp.asarray(density(pts[outside], c, beta=200.0))
    assert d_out_large.max() <= 0.01
    assert d_out_large.mean() < d_out_small.mean()


# ---------- 8. automatic differentiation of density w.r.t. c ----------
def test_density_gradient_in_c_matches_finite_difference():
    pts = random_points(32)
    c0 = jnp.array(0.12)
    beta = 10.0

    def total(c):
        return jnp.sum(density(pts, c, beta))

    g_ad = float(jax.grad(total)(c0))
    h = 1e-5
    g_fd = float((total(c0 + h) - total(c0 - h)) / (2.0 * h))
    rel_err = abs(g_ad - g_fd) / abs(g_fd)
    print(f"\ndensity dc gradient: AD = {g_ad:.10e}, central FD = {g_fd:.10e}, rel err = {rel_err:.3e}")
    assert rel_err < 1e-5

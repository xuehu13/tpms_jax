"""Regular-grid sampling and volume fractions for the periodic Gyroid cell.

All sampling in this module is *geometric*: cell centers of an equal-volume
subdivision of the periodic unit cell [0, L)^3. These points are neither FEM
nodes nor HEX8 integration points; they define geometry-only volume
fractions and regular-grid data for visualization.
"""

import jax
import jax.numpy as jnp

from geometry import density, gyroid, solid_mask

__all__ = [
    "cell_centers",
    "binary_volume_fraction",
    "smooth_volume_fraction",
    "calibrate_c",
    "write_vti_cell_data",
]


def cell_centers(N, L=1.0):
    """Cell centers of [0, L)^3 subdivided into N^3 equal small cubes.

    x_i = (i + 0.5) * L / N for i = 0..N-1 (same in y and z). The periodic
    endpoints 0 and L are never sampled, so no pair of samples is periodic
    duplicates.

    Returns
    -------
    jax.Array, shape (N, N, N, 3)
    """
    i = jnp.arange(N, dtype=jnp.float64)
    c = (i + 0.5) * L / N
    X, Y, Z = jnp.meshgrid(c, c, c, indexing="ij")
    return jnp.stack((X, Y, Z), axis=-1)


def binary_volume_fraction(xyz, c, L=1.0):
    """Volume fraction of the true binary solid phase {|G| <= c}.

    Equal-volume cell-center samples make the plain mean an unbiased
    estimator. This is a geometric volume fraction, NOT a material mass
    density. At fixed N it is a monotone staircase in c.
    """
    return jnp.mean(solid_mask(xyz, c, L).astype(jnp.float64))


def smooth_volume_fraction(xyz, c, beta, L=1.0):
    """Spatial mean of the differentiable design density over the cell.

    Kept strictly separate from ``binary_volume_fraction``: the two agree
    only as beta -> inf.
    """
    return jnp.mean(density(xyz, c, beta, L))


def calibrate_c(target, xyz, L=1.0, c_lo=1e-4, c_hi=1.0, tol=1e-3, max_iter=60):
    """Bisection for the geometric threshold c such that rho_binary(c) ~= target.

    At fixed sampling resolution rho_binary is a monotone staircase in c with
    step height 1/N^3, so the search stops once the volume-fraction error is
    within ``tol`` or the bracket collapses; machine-precision roots are
    neither achievable nor attempted.

    Returns
    -------
    (c, rho_binary, abs_error)
        The best midpoint visited (smallest volume-fraction error).
    """

    def f(c):
        return float(binary_volume_fraction(xyz, c, L))

    flo, fhi = f(c_lo), f(c_hi)
    if not flo <= target <= fhi:
        raise ValueError(
            f"target volume fraction {target} outside bracket [{flo}, {fhi}]; "
            "widen c_lo/c_hi"
        )

    best_c, best_f, best_err = 0.5 * (c_lo + c_hi), None, float("inf")
    for _ in range(max_iter):
        cm = 0.5 * (c_lo + c_hi)
        fm = f(cm)
        err = abs(fm - target)
        if err < best_err:
            best_c, best_f, best_err = cm, fm, err
        if err <= tol or (c_hi - c_lo) < 1e-8:
            break
        if fm < target:
            c_lo = cm
        else:
            c_hi = cm
    return best_c, best_f, best_err


def write_vti_cell_data(filepath, N, L, fields):
    """Write cell-centered regular-grid fields as a VTK XML ImageData (.vti).

    meshio is deliberately not used: its data model cannot express
    cell-centered ImageData (its cell_data requires explicit cell blocks).
    The output conforms to the standard VTK XML schema and opens directly in
    ParaView. The fields are cell-center geometry samples -- solid indicator
    and design density on a regular grid -- NOT FEM integration-point data.
    """
    h = L / N
    arrays = "\n".join(
        _dataarray_xml(name, jnp.asarray(v).reshape(-1), dtype)
        for name, (v, dtype) in fields.items()
    )
    xml = (
        '<?xml version="1.0"?>\n'
        f"<!-- Regular cell-center geometry sampling of the periodic cell [0,{L})^3.\n"
        f"     Grid: {N}^3 equal-volume cubes, samples at (i+0.5)*L/{N}.\n"
        "     Geometric fields only: not FEM nodes or integration points. -->\n"
        '<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">\n'
        f'  <ImageData WholeExtent="0 {N} 0 {N} 0 {N}" Origin="0 0 0" Spacing="{h} {h} {h}">\n'
        f'    <Piece Extent="0 {N} 0 {N} 0 {N}">\n'
        "      <CellData>\n"
        f"{arrays}\n"
        "      </CellData>\n"
        "    </Piece>\n"
        "  </ImageData>\n"
        "</VTKFile>\n"
    )
    with open(filepath, "w") as fh:
        fh.write(xml)


def _dataarray_xml(name, values, dtype="Float64", precision=12):
    nums = " ".join(f"{float(v):.{precision}g}" for v in values)
    return (
        f'        <DataArray type="{dtype}" Name="{name}" '
        'NumberOfComponents="1" format="ascii">\n'
        f"{nums}\n"
        "        </DataArray>"
    )

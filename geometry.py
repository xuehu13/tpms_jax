"""TPMS (Gyroid) implicit geometry: implicit function, binary solid phase,
and differentiable design density.

Coordinate convention: physical coordinates ``xyz`` with shape ``(..., 3)``
(a single point ``(3,)`` is also accepted). The unit-cell period ``L`` scales
the normalized coordinates X = 2*pi*x/L, Y = 2*pi*y/L, Z = 2*pi*z/L.

Three distinct concepts live in this module and must not be conflated:

1. ``gyroid``     -- the implicit function G(x, y, z). G = 0 is the TPMS
   surface; G is smooth and periodic with period L in every direction.
2. ``solid_mask`` -- the *true binary geometry*: the sheet-based Gyroid solid
   phase {|G| <= c}. Boolean output, no gradients, used for meshing/volume
   definitions and as the beta -> infinity reference.
3. ``density``    -- the *differentiable design density* phi in [0, 1] used
   later by FEM/optimization. It is a smooth projection of the binary
   geometry, NOT a physical mass density.
"""

import jax
import jax.numpy as jnp

__all__ = ["primitive", "gyroid", "solid_mask", "density", "project_field"]


def gyroid(xyz, L=1.0):
    """Gyroid implicit function.

    G(x, y, z) = sin(X)cos(Y) + sin(Y)cos(Z) + sin(Z)cos(X),
    with X = 2*pi*x/L, Y = 2*pi*y/L, Z = 2*pi*z/L.

    Parameters
    ----------
    xyz : array_like, shape (..., 3) or (3,)
        Physical coordinates.
    L : float
        Unit-cell period (all three directions), default 1.0.

    Returns
    -------
    jax.Array
        Implicit values, shape ``(...)``, differentiable in ``xyz`` and ``L``.
    """
    xyz = jnp.asarray(xyz)
    scale = 2.0 * jnp.pi / L
    X = scale * xyz[..., 0]
    Y = scale * xyz[..., 1]
    Z = scale * xyz[..., 2]
    return jnp.sin(X) * jnp.cos(Y) + jnp.sin(Y) * jnp.cos(Z) + jnp.sin(Z) * jnp.cos(X)


def solid_mask(xyz, c, L=1.0):
    """True binary solid phase of the sheet-based Gyroid: ``|G| <= c``.

    Parameters
    ----------
    xyz : array_like, shape (..., 3) or (3,)
        Physical coordinates.
    c : float
        Positive geometric threshold controlling the sheet thickness.
        This is a level-set width, NOT a physical wall thickness.
    L : float
        Unit-cell period.

    Returns
    -------
    jax.Array (bool)
        True where the point lies in the solid sheet, shape ``(...)``.
        No differentiable information is carried by this output.
    """
    return jnp.abs(gyroid(xyz, L)) <= c


def density(xyz, c, beta, L=1.0):
    """Smooth differentiable design density projecting the binary geometry.

    phi = sigmoid(beta * (G + c)) - sigmoid(beta * (G - c)).

    As beta -> inf, phi converges to the binary indicator of {|G| < c}
    (value 1/2 exactly on the boundary |G| = c). ``c`` and the coordinates
    remain differentiable through this function.

    Parameters
    ----------
    xyz : array_like, shape (..., 3) or (3,)
        Physical coordinates.
    c : float or jax.Array
        Positive geometric threshold (same level-set width as ``solid_mask``).
    beta : float
        Positive projection sharpness.
    L : float
        Unit-cell period.

    Returns
    -------
    jax.Array
        Design density in [0, 1], shape ``(...)``, differentiable in
        ``xyz``, ``c`` and ``L``. This is a design field for FEM/optimization,
        NOT a physical mass density.
    """
    return project_field(gyroid(xyz, L), c, beta)


def project_field(g, c, beta):
    """Project a signed implicit field with the stated implicit-field amplitude convention."""
    return jax.nn.sigmoid(beta * (g + c)) - jax.nn.sigmoid(beta * (g - c))


def primitive(xyz, L=1.0):
    """Schwarz Primitive nodal approximation: cos(X)+cos(Y)+cos(Z)."""
    return jnp.sum(jnp.cos(2*jnp.pi*jnp.asarray(xyz)/L), axis=-1)

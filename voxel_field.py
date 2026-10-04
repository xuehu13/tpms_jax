"""Periodic signed-field inputs and historical occupancy arrays on the fixed FEM."""
import numpy as np
import jax
import jax.numpy as jnp
from geometry import density, gyroid, project_field
from design_fem import GyroidDesign


@jax.jit
def periodic_trilinear(values, points):
    """Unit-cell xyz array -> point values; differentiate with respect to values."""
    if values.ndim != 3 or len(set(values.shape)) != 1 or points.shape[-1] != 3:
        raise ValueError("Expected cubic xyz array and (...,3) unit-cell points")
    M = values.shape[0]
    scaled = points * M - .5
    base = jnp.floor(scaled).astype(jnp.int32)
    fraction = scaled - base
    output = jnp.zeros(points.shape[:-1], dtype=values.dtype)
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                weight = ((fraction[..., 0] if i else 1-fraction[..., 0]) *
                          (fraction[..., 1] if j else 1-fraction[..., 1]) *
                          (fraction[..., 2] if k else 1-fraction[..., 2]))
                output = output + weight * values[(base[..., 0]+i) % M,
                                                 (base[..., 1]+j) % M,
                                                 (base[..., 2]+k) % M]
    return output


def sample_gyroid(M, c=.541062, beta=40.):
    coordinates = (jnp.arange(M, dtype=jnp.float64)+.5)/M
    points = jnp.stack(jnp.meshgrid(coordinates, coordinates, coordinates, indexing='ij'), axis=-1)
    return density(points, c, beta)


class VoxelDesign(GyroidDesign):
    """Reuse the same fixed-lateral solver, observables and adjoint for array input."""
    def __init__(self, N, M, beta=40., emin_ratio=1e-4, eps_z=-.01):
        self.M = M
        super().__init__(N, beta, emin_ratio, eps_z)

    def validate_theta(self, values):
        values = np.asarray(values, dtype=float)
        if values.shape != (self.M,)*3 or not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
            raise ValueError("Expected finite xyz occupancies in [0,1]")
        return jnp.asarray(values)

    def density_at(self, values, points):
        return periodic_trilinear(values, points)


def sample_implicit(M):
    """Signed Gyroid G at unit-period xyz cell centres; this is not an SDF."""
    x = (jnp.arange(M, dtype=jnp.float64)+.5)/M
    return gyroid(jnp.stack(jnp.meshgrid(x, x, x, indexing='ij'), axis=-1))


class ImplicitVoxelDesign(GyroidDesign):
    """Packed (flattened signed g, positive c): interpolate first, project once."""
    def __init__(self, N, M, beta=40., emin_ratio=1e-4, eps_z=-.01, **solver_kwargs):
        self.M = M
        super().__init__(N, beta, emin_ratio, eps_z, **solver_kwargs)

    def validate_theta(self, theta):
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.M**3+1,) or not np.isfinite(theta).all() or theta[-1] <= 0:
            raise ValueError('Expected finite signed xyz field followed by positive c')
        return jnp.asarray(theta)

    def density_at(self, theta, points):
        g = theta[:-1].reshape((self.M,)*3)
        return project_field(periodic_trilinear(g, points), theta[-1], self.beta)

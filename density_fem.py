"""M3-A: integration-point design density in the periodic HEX8 problem.

The design density lives on the TRUE finite-element Gauss points:
rho[e, q] = density(X[e, q], c, beta, L) with X[e, q] taken from
``Problem.physical_quad_points`` (shape (num_cells, num_quads, 3)) and
JxW from ``Problem.JxW``. The pointwise modulus is

    E_q = E_min + rho_q * (E_s - E_min)        (p = 1)

with fixed nu; lambda_q and mu_q follow from E_q per quadrature point.
H_macro, lambda_q and mu_q reach the constitutive law through jax-fem's
runtime ``internal_vars``; the density stays a pure JAX array, so the
c -> rho path remains differentiable.
"""

import numpy as onp
import scipy.sparse
import jax
import jax.numpy as jnp

from fem import NU, ELE_TYPE
from geometry import density
from pbc import PeriodicLinearElasticityCube, periodic_p_mat

__all__ = ["DensityLinearElasticityPeriodic", "make_density_problem",
           "layered_rho"]


class DensityLinearElasticityPeriodic(PeriodicLinearElasticityCube):
    """u = H_macro @ X + w with integration-point modulus interpolation."""

    def custom_init(self):
        super().custom_init()
        num_cells, num_quads = self.fe.num_cells, self.fe.num_quads
        self.rho = jnp.ones((num_cells, num_quads))

    def set_params(self, H_macro, rho, E_s=10.0, E_min=None):
        """``rho``: (num_cells, num_quads) design density in [0, 1]."""
        self.H_macro = jnp.asarray(H_macro)
        self.rho = jnp.asarray(rho)
        if E_min is None:
            E_min = 1e-3 * E_s
        E_q = E_min + self.rho * (jnp.asarray(E_s) - E_min)
        lam_q = E_q * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
        mu_q = E_q / (2.0 * (1.0 + NU))
        num_cells, num_quads = self.fe.num_cells, self.fe.num_quads
        self.internal_vars = [
            jnp.broadcast_to(self.H_macro, (num_cells, num_quads, 3, 3)),
            jnp.broadcast_to(lam_q, (num_cells, num_quads)),
            jnp.broadcast_to(mu_q, (num_cells, num_quads)),
        ]

    def get_tensor_map(self):
        def first_PK(u_grad, H_macro, lam_q, mu_q):
            grad_u_total = H_macro + u_grad
            eps = 0.5 * (grad_u_total + grad_u_total.T)
            sigma = lam_q * jnp.trace(eps) * jnp.eye(self.dim) + 2.0 * mu_q * eps
            return sigma

        return first_PK

    def get_mass_map(self):
        def mass(u, x, H_macro, lam_q, mu_q):
            return jnp.zeros(self.vec)

        return mass


def make_density_problem(Nx, Ny, Nz, H_macro, rho_quad, E_s=10.0, E_min=None,
                         cell_size=1.0, periodic_axes=(0, 1), fixed_class=None,
                         fixed_dofs=()):
    """Density-periodic problem on an Nx x Ny x Nz HEX8 grid.

    ``rho_quad`` must have shape (num_cells, num_quads); build it from
    ``density(problem.physical_quad_points, c, beta, L)`` (see tests and
    scripts/m3_gyroid_first.py).
    """
    from jax_fem.generate_mesh import box_mesh, get_meshio_cell_type, Mesh

    cell_type = get_meshio_cell_type(ELE_TYPE)
    meshio_mesh = box_mesh(Nx, Ny, Nz, cell_size, cell_size, cell_size)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])
    problem = DensityLinearElasticityPeriodic(
        mesh, vec=3, dim=3, ele_type=ELE_TYPE,
        dirichlet_bc_info=[[], [], []])
    if callable(fixed_dofs):
        fixed_dofs = fixed_dofs(problem.fe.points)
    H = jnp.zeros((3, 3)) if H_macro is None else jnp.asarray(H_macro)
    rho = jnp.ones((problem.fe.num_cells, problem.fe.num_quads)) \
        if rho_quad is None else jnp.asarray(rho_quad)
    problem.set_params(H, rho, E_s, E_min)
    P, class_ids, fixed_id = periodic_p_mat(
        problem.fe.points, Nx, Ny, Nz,
        Lx=cell_size, Ly=cell_size, Lz=cell_size,
        periodic_axes=periodic_axes, fixed_class=fixed_class,
        fixed_dofs=fixed_dofs)
    problem.P_mat = P
    problem.class_ids = class_ids
    problem.fixed_class_id = fixed_id
    return problem


def layered_rho(problem, rho_bottom, rho_top, interface_z=0.5):
    """Two-layer density along Z with the interface on an element boundary.

    Layer membership is decided by the cell centroid; all quadrature points
    of a cell share the layer value.
    """
    pts = onp.asarray(problem.fe.points)
    cells = onp.asarray(problem.fe.cells)
    cell_z = pts[cells].mean(axis=1)[:, 2]
    rho = onp.where(cell_z[:, None] >= interface_z,
                    float(rho_top), float(rho_bottom))
    return jnp.asarray(rho)

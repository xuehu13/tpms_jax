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

from fem import NU, ELE_TYPE, solve
from geometry import density
from pbc import PeriodicLinearElasticityCube, periodic_p_mat

__all__ = ["DensityLinearElasticityPeriodic", "make_density_problem",
           "layered_rho", "avg_stress", "solve_lateral_relaxation"]


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
    if callable(rho_quad):
        rho_quad = rho_quad(problem)
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
    of a cell share the layer value. The output is the standard integration-
    point density shape (num_cells, num_quads).
    """
    pts = onp.asarray(problem.fe.points)
    cells = onp.asarray(problem.fe.cells)
    cell_z = pts[cells].mean(axis=1)[:, 2]
    num_cells, num_quads = problem.fe.num_cells, problem.fe.num_quads
    per_cell = onp.where(cell_z >= interface_z, float(rho_top), float(rho_bottom))
    rho = onp.repeat(per_cell[:, None], num_quads, axis=1)
    assert rho.shape == (num_cells, num_quads)
    return jnp.asarray(rho)


def avg_stress(problem, sol_list):
    """JxW-weighted volume-average stress over the periodic cell: (3, 3)."""
    H = onp.asarray(problem.H_macro)
    lam = onp.asarray(problem.internal_vars[1])[..., None, None]
    mu = onp.asarray(problem.internal_vars[2])[..., None, None]
    u_grad = onp.asarray(problem.fe.sol_to_grad(sol_list[0]))
    eps = 0.5 * (u_grad + onp.swapaxes(u_grad, -1, -2)) + H
    sigma = lam * onp.trace(eps, axis1=-2, axis2=-1)[..., None, None] * onp.eye(3)         + 2.0 * mu * eps
    JxW = onp.asarray(problem.JxW)[:, 0, :]
    return onp.sum(sigma * JxW[..., None, None], axis=(0, 1)) / JxW.sum()


def solve_lateral_relaxation(Nx, Ny, Nz, rho_quad, eps_z=-0.01, h=0.01,
                             E_s=10.0, E_min=None, cell_size=1.0,
                             periodic_axes=(0, 1), fixed_class=None,
                             fixed_dofs=()):
    """Solve the macroscopic lateral strains with zero average lateral stress.

    The material is linear elastic with a FIXED density, so the average
    lateral stress is affine in (eps_x, eps_y). With the axial strain fixed
    at ``eps_z``:

        H0 = diag(0, 0, eps_z)
        Hx = diag(h, 0, eps_z)
        Hy = diag(0, h, eps_z)

    two probe solves give the 2x2 lateral response matrix
    A[:, i] = (sigma_lat(H_i) - sigma_lat(H0)) / h and the load vector
    b = sigma_lat(H0); eps_lateral = solve(A, -b) cancels the average
    lateral stress exactly (to linear-solver tolerance). The same Problem
    instance (mesh, P_mat, density) is reused; only internal_vars change.

    Returns a dict with the response matrix, eps_x/eps_y, the final H and
    solution, and the average stress of every solve.
    """
    problem = make_density_problem(
        Nx, Ny, Nz, jnp.zeros((3, 3)),
        None if callable(rho_quad) else rho_quad,
        E_s, E_min, cell_size, periodic_axes, fixed_class, fixed_dofs)
    if callable(rho_quad):
        # 依赖问题几何的密度场(例如 M1 density 作用在真实 Gauss 点上)
        rho_quad = rho_quad(problem)

    def run(H_vec):
        H = onp.zeros((3, 3))
        H[0, 0], H[1, 1], H[2, 2] = H_vec
        problem.set_params(H, rho_quad, E_s, E_min)
        sol = solve(problem)
        return sol, avg_stress(problem, sol)

    s0, sol0 = None, None
    sol0, s0 = run((0.0, 0.0, eps_z))
    b = onp.array([s0[0, 0], s0[1, 1]])
    A = onp.zeros((2, 2))
    probe_solutions = {}
    for i, ax in enumerate((0, 1)):
        hv = [0.0, 0.0, eps_z]
        hv[ax] = h
        sol_i, s_i = run(hv)
        A[:, i] = (onp.array([s_i[0, 0], s_i[1, 1]]) - b) / h
        probe_solutions[ax] = (sol_i, s_i)
    eps_lat = onp.linalg.solve(A, -b)
    H_final = onp.zeros((3, 3))
    H_final[0, 0], H_final[1, 1], H_final[2, 2] = eps_lat[0], eps_lat[1], eps_z
    sol_final, s_final = run((eps_lat[0], eps_lat[1], eps_z))
    return {"problem": problem, "A": A, "b": b, "eps_x": float(eps_lat[0]),
            "eps_y": float(eps_lat[1]), "H_final": H_final,
            "sol_list": sol_final, "sigma_avg": s_final,
            "sigma_avg_base": s0, "base_sol": sol0,
            "probe_solutions": probe_solutions}

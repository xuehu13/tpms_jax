"""M2-A: full-solid HEX8 linear elasticity benchmark on the unit cube.

Every element is complete homogeneous material (no TPMS density, no stiffness
interpolation). Built on the installed JAX-FEM 0.0.12 public API:
generate_mesh.box_mesh, Problem.get_tensor_map, solver.compute_residual.
"""

import numpy as onp
import jax
import jax.numpy as jnp

from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.generate_mesh import box_mesh, get_meshio_cell_type, Mesh

E = 10.0
NU = 0.3
LAMBDA = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
MU = E / (2.0 * (1.0 + NU))

ELE_TYPE = "HEX8"


class LinearElasticityCube(Problem):
    """Isotropic small-strain linear elasticity: sigma = lam tr(eps) I + 2 mu eps."""

    def get_tensor_map(self):
        def stress(u_grad):
            eps = 0.5 * (u_grad + u_grad.T)
            sigma = LAMBDA * jnp.trace(eps) * jnp.eye(self.dim) + 2.0 * MU * eps
            return sigma

        return stress

    def custom_init(self):
        self.fe = self.fes[0]


def make_cube_problem(nx, ny, nz, dirichlet_bc_info, location_fns=None):
    """Unit-cube HEX8 mesh (jax_fem box_mesh) + the elasticity problem."""
    cell_type = get_meshio_cell_type(ELE_TYPE)
    meshio_mesh = box_mesh(nx, ny, nz, 1.0, 1.0, 1.0)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])
    return LinearElasticityCube(
        mesh, vec=3, dim=3, ele_type=ELE_TYPE,
        dirichlet_bc_info=dirichlet_bc_info, location_fns=location_fns)


def solve(problem):
    return solver(problem)


def internal_force(problem, sol_list):
    """Uncorrected weak-form residual r_unc = f_int - f_ext (no body force).

    jax-fem 0.0.12 imposes Dirichlet BCs by row elimination on a COPY of the
    system; ``Problem.compute_residual`` is documented (solver.py) as the
    residual function *without* Dirichlet modification. Hence at a
    constrained DOF ``r_unc`` equals the internal force there, which (with
    no Neumann loading) is the reaction transmitted through that support.
    Returns onp array of shape (num_nodes, 3).
    """
    return onp.asarray(problem.compute_residual(sol_list)[0])


def affine_bc(ex, ey, ez):
    """Dirichlet info prescribing u=(ex*x, ey*y, ez*z) on all six faces.

    jax-fem Dirichlet value functions return scalars for one component of
    one point set, so the affine field is split into three component fns.
    """

    def u_x(point):
        return ex * point[0]

    def u_y(point):
        return ey * point[1]

    def u_z(point):
        return ez * point[2]

    def left(p):
        return jnp.isclose(p[0], 0.0, atol=1e-8)

    def right(p):
        return jnp.isclose(p[0], 1.0, atol=1e-8)

    def bottom(p):
        return jnp.isclose(p[1], 0.0, atol=1e-8)

    def top(p):
        return jnp.isclose(p[1], 1.0, atol=1e-8)

    def back(p):
        return jnp.isclose(p[2], 0.0, atol=1e-8)

    def front(p):
        return jnp.isclose(p[2], 1.0, atol=1e-8)

    faces = [left, right, bottom, top, back, front]
    location_fns = [fn for fn in faces for _ in range(3)]
    vecs = [0, 1, 2] * 6
    value_fns = [u_x, u_y, u_z] * 6
    return [location_fns, vecs, value_fns]


def uniaxial_bc(delta):
    """Uniaxial compression: bottom uz=0, top uz=delta, minimal rigid-mode
    removal (one bottom corner ux+uy, opposite-ish bottom corner uy).

    The constraints sit where the analytic uniaxial field
    u=(-NU*delta*x, -NU*delta*y, delta*z) vanishes anyway, so the lateral
    Poisson contraction stays unconstrained.
    """

    def bottom(p):
        return jnp.isclose(p[2], 0.0, atol=1e-8)

    def top(p):
        return jnp.isclose(p[2], 1.0, atol=1e-8)

    def corner_a(p):
        return (jnp.isclose(p[0], 0.0, atol=1e-8) &
                jnp.isclose(p[1], 0.0, atol=1e-8) & bottom(p))

    def corner_b(p):
        return (jnp.isclose(p[0], 1.0, atol=1e-8) &
                jnp.isclose(p[1], 0.0, atol=1e-8) & bottom(p))

    def zero(p):
        return 0.0

    def top_uz(p):
        return delta

    location_fns = [bottom, top, corner_a, corner_a, corner_b]
    vecs = [2, 2, 0, 1, 1]
    value_fns = [zero, top_uz, zero, zero, zero]
    return [location_fns, vecs, value_fns]


def face_nodes(problem, axis, value, tol=1e-8):
    """Node indices on the plane x_axis == value of the unit cube."""
    pts = onp.asarray(problem.fe.points)
    return onp.where(onp.isclose(pts[:, axis], value, atol=tol))[0]


def strain_energy(problem, sol_list):
    """0.5 * int sigma:eps dV from the recovered displacement gradient."""
    u_grad = problem.fe.sol_to_grad(sol_list[0])  # (num_cells, num_quads, 3, 3)
    eps = 0.5 * (u_grad + jnp.swapaxes(u_grad, -1, -2))
    sigma = LAMBDA * jnp.trace(eps, axis1=-2, axis2=-1)[..., None, None] * jnp.eye(3) \
        + 2.0 * MU * eps
    JxW = problem.JxW[:, 0, :]  # (num_cells, num_quads)
    return 0.5 * float(jnp.sum(sigma * eps * JxW[..., None, None]))


def hex_jacobian_check(problem):
    """det(J) of the trilinear isoparametric map at the 2x2x2 Gauss points.

    Independent of JAX-FEM: uses only the nodal coordinates of every cell.
    Returns (min_detJ, total_volume).
    """
    g = 1.0 / onp.sqrt(3.0)
    gauss = onp.array([-g, g])
    pts = onp.asarray(problem.fe.points)
    cells = onp.asarray(problem.fe.cells)
    dN = onp.zeros((8, 3))
    min_det = onp.inf
    total_vol = 0.0
    for cell in cells:
        xa = pts[cell]  # (8, 3)
        for gz in gauss:
            for gy in gauss:
                for gx in gauss:
                    for a in range(8):
                        sx = 1.0 if a in (1, 2, 5, 6) else -1.0
                        sy = 1.0 if a in (2, 3, 6, 7) else -1.0
                        sz = 1.0 if a in (4, 5, 6, 7) else -1.0
                        dN[a, 0] = 0.125 * sx * (1 + sy * gy) * (1 + sz * gz)
                        dN[a, 1] = 0.125 * (1 + sx * gx) * sy * (1 + sz * gz)
                        dN[a, 2] = 0.125 * (1 + sx * gx) * (1 + sy * gy) * sz
                    J = dN.T @ xa
                    det = onp.linalg.det(J)
                    min_det = min(min_det, det)
                    total_vol += det
    return float(min_det), float(total_vol)

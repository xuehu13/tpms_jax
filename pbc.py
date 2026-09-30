"""Full XYZ periodic boundary conditions on regular HEX8 grids.

The unknown field is the periodic fluctuation w in
u(X) = H_macro @ X + w(X); the macroscopic gradient H_macro enters the
constitutive law through jax-fem's runtime ``internal_vars`` mechanism.

Constraints u_a(node) = u_a(partner(node)) are imposed with jax-fem 0.0.12's
``P_mat`` mechanism: the user attaches a scipy CSR matrix ``problem.P_mat``
of shape (num_full_dofs, num_reduced_dofs) with u_full = P @ u_reduced; the
solver forms P.T @ R and P.T @ K @ P (verified in the installed source).
"""

import numpy as onp
import scipy.sparse
import jax
import jax.numpy as jnp

from fem import ELE_TYPE, E, LAMBDA, MU, LinearElasticityCube

__all__ = ["PeriodicLinearElasticityCube", "make_periodic_problem",
           "periodic_node_classes", "periodic_p_mat"]


def periodic_node_classes(points, Nx, Ny, Nz, Lx=1.0, Ly=1.0, Lz=1.0):
    """Periodic equivalence class id of every node.

    Integer node indices are recovered exactly from the regular-grid
    coordinates (no nearest-neighbour search): ix = round(x/Lx*Nx) in
    0..Nx. The class key is (ix mod Nx, iy mod Ny, iz mod Nz), flattened
    row-major; nodes sharing a key (e.g. all 8 corners) share one class.
    """
    pts = onp.asarray(points)
    ix = onp.round(pts[:, 0] / Lx * Nx).astype(int)
    iy = onp.round(pts[:, 1] / Ly * Ny).astype(int)
    iz = onp.round(pts[:, 2] / Lz * Nz).astype(int)
    assert ix.min() >= 0 and ix.max() <= Nx, "point outside the periodic grid"
    assert iy.min() >= 0 and iy.max() <= Ny, "point outside the periodic grid"
    assert iz.min() >= 0 and iz.max() <= Nz, "point outside the periodic grid"
    cx, cy, cz = ix % Nx, iy % Ny, iz % Nz
    class_ids = (cx * Ny + cy) * Nz + cz
    return class_ids, (ix, iy, iz)


def periodic_p_mat(points, Nx, Ny, Nz, Lx=1.0, Ly=1.0, Lz=1.0,
                   fixed_class=(0, 0, 0)):
    """CSR constraint matrix u_full = P @ u_reduced for full XYZ periodicity.

    One reduced DOF per (class, component). The equivalence class of
    ``fixed_class`` is EXCLUDED from the reduced space (zero rows in P),
    which removes the three rigid translations of the fluctuation field.

    Returns (P, class_ids, fixed_class_id).
    """
    class_ids, _ = periodic_node_classes(points, Nx, Ny, Nz, Lx, Ly, Lz)
    num_classes = Nx * Ny * Nz
    fcx, fcy, fcz = fixed_class[0] % Nx, fixed_class[1] % Ny, fixed_class[2] % Nz
    fixed_id = (fcx * Ny + fcy) * Nz + fcz

    class_to_red = onp.full(num_classes, -1, dtype=onp.int64)
    nxt = 0
    for cid in range(num_classes):
        if cid == fixed_id:
            continue
        class_to_red[cid] = nxt
        nxt += 1
    N_red = 3 * nxt  # three components per class

    num_nodes = len(points)
    N_full = 3 * num_nodes
    rows = onp.arange(N_full)
    cols = onp.empty(N_full, dtype=onp.int64)
    node_class_per_comp = onp.repeat(class_ids, 3)  # dof (3n+c) -> class
    comp = onp.tile(onp.arange(3), num_nodes)
    red_dof = 3 * class_to_red[node_class_per_comp] + comp
    is_fixed = node_class_per_comp == fixed_id
    cols[~is_fixed] = red_dof[~is_fixed]
    cols[is_fixed] = 0  # placeholder; these rows get no entry
    data = onp.ones(N_full - int(is_fixed.sum()))
    keep = ~is_fixed
    P = scipy.sparse.csr_array((data, (rows[keep], cols[keep])), shape=(N_full, N_red))
    return P, class_ids, fixed_id


def make_periodic_problem(Nx, Ny, Nz, H_macro=None,
                          sine_force_amplitude=None, cell_size=1.0):
    """Periodic unit cell: HEX8 grid of Nx x Ny x Nz cells, full XYZ PBC.

    ``H_macro`` sets the macroscopic gradient; ``sine_force_amplitude`` adds
    the verification body force f_x = (lambda+2mu) A k^2 sin(k x), k = 2*pi/L.
    """
    from jax_fem.generate_mesh import box_mesh, get_meshio_cell_type, Mesh
    from fem import ELE_TYPE

    cell_type = get_meshio_cell_type(ELE_TYPE)
    meshio_mesh = box_mesh(Nx, Ny, Nz, cell_size, cell_size, cell_size)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])
    problem = PeriodicLinearElasticityCube(mesh, vec=3, dim=3, ele_type=ELE_TYPE,
                                           dirichlet_bc_info=[[], [], []])
    H = jnp.zeros((3, 3)) if H_macro is None else jnp.asarray(H_macro)
    problem.set_params(H, sine_force_amplitude, cell_size)
    P, class_ids, fixed_id = periodic_p_mat(
        problem.fe.points, Nx, Ny, Nz,
        Lx=cell_size, Ly=cell_size, Lz=cell_size)
    problem.P_mat = P
    problem.class_ids = class_ids
    problem.fixed_class_id = fixed_id
    return problem


class PeriodicLinearElasticityCube(LinearElasticityCube):
    """u(X) = H_macro @ X + w(X); the solved field is the periodic w.

    H_macro reaches the constitutive law through runtime ``internal_vars``
    (one 3x3 gradient per quadrature point), so the solved problem is the
    weak form of div P(H + grad w) = b on the periodic cell.
    """

    def custom_init(self):
        super().custom_init()
        self.H_macro = jnp.zeros((3, 3))
        self.sine_force_amplitude = None
        self.cell_size = 1.0

    def set_params(self, H_macro, sine_force_amplitude=None, cell_size=1.0):
        self.H_macro = jnp.asarray(H_macro)
        self.sine_force_amplitude = sine_force_amplitude
        self.cell_size = cell_size
        self.internal_vars = [jnp.broadcast_to(
            self.H_macro, (self.fe.num_cells, self.fe.num_quads, 3, 3))]

    def get_tensor_map(self):
        def first_PK(u_grad, H_macro):
            grad_u_total = H_macro + u_grad
            eps = 0.5 * (grad_u_total + grad_u_total.T)
            sigma = LAMBDA * jnp.trace(eps) * jnp.eye(self.dim) + 2.0 * MU * eps
            return sigma

        return first_PK

    def get_mass_map(self):
        """Weak-form mass term m with div(sigma) = m, i.e. m = -body_force.

        The verification load b_x = (lambda+2mu) A k^2 sin(k x), b_y = b_z = 0
        gives the analytic fluctuation w_x = A sin(k x).
        """

        def mass(u, x, H_macro):  # H_macro: per-quad internal var, unused here
            if self.sine_force_amplitude is None:
                return jnp.zeros(self.vec)
            A = self.sine_force_amplitude
            k = 2.0 * jnp.pi / self.cell_size
            b_x = (LAMBDA + 2.0 * MU) * A * k**2 * jnp.sin(k * x[0])
            return -jnp.array([b_x, 0.0, 0.0])

        return mass

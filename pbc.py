"""Periodic boundary conditions on regular HEX8 grids: full XYZ or XY.

The unknown field is the periodic fluctuation w in
u(X) = H_macro @ X + w(X). Supported selections: ``periodic_axes=(0, 1, 2)``
(full XYZ) and ``periodic_axes=(0, 1)`` (XY; the two z-surfaces stay
independent, e.g. for flat uniaxial loading). The constraints couple only
the fluctuation w -- the macroscopic gradient H_macro enters the
constitutive law through jax-fem's runtime ``internal_vars`` mechanism and
is never constrained by P_mat.

Constraints u_a(node) = u_a(partner(node)) are imposed with jax-fem 0.0.12's
``P_mat`` mechanism: the user attaches a scipy CSR matrix ``problem.P_mat``
of shape (num_full_dofs, num_reduced_dofs) with u_full = P @ u_reduced; the
solver forms P.T @ R and P.T @ K @ P (verified in the installed source).
Zero rows in P pin the corresponding w DOFs to zero (rigid modes, flat
loaded faces).
"""

import numpy as onp
import scipy.sparse
import jax
import jax.numpy as jnp

from fem import ELE_TYPE, E, LAMBDA, MU, LinearElasticityCube

__all__ = ["PeriodicLinearElasticityCube", "make_periodic_problem",
           "periodic_node_classes", "periodic_p_mat"]


def periodic_node_classes(points, Nx, Ny, Nz, Lx=1.0, Ly=1.0, Lz=1.0,
                          periodic_axes=(0, 1, 2)):
    """Periodic equivalence class id of every node.

    Axes in ``periodic_axes`` are identified modulo the cell count (default
    XYZ); all other axes keep their integer levels as distinct classes, so
    e.g. ``periodic_axes=(0, 1)`` makes the cell XY-periodic while the two
    z-surfaces stay independent. Integer node indices are recovered exactly
    from the regular-grid coordinates (no nearest-neighbour search).
    """
    pts = onp.asarray(points)
    idx = (onp.round(pts[:, 0] / Lx * Nx).astype(int),
           onp.round(pts[:, 1] / Ly * Ny).astype(int),
           onp.round(pts[:, 2] / Lz * Nz).astype(int))
    Ns = (Nx, Ny, Nz)
    per = [a in periodic_axes for a in range(3)]
    dims = onp.array([Ns[a] if per[a] else Ns[a] + 1 for a in range(3)])
    c = [idx[a] % Ns[a] if per[a] else idx[a] for a in range(3)]
    class_ids = (c[0] * dims[1] + c[1]) * dims[2] + c[2]
    return class_ids, idx


def periodic_p_mat(points, Nx, Ny, Nz, Lx=1.0, Ly=1.0, Lz=1.0,
                   periodic_axes=(0, 1, 2), fixed_class=(0, 0, 0), fixed_dofs=()):
    """CSR constraint matrix u_full = P @ u_reduced.

    One reduced DOF per (periodic equivalence class, component); classes on
    a periodic boundary own 2/4/8 member nodes that all share the column.

    Zero rows in P pin the corresponding full DOFs to w = 0 and remove them
    from the reduced space:
    - ``fixed_class``: one whole equivalence class (3 DOFs), e.g. the XYZ
      corner class removing the three rigid translations;
    - ``fixed_dofs``: individual full-DOF indices, e.g. w_z = 0 on the two
      loaded z-surfaces of an XY-periodic cell. Any (class, component)
      group containing a pinned DOF is excluded as a whole, which for
      periodically consistent pinning is exactly that group.

    Returns (P, class_ids, fixed_class_id) with ``fixed_class_id = None``
    when ``fixed_class`` is None.
    """
    class_ids, _ = periodic_node_classes(points, Nx, Ny, Nz, Lx, Ly, Lz,
                                         periodic_axes)
    per = [a in periodic_axes for a in range(3)]
    Ns = (Nx, Ny, Nz)
    dims = onp.array([Ns[a] if per[a] else Ns[a] + 1 for a in range(3)])
    num_classes = int(onp.prod(dims))
    num_nodes = len(points)
    N_full = 3 * num_nodes

    # one (class, component) group shares a single reduced DOF
    comp = onp.tile(onp.arange(3), num_nodes)
    group_ids = onp.repeat(class_ids, 3) * 3 + comp

    excluded = set()
    fixed_id = None
    if fixed_class is not None:
        fc = [fixed_class[a] % dims[a] for a in range(3)]
        fixed_id = int((fc[0] * dims[1] + fc[1]) * dims[2] + fc[2])
        excluded.update(fixed_id * 3 + c for c in range(3))
    if len(fixed_dofs):
        excluded.update(onp.asarray(group_ids)[
            onp.asarray(list(fixed_dofs), dtype=int)].tolist())
    N_red = 3 * num_classes - len(excluded)

    red_of_group = onp.full(3 * num_classes, -1, dtype=onp.int64)
    nxt = 0
    for g in range(3 * num_classes):
        if g in excluded:
            continue
        red_of_group[g] = nxt
        nxt += 1

    keep = red_of_group[group_ids] >= 0
    rows = onp.arange(N_full)
    P = scipy.sparse.csr_array(
        (onp.ones(int(keep.sum())), (rows[keep], red_of_group[group_ids[keep]])),
        shape=(N_full, N_red))
    return P, class_ids, fixed_id


def xy_compression_fixed_dofs(points, Nx, Ny, Nz, Lz=1.0, corner=(0.0, 0.0, 0.0)):
    """Zero-fluctuation pins for XY-periodic uniaxial loading.

    w_z = 0 on the z = 0 and z = L surfaces (flat loaded faces; the lateral
    components of those faces stay free) and w_x = w_y = 0 at one bottom
    corner, which removes the two remaining rigid translations (the in-plane
    rotation is already excluded by the XY periodicity itself).
    """
    pts = onp.asarray(points)
    iz = onp.round(pts[:, 2] / Lz * Nz).astype(int)
    surface = onp.where((iz == 0) | (iz == Nz))[0]
    dist = onp.abs(pts - onp.asarray(corner, dtype=float)).max(axis=1)
    corner_node = int(onp.argmin(dist))
    assert dist[corner_node] < 1e-10, "corner node not found on the grid"
    return onp.concatenate([3 * surface + 2,
                            [3 * corner_node, 3 * corner_node + 1]])


def make_periodic_problem(Nx, Ny, Nz, H_macro=None, sine_force_amplitude=None,
                          cell_size=1.0, periodic_axes=(0, 1, 2),
                          fixed_class=(0, 0, 0), fixed_dofs=()):
    """Periodic unit cell: HEX8 grid of Nx x Ny x Nz cells.

    ``periodic_axes`` selects the periodic directions; ``H_macro`` sets the
    macroscopic gradient; ``sine_force_amplitude`` adds the verification body
    force f_x = (lambda+2mu) A k^2 sin(k x), k = 2*pi/L (x direction).
    """
    from jax_fem.generate_mesh import box_mesh, get_meshio_cell_type, Mesh
    from fem import ELE_TYPE

    cell_type = get_meshio_cell_type(ELE_TYPE)
    meshio_mesh = box_mesh(Nx, Ny, Nz, cell_size, cell_size, cell_size)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])
    problem = PeriodicLinearElasticityCube(mesh, vec=3, dim=3, ele_type=ELE_TYPE,
                                           dirichlet_bc_info=[[], [], []])
    if callable(fixed_dofs):
        # 一致性由调用方保证:被钉扎的组内所有周期成员一起传入
        fixed_dofs = fixed_dofs(problem.fe.points)
    H = jnp.zeros((3, 3)) if H_macro is None else jnp.asarray(H_macro)
    problem.set_params(H, sine_force_amplitude, cell_size)
    P, class_ids, fixed_id = periodic_p_mat(
        problem.fe.points, Nx, Ny, Nz,
        Lx=cell_size, Ly=cell_size, Lz=cell_size,
        periodic_axes=periodic_axes, fixed_class=fixed_class,
        fixed_dofs=fixed_dofs)
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

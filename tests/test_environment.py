"""M0 环境自检:依赖导入、JAX 基础计算与 JAX-FEM 有限元 smoke test。"""
import sys

import numpy as onp
import jax
import jax.numpy as jnp


def test_python_version():
    assert sys.version_info >= (3, 12)


def test_core_imports():
    import scipy
    import meshio
    import gmsh
    import jax_fem

    print("jax_fem from:", jax_fem.__file__)


def test_jax_versions_match():
    import jaxlib

    assert jax.__version__ == jaxlib.__version__


def test_jax_array_and_grad():
    x = jnp.arange(4, dtype=jnp.float64)
    assert onp.allclose(onp.asarray(x * 2), onp.arange(4) * 2)

    # d/dt sin(t) 在 t=1 处应等于 cos(1)
    g = jax.grad(lambda t: jnp.sin(t))(jnp.array(1.0))
    assert abs(float(g) - float(onp.cos(1.0))) < 1e-12


def test_device_visible():
    devices = jax.devices()
    assert len(devices) >= 1
    print("jax backend:", jax.default_backend(), "| devices:", devices)


# ---------- JAX-FEM 有限元 smoke test(官方 Quickstart 同款 Poisson 弱形式,缩小网格) ----------
from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh


class Poisson(Problem):
    def get_tensor_map(self):
        return lambda x: x

    def get_mass_map(self):
        def mass_map(u, x):
            val = -jnp.array([10 * jnp.exp(-(jnp.power(x[0] - 0.5, 2) + jnp.power(x[1] - 0.5, 2)) / 0.02)])
            return val

        return mass_map


def test_fem_poisson_smoke():
    ele_type = "QUAD4"
    cell_type = get_meshio_cell_type(ele_type)
    L = 1.0
    meshio_mesh = rectangle_mesh(Nx=16, Ny=16, domain_x=L, domain_y=L)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    def left(p):
        return jnp.isclose(p[0], 0., atol=1e-5)

    def right(p):
        return jnp.isclose(p[0], L, atol=1e-5)

    def bottom(p):
        return jnp.isclose(p[1], 0., atol=1e-5)

    def top(p):
        return jnp.isclose(p[1], L, atol=1e-5)

    def dirichlet_val(p):
        return 0.

    dirichlet_bc_info = [[left, right, bottom, top], [0, 0, 0, 0],
                         [dirichlet_val] * 4]
    problem = Poisson(mesh=mesh, vec=1, dim=2, ele_type=ele_type,
                      dirichlet_bc_info=dirichlet_bc_info)
    sol = solver(problem)[0]

    u = onp.asarray(sol).reshape(-1)
    pts = onp.asarray(mesh.points)
    assert onp.all(onp.isfinite(u)), "解中含 NaN/Inf"

    on_boundary = (onp.isclose(pts[:, 0], 0., atol=1e-5) | onp.isclose(pts[:, 0], L, atol=1e-5) |
                   onp.isclose(pts[:, 1], 0., atol=1e-5) | onp.isclose(pts[:, 1], L, atol=1e-5))
    assert onp.allclose(u[on_boundary], 0., atol=1e-10), "Dirichlet 边界不满足"
    assert u[~on_boundary].min() > 0., "正源项下内域应为正"
    assert 0. < u.max() < 1.0

    print(f"FEM smoke OK: backend={jax.default_backend()}, nodes={len(pts)}, u_max={u.max():.6f}")

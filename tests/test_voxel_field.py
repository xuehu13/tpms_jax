"""Map invariants: constants, affine interior, centre values and periodic translations."""
import numpy as np
import jax.numpy as jnp
from voxel_field import periodic_trilinear


def test_constant_and_periodic_translation():
    values = jnp.full((4,4,4), .37)
    points = jnp.array([[0., .3, 1.], [-.01, .9, 1.2], [.31, .43, .72]])
    np.testing.assert_allclose(periodic_trilinear(values, points), .37, atol=1e-14)
    varied = jnp.arange(64., dtype=jnp.float64).reshape(4,4,4)/63
    np.testing.assert_allclose(periodic_trilinear(varied, points), periodic_trilinear(varied, points+jnp.array([1.,-1.,2.])), atol=1e-14)


def test_centres_and_affine_xyz_interior():
    x = (jnp.arange(4., dtype=jnp.float64)+.5)/4
    centres = jnp.stack(jnp.meshgrid(x,x,x,indexing='ij'), axis=-1)
    values = .1 + .1*centres[...,0] + .2*centres[...,1] + .3*centres[...,2]
    np.testing.assert_allclose(periodic_trilinear(values, centres), values, atol=1e-14)
    points = jnp.array([[.23,.51,.72], [.71,.29,.38]])
    expected = .1 + .1*points[:,0] + .2*points[:,1] + .3*points[:,2]
    np.testing.assert_allclose(periodic_trilinear(values, points), expected, atol=1e-14)


def test_signed_field_projection_symmetry_and_periodicity():
    from voxel_field import sample_implicit
    from geometry import project_field
    g = sample_implicit(8)
    assert float(g.min()) < -1 and float(g.max()) > 1
    pts = jnp.array([[0.,.24,.78],[-.03,.48,1.2],[.52,.3,.82]])
    q = periodic_trilinear(g, pts)
    phi = project_field(q,.541062,40.)
    np.testing.assert_allclose(phi, project_field(-q,.541062,40.), atol=1e-14)
    np.testing.assert_allclose(phi, project_field(periodic_trilinear(g,pts+1),.541062,40.), atol=1e-13)
    assert np.all((np.asarray(phi)>=0)&(np.asarray(phi)<=1))

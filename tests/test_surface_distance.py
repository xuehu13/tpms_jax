import numpy as np
import jax
import jax.numpy as jnp
from surface_distance import PeriodicSurfaceDistance,thickness_occupancy


def test_periodic_plane_distance_including_wrapped_images():
    v=np.array([[0,0,.1],[1,0,.1],[1,1,.1],[0,1,.1]])
    f=np.array([[0,1,2],[0,2,3]])
    surface=PeriodicSurfaceDistance(v,f)
    q=np.array([[.3,.4,.12],[.4,.8,.99],[1.3,-.6,1.12]])
    d,face,hit=surface.query(q,chunk_size=2,details=True)
    np.testing.assert_allclose(d,[.02,.11,.02],atol=1e-14)
    np.testing.assert_allclose(d,surface.query(q+np.array([2,-1,3])),atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(q%1-hit,axis=1),d,atol=1e-14)
    assert np.all((face>=0)&(face<2))


def test_triangle_distance_uses_triangle_interior_not_vertex_cloud():
    surface=PeriodicSurfaceDistance(np.array([[.2,.2,.5],[.8,.2,.5],[.2,.8,.5]]),np.array([[0,1,2]]))
    np.testing.assert_allclose(surface.query(np.array([[.3,.3,.53]])),[.03],atol=1e-14)


def test_physical_width_and_thickness_derivative():
    t=.05;width=.005;ell=width/(2*np.log(9))
    d=jnp.array([t/2-width/2,t/2,t/2+width/2])
    np.testing.assert_allclose(thickness_occupancy(d,t,ell),[.9,.5,.1],atol=1e-14)
    derivative=jax.jacfwd(lambda thick:thickness_occupancy(d,thick,ell))(t)
    step=1e-6
    finite=(thickness_occupancy(d,t+step,ell)-thickness_occupancy(d,t-step,ell))/(2*step)
    np.testing.assert_allclose(derivative,finite,rtol=3e-8)
    assert np.all(np.asarray(derivative)>0)

import numpy as np
import jax, jax.numpy as jnp
import binary_gyroid as binary
from design_fem import GyroidDesign


def test_stationary_custom_vjp_for_coupled_loss():
    model=GyroidDesign(4,beta=40,emin_ratio=1e-4)
    theta=jnp.array([.541062,.025,-.02,.015])
    value=model.stationary_observables()
    loss=lambda q:jnp.dot(value(q),jnp.array([.7,.3]))
    actual,grad=jax.value_and_grad(loss)(theta)
    d=jnp.array([.5,-.4,.6,-.3])/jnp.sqrt(.86);h=3e-5
    plus=model.forward(theta+h*d);minus=model.forward(theta-h*d)
    assert plus['status']==minus['status']=='ok'
    fd=np.dot((np.array(plus['values'][:2])-np.array(minus['values'][:2]))/(2*h),[.7,.3])
    np.testing.assert_allclose(grad@d,fd,rtol=1e-4,atol=1e-8)
    assert np.isfinite(actual)


def test_two_affine_width_interfaces_are_clipped_independently():
    xy=np.array([[0.,0.],[1.,0.],[0.,1.]])
    g=-2+2*xy[:,0]+4*xy[:,1];c=.3+.2*xy[:,0]-.1*xy[:,1]
    polygon=[np.r_[p,0.,gg,cc] for p,gg,cc in zip(xy,g,c)]
    cut=binary.clip_polygon(binary.clip_polygon(polygon,lambda p:-p[4],True),lambda p:p[4],False)
    points=np.array(cut)
    np.testing.assert_allclose(points[:,3],-2+2*points[:,0]+4*points[:,1],atol=1e-15)
    np.testing.assert_allclose(points[:,4],.3+.2*points[:,0]-.1*points[:,1],atol=1e-15)
    assert np.all(abs(points[:,3])<=points[:,4]+1e-15)
    assert np.any(np.isclose(points[:,3],points[:,4])) and np.any(np.isclose(points[:,3],-points[:,4]))


def test_periodic_width_mesh_is_closed_connected_and_matched():
    q=np.array([.541038,.035,0.,0.])
    points,cells=binary.linear_domain(4,c=q)
    audit=binary.audit_linear(points,cells,c=q)
    assert audit['periodic_connected_components']==1 and audit['min_detJ']>0
    sample=np.array([[.13,.24,.41],[.43,.57,.68]])
    np.testing.assert_allclose(binary.width_numpy(sample,q),binary.width_numpy(sample+[1,-1,2],q),atol=1e-14,rtol=0)

"""Check known limits and local total derivatives; expose rotation limitations."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hyperelastic_fem import (MU, KAPPA, neo_hookean_energy, first_piola,
    linearized_neo_hookean_energy, void_interpolation_gamma,
    interpolated_neo_hookean_energy, interpolated_first_piola)

ETA = 1e-4
F = jnp.array([[.93,.15,-.04],[.02,1.06,.03],[.03,-.02,.81]])

def test_endpoints_preserve_solid_and_linear_void():
    assert float(void_interpolation_gamma(0.)) == 0.
    assert float(void_interpolation_gamma(1.)) == 1.
    np.testing.assert_allclose(interpolated_neo_hookean_energy(F,1.),neo_hookean_energy(F),rtol=1e-13)
    np.testing.assert_allclose(interpolated_first_piola(F,1.),first_piola(F),rtol=1e-13)
    np.testing.assert_allclose(interpolated_neo_hookean_energy(F,0.),ETA*linearized_neo_hookean_energy(F),rtol=1e-13)

@pytest.mark.parametrize('rho',[0.,1e-4,.005,.01,.02,.5,1.])
def test_reference_tangent_preserved_at_every_occupancy(rho):
    identity=jnp.eye(3)
    reference=jax.hessian(neo_hookean_energy)(identity)
    candidate=jax.hessian(interpolated_neo_hookean_energy,argnums=0)(identity,rho)
    np.testing.assert_allclose(candidate,(ETA+(1-ETA)*rho)*reference,atol=3e-13,rtol=3e-12)
    np.testing.assert_allclose(interpolated_first_piola(identity,rho),0.,atol=2e-14)

def test_energy_stress_and_occupancy_derivatives_include_switch():
    rho=.00925; gamma=void_interpolation_gamma(rho); mapped=jnp.eye(3)+gamma*(F-jnp.eye(3))
    linear_stress=jax.grad(linearized_neo_hookean_energy)(F)
    expected=(ETA+(1-ETA)*rho)*(gamma*first_piola(mapped)+(1-gamma**2)*linear_stress)
    np.testing.assert_allclose(interpolated_first_piola(F,rho),expected,rtol=1e-12,atol=1e-14)
    direction=jnp.array([[.02,.03,.01],[.04,-.01,.02],[-.02,.01,.03]])
    h=1e-6
    ad=jnp.sum(interpolated_first_piola(F,rho)*direction)
    fd=(interpolated_neo_hookean_energy(F+h*direction,rho)-interpolated_neo_hookean_energy(F-h*direction,rho))/(2*h)
    np.testing.assert_allclose(ad,fd,rtol=2e-7,atol=1e-10)
    # Occupancy affects both stiffness and mapped kinematics, not only scale.
    ad_rho=jax.grad(interpolated_neo_hookean_energy,argnums=1)(F,rho)
    eps=1e-7
    fd_rho=(interpolated_neo_hookean_energy(F,rho+eps)-interpolated_neo_hookean_energy(F,rho-eps))/(2*eps)
    np.testing.assert_allclose(ad_rho,fd_rho,rtol=3e-7,atol=1e-9)

def test_deep_void_inversion_has_defined_extension_without_clipping_wall():
    inverted=jnp.diag(jnp.array([-2.1,.7,1.2]))
    assert float(jnp.linalg.det(inverted)) < 0
    for rho in [0.,.00015010447265510502]:
        mapped=jnp.eye(3)+void_interpolation_gamma(rho)*(inverted-jnp.eye(3))
        assert float(jnp.linalg.det(mapped)) > 0
        assert np.isfinite(float(interpolated_neo_hookean_energy(inverted,rho)))
        assert np.isfinite(np.asarray(interpolated_first_piola(inverted,rho))).all()
    assert not np.isfinite(float(interpolated_neo_hookean_energy(inverted,1.)))

def test_rotation_defect_is_exposed_not_certified_as_objectivity():
    rotation=jnp.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    np.testing.assert_allclose(interpolated_neo_hookean_energy(rotation,1.),0.,atol=1e-13)
    np.testing.assert_allclose(interpolated_first_piola(rotation,1.),0.,atol=1e-13)
    expected=ETA*(2*MU/3+2*KAPPA)
    np.testing.assert_allclose(interpolated_neo_hookean_energy(rotation,0.),expected,rtol=1e-13)
    assert expected > 0  # A nonzero pure-rotation energy is a known limitation.

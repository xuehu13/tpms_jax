"""Physical limits, objectivity, singular-domain and derivative checks."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hyperelastic_fem import (MU,KAPPA,neo_hookean_energy,first_piola,
    stable_neo_hookean_extension,void_nh_weight,objective_void_energy,objective_void_first_piola)

ETA=1e-4
F=jnp.array([[.93,.15,-.04],[.02,1.06,.03],[.03,-.02,.81]])
D=jnp.array([[.02,.03,.01],[.04,-.01,.02],[-.02,.01,.03]])
def rotation(angle):
    a=np.deg2rad(angle);c,s=np.cos(a),np.sin(a)
    return jnp.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])

@pytest.mark.parametrize('rho',[0.,.00015010447265510502,.001,.0055,.01,.5,1.])
def test_reference_tangent_and_zero_stress(rho):
    eye=jnp.eye(3)
    np.testing.assert_allclose(objective_void_energy(eye,rho),0.,atol=1e-15)
    np.testing.assert_allclose(objective_void_first_piola(eye,rho),0.,atol=1e-14)
    candidate=jax.hessian(objective_void_energy,argnums=0)(eye,rho)
    old=(ETA+(1-ETA)*rho)*jax.hessian(neo_hookean_energy)(eye)
    np.testing.assert_allclose(candidate,old,rtol=3e-12,atol=3e-13)

def test_matched_nh_unchanged_and_deep_void_endpoint():
    for rho in [.01,.05,.5,.95,1.]:
        scale=ETA+(1-ETA)*rho
        np.testing.assert_allclose(objective_void_energy(F,rho),scale*neo_hookean_energy(F),rtol=1e-13)
        np.testing.assert_allclose(objective_void_first_piola(F,rho),scale*first_piola(F),rtol=1e-13)
    np.testing.assert_allclose(objective_void_energy(F,0.),ETA*stable_neo_hookean_extension(F),rtol=1e-13)

@pytest.mark.parametrize('rho',[0.,.00015,.0055,.01,1.])
def test_energy_objectivity_and_stress_covariance(rho):
    for angle in [15.,60.,90.,173.]:
        Q=rotation(angle)
        np.testing.assert_allclose(objective_void_energy(Q,rho),0.,atol=2e-14)
        np.testing.assert_allclose(objective_void_first_piola(Q,rho),0.,atol=2e-13)
        np.testing.assert_allclose(objective_void_energy(Q@F,rho),objective_void_energy(F,rho),rtol=1e-12,atol=2e-14)
        np.testing.assert_allclose(objective_void_first_piola(Q@F,rho),Q@objective_void_first_piola(F,rho),rtol=1e-12,atol=2e-13)

@pytest.mark.parametrize('F_bad',[jnp.diag(jnp.array([-2.1,.7,1.2])),
    jnp.diag(jnp.array([1.,1.,0.])),jnp.diag(jnp.array([1.,0.,0.])),jnp.zeros((3,3))])
def test_inverted_and_singular_void_derivatives_defined(F_bad):
    for rho in [0.,.00015010447265510502,.001]:
        assert np.isfinite(float(objective_void_energy(F_bad,rho)))
        assert np.isfinite(np.asarray(objective_void_first_piola(F_bad,rho))).all()
        assert np.isfinite(np.asarray(jax.hessian(objective_void_energy,argnums=0)(F_bad,rho))).all()
        assert np.isfinite(float(jax.grad(objective_void_energy,argnums=1)(F_bad,rho)))
        # Second derivatives also agree with independent stress differences
        # through rank loss; finite values alone would not establish this.
        h=1e-6
        tangent=jax.jvp(lambda f:objective_void_first_piola(f,rho),(F_bad,),(D,))[1]
        difference=(objective_void_first_piola(F_bad+h*D,rho)-objective_void_first_piola(F_bad-h*D,rho))/(2*h)
        np.testing.assert_allclose(tangent,difference,rtol=2e-6,atol=1e-10)
        Q=rotation(60.)
        np.testing.assert_allclose(objective_void_energy(Q@F_bad,rho),objective_void_energy(F_bad,rho),rtol=2e-13,atol=1e-14)

def test_invalid_actual_j_in_uncontinued_nh_domain_is_not_hidden():
    bad=jnp.diag(jnp.array([-2.1,.7,1.2]))
    assert float(jnp.linalg.det(bad))<0
    for rho in [.01,.5,1.]:
        assert not np.isfinite(float(objective_void_energy(bad,rho)))

@pytest.mark.parametrize('rho',[.00015,.0055,.01,.5])
def test_local_state_and_occupancy_total_derivatives(rho):
    h=1e-6
    ad=float(jnp.sum(objective_void_first_piola(F,rho)*D))
    fd=float((objective_void_energy(F+h*D,rho)-objective_void_energy(F-h*D,rho))/(2*h))
    np.testing.assert_allclose(ad,fd,rtol=5e-7,atol=2e-10)
    h=1e-7
    ad=float(jax.grad(objective_void_energy,argnums=1)(F,rho))
    fd=float((objective_void_energy(F,rho+h)-objective_void_energy(F,rho-h))/(2*h))
    np.testing.assert_allclose(ad,fd,rtol=5e-7,atol=2e-9)

def test_gate_is_c2_at_both_boundaries():
    for rho,expected in [(.001,0.),(.01,1.)]:
        np.testing.assert_allclose(void_nh_weight(rho),expected,atol=1e-14)
        np.testing.assert_allclose(jax.grad(void_nh_weight)(rho),0.,atol=1e-12)
        np.testing.assert_allclose(jax.grad(jax.grad(void_nh_weight))(rho),0.,atol=1e-9)
    assert float(jax.grad(void_nh_weight)(.0055))>0

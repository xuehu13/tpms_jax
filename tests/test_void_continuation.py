"""C2 energy continuation, objective stress and mixed-tail derivatives."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hyperelastic_fem import (MU,KAPPA,void_nh_weight,void_nh_cutoff,
    continued_void_nh_energy,objective_void_energy,objective_void_first_piola,
    neo_hookean_energy,stable_neo_hookean_extension)

D=jnp.array([[.02,.03,.01],[.04,-.01,.02],[-.02,.01,.03]])
Q=jnp.array([[.5,-np.sqrt(3)/2,0.],[np.sqrt(3)/2,.5,0.],[0.,0.,1.]])

def test_cutoff_preserves_gate_boundaries_and_has_c2_zero_endpoint():
    for phi,value in [(.001,.1),(.01,0.)]:
        np.testing.assert_allclose(void_nh_cutoff(phi),value,atol=1e-15)
        np.testing.assert_allclose(jax.grad(void_nh_cutoff)(phi),0.,atol=1e-11)
        np.testing.assert_allclose(jax.grad(jax.grad(void_nh_cutoff))(phi),0.,atol=1e-8)
    assert 0<float(void_nh_cutoff(.004))<.1

@pytest.mark.parametrize('phi',[.0011,.004,.008,.00999])
def test_mixed_void_singular_and_inverted_energy_stress_tangent_are_finite(phi):
    for F in [jnp.diag(jnp.array([1.,1.,-.03])),jnp.diag(jnp.array([1.,1.,0.])),
              jnp.diag(jnp.array([1.,0.,0.])),jnp.zeros((3,3))]:
        W=objective_void_energy(F,phi);P=objective_void_first_piola(F,phi)
        A=jax.hessian(objective_void_energy,argnums=0)(F,phi)
        dp=jax.grad(objective_void_energy,argnums=1)(F,phi)
        assert np.isfinite(float(W)) and np.isfinite(np.asarray(P)).all()
        assert np.isfinite(np.asarray(A)).all() and np.isfinite(float(dp))
        np.testing.assert_allclose(objective_void_energy(Q@F,phi),W,rtol=5e-12,atol=2e-13)
        np.testing.assert_allclose(objective_void_first_piola(Q@F,phi),Q@P,rtol=5e-12,atol=2e-11)
        # Near phi=.01 the joining volume tends to zero. Keep an independent
        # local difference inside that curvature scale, rather than taking
        # a fixed step across the join and calling it a tangent comparison.
        f=np.asarray(F);cof=np.stack([np.cross(f[:,1],f[:,2]),
            np.cross(f[:,2],f[:,0]),np.cross(f[:,0],f[:,1])],axis=1)
        dJ=abs(float(np.sum(cof*np.asarray(D))))
        distance=abs(float(np.linalg.det(f))-float(void_nh_cutoff(phi)))
        eps=min(1e-6,.005*distance/dJ) if dJ>1e-20 else 1e-6
        ad=jax.jvp(lambda x:objective_void_first_piola(x,phi),(F,),(D,))[1]
        fd=(objective_void_first_piola(F+eps*D,phi)-objective_void_first_piola(F-eps*D,phi))/(2*eps)
        np.testing.assert_allclose(ad,fd,rtol=2e-5,atol=2e-8)

@pytest.mark.parametrize('phi',[.004,.008])
def test_energy_stress_and_tangent_match_original_at_positive_join(phi):
    F=jnp.diag(jnp.array([1.,1.,void_nh_cutoff(phi)]))
    np.testing.assert_allclose(continued_void_nh_energy(F,phi),neo_hookean_energy(F),rtol=3e-13)
    np.testing.assert_allclose(jax.grad(continued_void_nh_energy)(F,phi),jax.grad(neo_hookean_energy)(F),rtol=3e-12,atol=2e-10)
    np.testing.assert_allclose(jax.hessian(continued_void_nh_energy)(F,phi),jax.hessian(neo_hookean_energy)(F),rtol=3e-12,atol=2e-8)
    # Joint state/occupancy derivative through a moving branch boundary.
    fun=lambda x:continued_void_nh_energy(F+x*D,phi+x*.0002)
    eps=1e-6
    np.testing.assert_allclose(jax.grad(fun)(0.),(fun(eps)-fun(-eps))/(2*eps),rtol=5e-6,atol=3e-8)
    np.testing.assert_allclose(jax.grad(jax.grad(fun))(0.),
        (jax.grad(fun)(eps)-jax.grad(fun)(-eps))/(2*eps),rtol=3e-5,atol=3e-7)

def test_safe_region_and_true_wall_match_original_candidate_even_at_small_j():
    F=jnp.array([[.93,.15,-.04],[.02,1.06,.03],[.03,-.02,.81]])
    def previous(f,phi):
        h=void_nh_weight(phi)
        return (1e-4+.9999*phi)*(h*neo_hookean_energy(f)+(1-h)*stable_neo_hookean_extension(f))
    for phi in [.0011,.004,.0095,.01,.4]:
        np.testing.assert_allclose(objective_void_energy(F,phi),previous(F,phi),rtol=2e-13)
        np.testing.assert_allclose(objective_void_first_piola(F,phi),jax.grad(previous)(F,phi),rtol=3e-12,atol=1e-13)
    collapsed_but_positive=jnp.diag(jnp.array([1.,1.,.0001]))
    for phi in [.01,.05,1.]:
        scale=1e-4+.9999*phi
        np.testing.assert_allclose(objective_void_energy(collapsed_but_positive,phi),scale*neo_hookean_energy(collapsed_but_positive),rtol=3e-12)

def test_inverted_mixed_tail_local_state_and_occupancy_derivatives():
    F=jnp.array([[1.,.08,.02],[.02,.9,.03],[.01,-.02,-.03]])
    phi=.0035;eps=1e-7
    np.testing.assert_allclose(jnp.sum(objective_void_first_piola(F,phi)*D),
        (objective_void_energy(F+eps*D,phi)-objective_void_energy(F-eps*D,phi))/(2*eps),rtol=4e-6,atol=2e-9)
    np.testing.assert_allclose(jax.grad(objective_void_energy,argnums=1)(F,phi),
        (objective_void_energy(F,phi+eps)-objective_void_energy(F,phi-eps))/(2*eps),rtol=4e-6,atol=2e-8)

def test_negative_j_occupancy_derivative_is_now_c2_at_lower_boundary():
    F=jnp.diag(jnp.array([1.,1.,-.03]));phi=.001
    # At the lower gate boundary both energies and their derivatives exist
    # on both sides. This was undefined just above .001 in the old kernel.
    fun=lambda x:objective_void_energy(F,x)
    assert np.isfinite(float(jax.grad(jax.grad(fun))(phi)))
    eps=1e-8
    np.testing.assert_allclose(jax.grad(fun)(phi),(fun(phi+eps)-fun(phi-eps))/(2*eps),rtol=2e-5)
    np.testing.assert_allclose(jax.grad(jax.grad(fun))(phi),
        (jax.grad(fun)(phi+eps)-jax.grad(fun)(phi-eps))/(2*eps),atol=2e-2)

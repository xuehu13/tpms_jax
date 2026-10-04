import numpy as np
import jax
import jax.numpy as jnp
from hyperelastic_fem import (DensityHyperelasticity,make_density_hyperelastic_problem,
                             neo_hookean_energy,first_piola,finite_response,reduced_guess)


def test_scaled_material_point_tangent_recovers_linear_density():
    scale=1e-4+(1-1e-4)*.37
    tangent=jax.jacfwd(lambda grad:scale*first_piola(jnp.eye(3)+grad))(jnp.zeros((3,3)))
    grad=jnp.array([[.2,.3,.1],[-.1,-.4,.2],[.4,-.2,.1]])
    actual=jnp.einsum('ijkl,kl->ij',tangent,grad)
    mu=10/2.6;lam=10*.3/(1.3*.4)
    expected=scale*(lam*jnp.trace(grad)*jnp.eye(3)+mu*(grad+grad.T))
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)


def test_reference_Gauss_field_and_full_material_reduction():
    from geometry import density
    p=make_density_hyperelastic_problem(2)
    np.testing.assert_array_equal(p.rho,density(p.physical_quad_points,.541062,40))
    p.set_params(jnp.diag(jnp.array([0.,0.,-.2])),1.,1e-4)
    result=finite_response(p,[jnp.zeros_like(p.fe.points)])
    np.testing.assert_allclose(result['energy'],neo_hookean_energy(jnp.diag(jnp.array([1.,1.,.8]))),rtol=1e-12)
    np.testing.assert_allclose(result['mean_first_piola'],first_piola(jnp.diag(jnp.array([1.,1.,.8]))),atol=1e-12)


def test_invalid_newton_trial_is_rejected_without_detF_clipping():
    import pytest
    p=make_density_hyperelastic_problem(2);p.set_params(jnp.diag(jnp.array([0.,0.,-1.1])),p.rho)
    with pytest.raises(ValueError,match='Nonpositive'):p.newton_update([jnp.zeros_like(p.fe.points)])
    assert p.trial_detF['all_min']<0 and hasattr(p,'invalid_trial_w')


def test_nonuniform_equilibrium_force_and_weighted_energy():
    from fem import solve
    p=make_density_hyperelastic_problem(4);p.set_params(jnp.diag(jnp.array([0.,0.,-.001])),p.rho)
    sol=solve(p,{'newton':{'tol':1e-11,'rel_tol':1e-11,'linear':{'spsolve_solver':{}}}})
    result=finite_response(p,sol)
    assert result['J_min']>0 and result['reduced_residual_l2']<1e-9
    assert abs(result['Fz_top']+result['Fz_bottom'])<1e-9
    assert abs(result['Fz_top']-result['mean_first_piola'][2][2])<1e-9
    F=jnp.eye(3)+p.H_macro+p.fe.sol_to_grad(sol[0])
    W=jax.vmap(neo_hookean_energy)(F.reshape((-1,3,3))).reshape(p.rho.shape)
    weights=jnp.asarray(p.JxW)[:,0,:]
    separated=jnp.sum((p.eta*W+(1-p.eta)*p.rho*W)*weights)
    np.testing.assert_allclose(result['energy'],separated,atol=1e-14)


def test_traceable_parameter_update_residual_derivative_and_validation():
    import pytest
    p=make_density_hyperelastic_problem(2)
    H=jnp.diag(jnp.array([0.,0.,-.01]));rho=jnp.full(p.rho.shape,.4)
    direction=jnp.linspace(-.2,.2,rho.size).reshape(rho.shape)
    w=[jnp.zeros_like(p.fe.points)]

    def residual(field):
        p._set_params_jax(H,field)
        return p.compute_residual(w)[0]

    try:
        _,derivative=jax.jvp(residual,(rho,),(direction,))
        step=1e-5
        difference=(residual(rho+step*direction)-residual(rho-step*direction))/(2*step)
        assert np.linalg.norm(np.asarray(derivative))>0
        np.testing.assert_allclose(derivative,difference,rtol=1e-8,atol=1e-11)
    finally:
        p.set_params(H,rho)

    for invalid in [jnp.nan,-.1,1.1]:
        with pytest.raises(ValueError,match='rho'):p.set_params(H,invalid)
    for eta in [0.,jnp.nan,1.1]:
        with pytest.raises(ValueError,match='eta'):p.set_params(H,rho,eta)
    with pytest.raises(ValueError,match='H_macro'):p.set_params(jnp.zeros(3),rho)

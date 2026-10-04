"""Constitutive, finite-rotation and nonzero-Newton evidence; no design gradients."""
import numpy as np
import jax
import jax.numpy as jnp
from fem import LAMBDA, MU as LINEAR_MU, solve
from hyperelastic_fem import (MU,KAPPA,neo_hookean_energy,first_piola,
                              make_hyperelastic_problem,reduced_guess,finite_response)


def test_energy_identity_and_objectivity():
    theta=.63;Q=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1.]])
    F=np.array([[1.,.17,0],[.04,.93,.02],[0,.06,.83]])
    np.testing.assert_allclose(neo_hookean_energy(jnp.eye(3)),0,atol=1e-14)
    np.testing.assert_allclose(first_piola(jnp.eye(3)),0,atol=1e-14)
    np.testing.assert_allclose(neo_hookean_energy(Q),0,atol=1e-14)
    np.testing.assert_allclose(neo_hookean_energy(Q@F),neo_hookean_energy(F),atol=1e-13)
    np.testing.assert_allclose(first_piola(Q@F),Q@first_piola(F),atol=1e-13)


def test_independent_general_first_piola_formula():
    F=np.array([[1.,.12,.02],[.03,.97,.04],[.01,0,.8]])
    J=np.linalg.det(F);invT=np.linalg.inv(F).T;I1=np.sum(F*F)
    expected=MU*J**(-2/3)*(F-I1/3*invT)+KAPPA*(J-1)*J*invT
    np.testing.assert_allclose(first_piola(F),expected,atol=1e-13)


def test_tangent_at_identity_recovers_linear_elasticity():
    A=np.array([[.02,.13,-.02],[.04,-.05,.01],[.01,-.03,.07]])
    _,dP=jax.jvp(first_piola,(jnp.eye(3),),(jnp.asarray(A),))
    expected=LAMBDA*np.trace(A)*np.eye(3)+LINEAR_MU*(A+A.T)
    np.testing.assert_allclose(dP,expected,atol=1e-13)


def test_finite_strain_tangent_matches_independent_difference():
    F=np.array([[1.,.1,0],[0,1.,.03],[0,0,.8]]);A=np.array([[.1,.2,0],[-.07,-.1,.02],[.03,0,.05]])
    _,tangent=jax.jvp(first_piola,(jnp.asarray(F),),(jnp.asarray(A),))
    h=1e-6;difference=(np.asarray(first_piola(F+h*A))-np.asarray(first_piola(F-h*A)))/(2*h)
    np.testing.assert_allclose(tangent,difference,atol=1e-9,rtol=1e-8)


def test_simple_shear_energy_and_invalid_inversion():
    F=np.eye(3);F[0,1]=.4
    np.testing.assert_allclose(neo_hookean_energy(F),MU*.4**2/2,atol=1e-14)
    assert not np.isfinite(float(neo_hookean_energy(jnp.diag(jnp.array([1.,1.,-.1])))))


def test_nonzero_fluctuation_newton_converges_at_twenty_percent():
    problem=make_hyperelastic_problem(2);problem.set_params(jnp.diag(jnp.array([0.,0.,-.2])))
    pts=np.asarray(problem.fe.points)
    w=np.zeros_like(pts);w[:,0]=.003*(np.cos(2*np.pi*pts[:,0])-1)*np.sin(np.pi*pts[:,2])
    w[:,2]=.003*(np.cos(2*np.pi*pts[:,0])-1)*np.sin(np.pi*pts[:,2])
    q=reduced_guess(problem,w);initial=[jnp.asarray((problem.P_mat@np.asarray(q)).reshape((-1,3)))]
    assert finite_response(problem,initial)['reduced_residual_l2']>1e-4
    sol=solve(problem,{'newton':{'tol':1e-11,'rel_tol':1e-11,'initial_guess':q,'linear':{'spsolve_solver':{}}}})
    row=finite_response(problem,sol)
    s=.8;expected=2*MU/3*(s**(1/3)-s**(-5/3))+KAPPA*(s-1)
    assert row['reduced_residual_l2']<1e-10
    np.testing.assert_allclose(row['Fz_top'],expected,atol=1e-10)
    np.testing.assert_allclose(sol[0],0,atol=1e-10)
    np.testing.assert_allclose([row['J_min'],row['J_max']],[.8,.8],atol=1e-10)

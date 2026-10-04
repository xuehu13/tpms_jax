"""Meaningful nonlinear implicit-gradient checks; not physical TPMS certification."""
import numpy as np
import jax
import jax.numpy as jnp
from fem import solve
from hyperelastic_fem import make_density_hyperelastic_problem,reduced_guess
from surface_distance import thickness_occupancy
from scripts.thin_thickness_gradient import force_partial,equilibrium_gradient


def test_thickness_force_adjoint_matches_library_vjp_and_equilibrium_difference():
    from jax_fem.solver import ad_wrapper
    L=10.;ell=.003;a=.05;t=.5
    p=make_density_hyperelastic_problem(2,periodic_axes=(0,1,2))
    xyz=p.physical_quad_points
    distance=jnp.asarray(.024+.008*jnp.cos(2*jnp.pi*xyz[...,0])+.004*jnp.sin(2*jnp.pi*xyz[...,2]))
    H=jnp.diag(jnp.array([0.,0.,-a]));eta=1e-4
    def set_design(t):p._set_params_jax(H,thickness_occupancy(distance,t/L,ell),eta)
    set_design(t)
    sol=solve(p,{'newton':{'tol':1e-11,'rel_tol':1e-11,'linear':{'spsolve_solver':{}}}})
    actual=equilibrium_gradient(p,distance,t,L,ell,a,sol[0],{'spsolve_solver':{}})
    assert actual['adjoint_relative_residual']<1e-9
    assert abs(actual['equilibrium_correction_N_per_mm'])>.001
    for h in (.001,.0005):
        q=[]
        for value in (t-h,t+h):
            set_design(value);p.trial_calls=0
            w=solve(p,{'newton':{'tol':1e-11,'rel_tol':1e-11,'initial_guess':reduced_guess(p,sol[0]),'linear':{'spsolve_solver':{}}}})[0]
            q.append(float(force_partial(p,distance,value,L,ell,w)))
        np.testing.assert_allclose((q[1]-q[0])/(2*h),actual['total_gradient_N_per_mm'],rtol=3e-4)
    p.set_params=set_design;p.trial_calls=0
    prediction=ad_wrapper(p,{'newton':{'tol':1e-11,'rel_tol':1e-11,'linear':{'spsolve_solver':{}}}}, {'spsolve_solver':{}})
    def objective(t):return force_partial(p,distance,t,L,ell,prediction(t)[0])
    derivative=float(jax.grad(objective)(jnp.asarray(t)))
    np.testing.assert_allclose(derivative,actual['total_gradient_N_per_mm'],rtol=1e-8,atol=1e-8)


def test_load_residual_jvp_is_force_displacement_partial_with_physical_units():
    L=10.;ell=.002;a=.01;t=.5
    p=make_density_hyperelastic_problem(2,periodic_axes=(0,1,2))
    distance=jnp.full(p.rho.shape,.022);w=jnp.asarray(.001*(np.cos(2*np.pi*np.asarray(p.fe.points))-1))
    def residual(a):
        p._set_params_jax(jnp.diag(jnp.array([0.,0.,-a])),thickness_occupancy(distance,t/L,ell))
        return p.compute_residual([w])[0]
    _,Ra=jax.jvp(residual,(jnp.asarray(a),),(jnp.asarray(1.),))
    p.set_params(jnp.diag(jnp.array([0.,0.,-a])),thickness_occupancy(distance,t/L,ell))
    explicit=jax.grad(lambda w:force_partial(p,distance,t,L,ell,w))(w)
    volume=jnp.asarray(p.JxW)[:,0,:].sum()
    np.testing.assert_allclose(explicit,L**2*Ra/volume,rtol=1e-10,atol=1e-10)

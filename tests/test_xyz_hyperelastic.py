"""XYZ homogeneous physics and field ownership; not TPMS certification."""
import numpy as np
import jax.numpy as jnp
from fem import solve
from hyperelastic_fem import make_hyperelastic_problem,make_density_hyperelastic_problem,finite_response,MU,KAPPA


def test_xyz_uniform_recovers_finite_compression_and_has_no_flat_end_pins():
    p=make_hyperelastic_problem(2,periodic_axes=(0,1,2))
    assert p.P_mat.shape[1]==3*2**3-3
    p.set_params(jnp.diag(jnp.array([0.,0.,-.05])))
    sol=solve(p,{'newton':{'tol':1e-11,'linear':{'spsolve_solver':{}}}})
    r=finite_response(p,sol);s=.95
    expected=2*MU/3*(s**(1/3)-s**(-5/3))+KAPPA*(s-1)
    np.testing.assert_allclose(r['Fz_top'],expected,rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose([r['J_min'],r['J_max']],[s,s],atol=1e-10)
    np.testing.assert_allclose(sol[0],0,atol=1e-10)


def test_xyz_callable_is_owned_by_solved_problem_and_scales_energy_and_force():
    called=[]
    def field(p):
        called.append(p)
        return jnp.full(p.physical_quad_points.shape[:2],.4)
    p=make_density_hyperelastic_problem(2,rho_quad=field,periodic_axes=(0,1,2),eta=.01)
    assert called==[p]
    p.set_params(jnp.diag(jnp.array([0.,0.,-.05])),p.rho,.01)
    sol=solve(p,{'newton':{'tol':1e-11,'linear':{'spsolve_solver':{}}}})
    r=finite_response(p,sol);s=.95;scale=.01+.99*.4
    expected_P=2*MU/3*(s**(1/3)-s**(-5/3))+KAPPA*(s-1)
    expected_W=MU/2*(s**(-2/3)*(2+s*s)-3)+KAPPA/2*(s-1)**2
    np.testing.assert_allclose(r['Fz_top'],scale*expected_P,atol=1e-10)
    np.testing.assert_allclose(r['energy'],scale*expected_W,atol=1e-10)


def test_repeated_xyz_tangent_matches_independent_residual_difference():
    from scipy.sparse import coo_matrix
    p=make_density_hyperelastic_problem(2,rho_quad=lambda p:.4+.2*np.cos(2*np.pi*p.physical_quad_points[...,0]),
                                        periodic_axes=(0,1,2))
    p.set_params(np.diag([0.,0.,-.02]),p.rho,p.eta)
    w=jnp.asarray(.001*(np.cos(2*np.pi*np.asarray(p.fe.points))-1))
    p.newton_update([w])
    tangent=coo_matrix((p.V.copy(),(p.I,p.J)),shape=(w.size,w.size)).tocsr()
    p.newton_update([w])
    repeated=coo_matrix((p.V,(p.I,p.J)),shape=tangent.shape).tocsr()
    np.testing.assert_allclose(repeated.toarray(),tangent.toarray(),atol=1e-13)
    direction=np.cos(2*np.pi*np.asarray(p.fe.points))-1
    h=1e-6
    finite=(np.asarray(p.compute_residual([w+h*direction])[0])-np.asarray(p.compute_residual([w-h*direction])[0]))/(2*h)
    np.testing.assert_allclose(tangent@direction.ravel(),finite.ravel(),rtol=1e-7,atol=1e-8)

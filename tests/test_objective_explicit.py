"""Consistent FEM work, material-domain protection, and a short path JVP."""
import importlib.util
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hyperelastic_fem import make_density_hyperelastic_problem,finite_response
spec=importlib.util.spec_from_file_location('explicit_entry',Path(__file__).resolve().parents[1]/'scripts/thin_target_explicit.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)

def make(phi,model='objective_void',n=2):
    p=make_density_hyperelastic_problem(n,rho_quad=phi,eta=1e-4,
        periodic_axes=(0,1,2),element_degree=2,material_model=model)
    return p,entry.ExplicitXYZ(p,force_batch_cells=1)

@pytest.mark.parametrize('model',['nh','objective_void'])
def test_periodic_force_is_total_energy_derivative_and_mass_is_unchanged(model):
    phi=np.resize(np.array([.0002,.0055,.2,.9]),(8,27))
    p,ex=make(phi,model)
    rng=np.random.default_rng(34)
    q=jnp.asarray(rng.normal(size=(ex.nc,3))*1e-4).at[ex.pin].set(0.)
    d=jnp.asarray(rng.normal(size=q.shape)*.03).at[ex.pin].set(0.)
    zeros=jnp.zeros_like(q);h=-.07
    energy=lambda x,h:ex.observables((x,zeros,jnp.array(0.)),ex.dt_estimate,lambda t:jnp.array([h,0.,0.]))['energy_N_mm']
    force=ex.reduce(ex.force(q,h))
    np.testing.assert_allclose(jax.jvp(lambda x:energy(x,h),(q,),(d,))[1],jnp.sum(force*d)*ex.L**3,rtol=2e-11,atol=2e-11)
    eps=1e-7
    np.testing.assert_allclose((energy(q+eps*d,h)-energy(q-eps*d,h))/(2*eps),jnp.sum(force*d)*ex.L**3,rtol=2e-6,atol=2e-7)
    row=ex.observe((q,zeros,jnp.array(0.)),ex.dt_estimate,lambda t:jnp.array([h,0.,0.]))
    np.testing.assert_allclose(jax.grad(lambda z:energy(q,z))(h)/ex.L,row['internal_macro_Fz_N'],rtol=2e-12)
    np.testing.assert_allclose(np.sum(ex.nodem),1e-9*ex.L**2*np.sum(np.asarray(p.fe.JxW)*(1e-4+.9999*phi)),rtol=2e-13)

@pytest.mark.parametrize('phi,allowed',[(.00015,True),(.0055,True),(.01,False)])
def test_folds_are_allowed_only_in_continued_virtual_domain(phi,allowed):
    _,ex=make(phi,n=1)
    state=(jnp.zeros((ex.nc,3)),jnp.zeros((ex.nc,3)),jnp.array(0.))
    motion=lambda t:jnp.array([-1.2,0.,0.])
    if allowed:
        row=ex.observe(state,ex.dt_estimate,motion)
        assert row['negative_J_points']==27 and row['required_positive_J_points']==0
        assert row['required_positive_J_min'] is None
        assert np.isfinite(row['energy_N_mm']) and np.isfinite(row['Fz_N'])
    else:
        with pytest.raises(ValueError):ex.observe(state,ex.dt_estimate,motion)

def test_objective_model_cannot_use_legacy_nh_response_and_eta_one():
    p,_=make(.5,n=1)
    with pytest.raises(ValueError,match='Explicit-only'):finite_response(p,[jnp.zeros((len(p.fe.points),3))])
    with pytest.raises(ValueError,match='fixed 0<eta<1'):
        make_density_hyperelastic_problem(1,rho_quad=.5,eta=1.,material_model='objective_void')

def test_short_path_total_jvp_includes_gate_mass_and_affine_inertia():
    phi=jnp.asarray(np.resize(np.array([.0002,.0055,.2,.9]),(8,27)))
    _,ex=make(phi)
    direction=jnp.asarray(np.resize(np.array([.00005,.0002,.002,.001]),(8,27)))
    state=(jnp.zeros((ex.nc,3)),jnp.zeros((ex.nc,3)),jnp.array(0.))
    advance,motion=ex.block(ex.dt_estimate,4,.0004)
    def objective(design):
        scale=1e-4+.9999*(phi+design*direction)
        scale,nodem,mass=ex.material_fields(scale)
        geometry=(*ex.kernel_geometry[:4],scale)
        material=(geometry,nodem,mass)
        final=advance(state,material=material)
        return ex.observables(final,ex.dt_estimate,motion,material)['Fz_N']
    derivative=jax.grad(objective)(0.)
    eps=1e-4;difference=(objective(eps)-objective(-eps))/(2*eps)
    assert np.isfinite(float(derivative)) and abs(float(derivative))>1e-10
    np.testing.assert_allclose(derivative,difference,rtol=2e-5,atol=1e-9)

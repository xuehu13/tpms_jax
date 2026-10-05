"""Known affine physics and conserved HRZ mass under supported Q2 rules."""
import importlib.util
from pathlib import Path
import numpy as np
import jax.numpy as jnp
import pytest
from hyperelastic_fem import make_density_hyperelastic_problem,MU,KAPPA

spec=importlib.util.spec_from_file_location('explicit_entry',Path(__file__).resolve().parents[1]/'scripts/thin_target_explicit.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)

@pytest.mark.parametrize('order,points',[(4,27),(6,64)])
def test_q2_rule_affine_response_and_positive_conserved_mass(order,points):
    eta=.0001;phi=.4;L=10.;density=1e-9;s=.8
    p=make_density_hyperelastic_problem(2,rho_quad=phi,eta=eta,
        periodic_axes=(0,1,2),element_degree=2,quadrature_order=order)
    assert p.fe.num_quads==points
    ex=entry.ExplicitXYZ(p,L=L,density=density,force_batch_cells=1)
    scale=eta+(1-eta)*phi
    np.testing.assert_allclose(np.sum(ex.nodem),density*L**2*scale,rtol=1e-13)
    assert np.min(ex.nodem)>0 and np.min(ex.mass)>0
    # The differentiable material interface must give the same initialized mass.
    _,nodem,mass=ex.material_fields(ex.scale)
    np.testing.assert_allclose(nodem,ex.nodem,rtol=1e-13)
    np.testing.assert_allclose(mass,ex.mass,rtol=1e-13)
    state=(jnp.zeros((ex.nc,3)),jnp.zeros((ex.nc,3)),jnp.array(0.))
    row=ex.observe(state,ex.dt_estimate,lambda t:jnp.array([s-1,0.,0.]))
    P=2*MU/3*(s**(1/3)-s**(-5/3))+KAPPA*(s-1)
    W=MU/2*(s**(-2/3)*(2+s*s)-3)+KAPPA/2*(s-1)**2
    np.testing.assert_allclose(row['internal_macro_Fz_N'],P*scale*L**2,rtol=1e-12)
    np.testing.assert_allclose(row['energy_N_mm'],W*scale*L**3,rtol=1e-12)
    np.testing.assert_allclose(row['J_min'],s,atol=1e-13)

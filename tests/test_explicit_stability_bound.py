"""Conservative bound versus full Jacobian of the actual shared residual."""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from hyperelastic_fem import make_density_hyperelastic_problem
from scripts.thin_target_explicit import ExplicitXYZ,conservative_step_bound

@pytest.mark.parametrize('N,model',[(1,'nh'),(2,'objective_void')])
def test_bound_covers_full_periodic_mass_normalized_spectrum(N,model):
    rho=1. if model=='nh' else np.resize(np.array([.0001,.005,.7]),(N**3,27))
    p=make_density_hyperelastic_problem(N,rho_quad=rho,eta=1e-4,
        periodic_axes=(0,1,2),element_degree=2,quadrature_order=4,material_model=model)
    ex=ExplicitXYZ(p,force_batch_cells=1)
    xyz=np.zeros((ex.nc,3));xyz[np.asarray(ex.ids)]=np.asarray(p.fe.points)%1.
    q=jnp.asarray(.002*(np.cos(2*np.pi*xyz)-1)).at[ex.pin].set(0.)
    h=-.1;before=np.asarray(ex.force(q,h))
    bound=conservative_step_bound(ex,batch_cells=1)(q,h)
    R=float(bound['row_sum_bound_s_minus2'])
    def residual(w):return ex.reduce(ex.force(w,h))
    K=np.asarray(jax.jacfwd(residual)(q)).reshape(ex.nc*3,ex.nc*3)
    keep=np.setdiff1d(np.arange(ex.nc*3),3*ex.pin+np.arange(3))
    K=K[np.ix_(keep,keep)];m=np.repeat(np.asarray(ex.mass),3)[keep]
    A=K/np.sqrt(m[:,None]*m[None,:])
    np.testing.assert_allclose(A,A.T,rtol=1e-10,atol=1e-3)
    eig=np.linalg.eigvalsh((A+A.T)/2)
    assert R>=np.max(np.abs(eig))*(1-1e-11)
    assert R>=np.max(np.sum(np.abs(A),axis=1))*(1-1e-11)
    assert bool(bound['all_material_tangents_finite'])
    assert int(bound['invalid_material_points'])==0
    np.testing.assert_array_equal(ex.force(q,h),before)

def test_bound_retains_inversion_tolerant_void_and_reports_invalid_NH():
    p=make_density_hyperelastic_problem(1,rho_quad=0.,eta=1e-4,
        periodic_axes=(0,1,2),element_degree=2,quadrature_order=4,material_model='objective_void')
    ex=ExplicitXYZ(p,force_batch_cells=1);q=jnp.zeros((ex.nc,3))
    # Macro J=-.2 is allowed only in the declared pure numerical void.
    result=conservative_step_bound(ex,1)(q,-1.2)
    assert float(result['J_min'])<0
    assert bool(result['all_material_tangents_finite'])
    assert int(result['required_positive_J_points'])==0
    assert float(result['row_sum_bound_s_minus2'])>0
    p2=make_density_hyperelastic_problem(1,rho_quad=1.,eta=1e-4,
        periodic_axes=(0,1,2),element_degree=2,quadrature_order=4,material_model='objective_void')
    ex2=ExplicitXYZ(p2,force_batch_cells=1)
    bad=conservative_step_bound(ex2,1)(jnp.zeros((ex2.nc,3)),-1.2)
    assert int(bad['invalid_material_points'])==27
    assert not bool(bad['all_material_tangents_finite'])

"""Actual shared NH block: runtime dt compatibility and controller retention."""
from types import SimpleNamespace
import json
import numpy as np
import pytest
import jax.numpy as jnp
from hyperelastic_fem import make_density_hyperelastic_problem
from scripts.thin_target_explicit import ExplicitXYZ,state_step_target

def setup():
    p=make_density_hyperelastic_problem(1,rho_quad=1.,periodic_axes=(0,1,2),
        element_degree=2,quadrature_order=4,material_model='nh')
    return ExplicitXYZ(p,force_batch_cells=1)

def test_runtime_dt_matches_constant_block_default():
    ex=setup();dt=ex.dt_estimate*.25
    q=jnp.zeros((ex.nc,3)).at[1,0].set(.001)
    state=(q,jnp.zeros_like(q),jnp.asarray(.0015))
    advance,_=ex.block(dt,32,.004)
    constant=advance(state,12);runtime=advance(state,12,step_dt=jnp.asarray(dt))
    for a,b in zip(constant,runtime):np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-14)

@pytest.mark.parametrize('budget',[30.,.001])
def test_actual_controller_completion_or_budget_retains_state(tmp_path,budget):
    ex=setup();a=SimpleNamespace(output=tmp_path,load_time=.004,stability_batch_cells=1,
        control_budget_seconds=budget,checkpoint_compressions=(.1,.2))
    if budget<1:
        with pytest.raises(TimeoutError):state_step_target(ex,a,{},1)
        assert not (tmp_path/'result.json').exists()
        failure=json.loads((tmp_path/'failure.json').read_text())
        assert failure['type']=='TimeoutError'
        with np.load(tmp_path/'last_valid_field.npz') as f:
            assert float(f['time'])==0.;assert np.isfinite(f['q']).all()
    else:
        state_step_target(ex,a,{},1)
        result=json.loads((tmp_path/'result.json').read_text())
        assert result['status']=='controlled_forward_completed_diagnostic'
        with np.load(tmp_path/'field.npz') as f:assert abs(float(f['time'])-.0044)<1e-13
        bounds=json.loads((tmp_path/'stability_path.json').read_text())
        for row,b in zip(result['path'][1:],bounds[1:]):
            assert row['dt_seconds']*np.sqrt(b['R_s_minus2'])<=2
            assert row['invalid_material_points']==0

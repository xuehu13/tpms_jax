"""Check a real NH periodic replay preserves time and centered velocity."""
import json
from types import SimpleNamespace
import numpy as np
import jax.numpy as jnp
from hyperelastic_fem import make_density_hyperelastic_problem
from scripts.thin_target_explicit import ExplicitXYZ,save_state,short_dt_replay

def test_short_replay_preserves_centered_velocity_and_physical_window(tmp_path):
    p=make_density_hyperelastic_problem(2,rho_quad=1.,eta=1e-4,
        periodic_axes=(0,1,2),element_degree=2,quadrature_order=4)
    ex=ExplicitXYZ(p,force_batch_cells=8)
    dt=2e-7;t=jnp.asarray(.0022)
    state=(jnp.zeros((ex.nc,3)),jnp.zeros((ex.nc,3)),t)
    source=tmp_path/'source.npz';save_state(source,state,dt,2)
    output=tmp_path/'out';output.mkdir()
    a=SimpleNamespace(replay_state=source,replay_dt_factor=.125,load_time=.004,output=output)
    short_dt_replay(ex,a,{},2)
    new_dt=dt/8;advance,motion=ex.block(new_dt,128,a.load_time,False)
    h,_,hdd=motion(t);acc=ex.acceleration(state[0],h,hdd)
    expected=(state[0],state[1]+.5*(dt-new_dt)*acc,t)
    with np.load(output/'start_state.npz') as f:
        np.testing.assert_allclose(f['vhalf']+.5*new_dt*acc,state[1]+.5*dt*acc,rtol=1e-13,atol=1e-14)
    for _ in range(8):expected=advance(expected,128)
    with np.load(output/'field.npz') as f:
        assert abs(float(f['time'])-(float(t)+128*dt))<1e-13
        np.testing.assert_allclose(f['q'],expected[0],rtol=1e-12,atol=1e-14)
    result=json.loads((output/'result.json').read_text())
    assert result['status']=='short_same_window_diagnostic_complete'
    assert result['steps']==1024
    assert json.loads((output/'input.json').read_text())['target_compression'] is None

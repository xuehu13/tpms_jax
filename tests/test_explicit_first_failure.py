"""Check physical block replay and retention on a small periodic NH wave."""
import numpy as np
import jax
import jax.numpy as jnp
from hyperelastic_fem import make_density_hyperelastic_problem
from scripts.thin_target_explicit import ExplicitXYZ, diagnose_rejected_block

def setup():
    p=make_density_hyperelastic_problem(2,rho_quad=1.,eta=1e-4,
        periodic_axes=(0,1,2),element_degree=2,quadrature_order=4)
    ex=ExplicitXYZ(p,force_batch_cells=8)
    xyz=np.zeros((ex.nc,3));xyz[np.asarray(ex.ids)]=np.asarray(p.fe.points)%1.
    q=jnp.zeros((ex.nc,3)).at[:,0].set(.001*jnp.sin(2*jnp.pi*jnp.asarray(xyz[:,0])))
    q=q.at[ex.pin].set(0.)
    return ex,q

def test_single_step_replay_matches_original_block():
    ex,q=setup();dt=ex.dt_estimate*.25
    state=(q,-.5*dt*ex.acceleration(q,0.,0.),jnp.asarray(0.))
    advance,motion=ex.block(dt,12,.004,wave=True)
    blocked=advance(state,12);replayed=state
    for _ in range(12):replayed=advance(replayed,1)
    for a,b in zip(blocked,replayed):np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-14)
    assert ex.observe(replayed,dt,motion)['invalid_material_points']==0

def test_first_invalid_internal_step_retains_valid_predecessor(tmp_path):
    ex,q=setup();dt=ex.dt_estimate*40
    state=(q,-.5*dt*ex.acceleration(q,0.,0.),jnp.asarray(0.))
    advance,motion=ex.block(dt,16,.004,wave=True)
    diagnosis=diagnose_rejected_block(ex,advance,motion,state,dt,16,tmp_path/'replay',2)
    assert diagnosis['status']=='first_invalid_internal_step_located'
    assert 1<=diagnosis['internal_step']<=16
    with np.load(tmp_path/'replay/before_first_invalid.npz') as f:
        previous=(jnp.asarray(f['q']),jnp.asarray(f['vhalf']),jnp.asarray(f['time']))
    ex.observe(previous,dt,motion)
    with np.load(tmp_path/'replay/first_invalid.npz') as f:
        bad=(jnp.asarray(f['q']),jnp.asarray(f['vhalf']),jnp.asarray(f['time']))
    import pytest
    with pytest.raises(ValueError):ex.observe(bad,dt,motion)
    assert len(diagnosis['valid_replay_observations'])==diagnosis['internal_step']-1

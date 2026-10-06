"""One objective material candidate on one frozen field; no path solve.

Invoke from an isolated repo containing the candidate kernel with --repo and
--output pointing to a fresh directory. Outputs are never overwritten.
"""
from pathlib import Path
import argparse,hashlib,json,os,sys,time
os.environ.setdefault('JAX_PLATFORMS','cpu')
import basix
import numpy as np
from scipy.special import expit
import jax
import jax.numpy as jnp

parser=argparse.ArgumentParser()
parser.add_argument('--repo',type=Path,default=Path('/home/xuehu/projects/tpms_jax'))
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
R=args.repo.resolve();O=args.output.resolve();O.mkdir(parents=True,exist_ok=True)
assert not (O/'probe_manifest.json').exists()
sys.path.insert(0,str(R))
from jax_fem.basis import get_elements
from hyperelastic_fem import (neo_hookean_energy,first_piola,objective_void_energy,
    objective_void_first_piola,void_nh_weight)
from surface_distance import PeriodicSurfaceDistance
Q=R/'validation/large_compression_20261005_r6/quadratic_candidate'
STATE=Q/'T0p004_compact'
G=R/'validation/thin_target_20261004_r5/gauss_field.npz'
L,T,ETA,N=10.,.5,1e-4,32
ELL=.05/(2*np.log(9))/L
ANGLES=[0.,15.,30.,60.,90.]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def rotation(degrees):
    theta=np.deg2rad(degrees);c,s=np.cos(theta),np.sin(theta)
    return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])

start=time.perf_counter()
inputs=[R/'hyperelastic_fem.py',R/'scripts/thin_target_explicit.py',R/'surface_distance.py',R/'pixi.lock',
        Q/'gauss_field.npz',STATE/'field.npz',STATE/'result.json',G,
        R/'validation/virtual_kernel_20261006_r9/result.json']
before={str(p.relative_to(R)):sha(p) for p in inputs}
write(O/'probe_manifest.json',{'input_sha256':before,'experiment_sha256':sha(Path(__file__)),
    'environment':{'python':sys.version,'jax':jax.__version__,'basix':basix.__version__,'numpy':np.__version__},
    'prespecified_rotation_degrees':ANGLES,'new_equilibrium_or_path':False,'new_Abaqus_jobs':0,
    'new_full_AD_jobs':0,'interpretation':'Material-frame diagnostics, not independent physical cases.'})
with np.load(G) as f:vertices=f['surface_vertices'];triangles=f['surface_triangles']
distance=PeriodicSurfaceDistance(vertices,triangles)
with np.load(STATE/'field.npz') as f:q=f['q'];tsaved=float(f['time'])
baseline=json.loads((STATE/'result.json').read_text());row=baseline['path'][-1]
assert abs(tsaved-row['time'])<1e-12
with np.load(Q/'gauss_field.npz') as f:oldphi=f['rho'];oldqp=f['physical_quad_points'];oldwt=f['JxW']
family,cell,_,_,degree,order=get_elements('HEX27');el=basix.create_element(family,cell,degree)
local=np.rint(2*el.points[order]).astype(int)
Wfun=jax.jit(jax.vmap(objective_void_energy));Pfun=jax.jit(jax.vmap(objective_void_first_piola))
Wold=jax.jit(jax.vmap(neo_hookean_energy));Pold=jax.jit(jax.vmap(first_piola))
weightfun=jax.jit(jax.vmap(void_nh_weight))
rules={}
bins=[('deep_void_le_0p001',-.000001,.001000000000001),('blend_lt_0p01',.001000000000001,.01),
      ('NH_0p01_to_0p05',.01,.05),('transition_0p05_to_0p95',.05,.95),('solid_ge_0p95',.95,1.000001)]
for qorder in [4,8]:
    xi,wtref=basix.make_quadrature(cell,qorder)
    vals=el.tabulate(1,xi).take(order,axis=2)
    shape=vals[0,:,:,0];grad=vals[1:4,:,:,0].transpose(1,2,0)*N
    weights=wtref/N**3
    result={'points_per_cell':len(xi),'Vf':0.,'raw_J_min':1e20,'raw_nonpositive_points':0,
        'raw_nonpositive_phi_max':None,'NH_active_J_min':1e20,'NH_active_nonpositive_points':0,
        'candidate_energy_N_mm':0.,'candidate_internal_Fz_N':0.,'candidate_W_min_MPa':1e20,
        'original_energy_N_mm':0. if qorder==4 else None,'original_internal_Fz_N':0. if qorder==4 else None,
        'domain_bins':{name:{'points':0,'raw_J_min':None,'raw_nonpositive_points':0,
            'reference_volume_mm3':0.,'candidate_energy_N_mm':0.,'candidate_internal_Fz_N':0.} for name,_,_ in bins},
        'rotation':{str(a):{'pure_rotation_energy_N_mm':0.,'superposed_energy_N_mm':0.,
            'stress_covariance_max_abs_MPa':0.,'stress_covariance_max_scaled_error':0.} for a in ANGLES}}
    for begin in range(0,N**3,512):
        indices=np.arange(begin,min(begin+512,N**3))
        origins=np.stack([indices//N**2,(indices//N)%N,indices%N],axis=1)
        full=2*origins[:,None,:]+local[None,:,:];nodes=full/(2*N)
        qp=np.einsum('qn,cnd->cqd',shape,nodes)
        if qorder==4:
            assert np.max(abs(qp-oldqp[indices]))<2e-15
            rho=oldphi[indices];wt=oldwt[indices]
        else:
            d=distance.query(qp);rho=expit((T/(2*L)-d)/ELL)
            wt=np.broadcast_to(weights,rho.shape)
        grid=full%(2*N);ids=(grid[...,0]*(2*N)+grid[...,1])*(2*N)+grid[...,2]
        F=np.eye(3)+np.diag([0.,0.,-row['compression']])+np.einsum('cni,qnj->cqij',q[ids],grad)
        J=np.linalg.det(F);bad=J<=0
        weight=np.asarray(weightfun(jnp.asarray(rho.ravel()))).reshape(rho.shape)
        active=weight>0
        result['Vf']+=float((rho*wt).sum())
        result['raw_J_min']=min(result['raw_J_min'],float(J.min()))
        result['raw_nonpositive_points']+=int(bad.sum())
        if bad.any():
            result['raw_nonpositive_phi_max']=max(result['raw_nonpositive_phi_max'] or 0.,float(rho[bad].max()))
        if active.any():result['NH_active_J_min']=min(result['NH_active_J_min'],float(J[active].min()))
        result['NH_active_nonpositive_points']+=int((bad&active).sum())
        # Reject an undefined active-material domain, rather than silently
        # summing only admissible points. All deep-void points are included.
        assert not (bad&active).any()
        W=np.asarray(Wfun(jnp.asarray(F.reshape((-1,3,3))),jnp.asarray(rho.ravel()))).reshape(rho.shape)
        P=np.asarray(Pfun(jnp.asarray(F.reshape((-1,3,3))),jnp.asarray(rho.ravel()))).reshape((*rho.shape,3,3))
        assert np.isfinite(W).all() and np.isfinite(P).all()
        result['candidate_W_min_MPa']=min(result['candidate_W_min_MPa'],float(W.min()))
        result['candidate_energy_N_mm']+=float((W*wt).sum())*L**3
        result['candidate_internal_Fz_N']+=float((P[:,:,2,2]*wt).sum())*L**2
        for name,lo,hi in bins:
            mask=(rho>=lo)&(rho<hi);b=result['domain_bins'][name]
            b['points']+=int(mask.sum());b['raw_nonpositive_points']+=int((bad&mask).sum())
            if mask.any():b['raw_J_min']=min(b['raw_J_min'] if b['raw_J_min'] is not None else 1e20,float(J[mask].min()))
            b['reference_volume_mm3']+=float(wt[mask].sum())*L**3
            b['candidate_energy_N_mm']+=float((W*wt)[mask].sum())*L**3
            b['candidate_internal_Fz_N']+=float((P[:,:,2,2]*wt)[mask].sum())*L**2
        if qorder==4:
            assert np.min(J)>0
            oldW=np.asarray(Wold(jnp.asarray(F.reshape((-1,3,3))))).reshape(rho.shape)
            oldP=np.asarray(Pold(jnp.asarray(F.reshape((-1,3,3))))).reshape((*rho.shape,3,3))
            factor=(ETA+(1-ETA)*rho)*wt
            result['original_energy_N_mm']+=float((oldW*factor).sum())*L**3
            result['original_internal_Fz_N']+=float((oldP[:,:,2,2]*factor).sum())*L**2
        for angle in ANGLES:
            rot=rotation(angle);r=result['rotation'][str(angle)]
            rigid=np.broadcast_to(rot,F.shape).reshape((-1,3,3))
            Wr=np.asarray(Wfun(jnp.asarray(rigid),jnp.asarray(rho.ravel()))).reshape(rho.shape)
            r['pure_rotation_energy_N_mm']+=float((Wr*wt).sum())*L**3
            rotated=np.einsum('ij,cqjk->cqik',rot,F)
            Wr=np.asarray(Wfun(jnp.asarray(rotated.reshape((-1,3,3))),jnp.asarray(rho.ravel()))).reshape(rho.shape)
            Pr=np.asarray(Pfun(jnp.asarray(rotated.reshape((-1,3,3))),jnp.asarray(rho.ravel()))).reshape(P.shape)
            assert np.isfinite(Wr).all() and np.isfinite(Pr).all()
            r['superposed_energy_N_mm']+=float((Wr*wt).sum())*L**3
            expected=np.einsum('ij,cqjk->cqik',rot,P)
            error=np.max(abs(Pr-expected),axis=(-1,-2))
            r['stress_covariance_max_abs_MPa']=max(r['stress_covariance_max_abs_MPa'],float(error.max()))
            r['stress_covariance_max_scaled_error']=max(r['stress_covariance_max_scaled_error'],
                float(np.max(error/np.maximum(np.max(abs(expected),axis=(-1,-2)),1.))))
    assert sum(b['points'] for b in result['domain_bins'].values())==N**3*len(xi)
    if qorder==4:
        assert abs(result['original_energy_N_mm']/row['energy_N_mm']-1)<1e-11
        assert abs(result['original_internal_Fz_N']/row['internal_macro_Fz_N']-1)<1e-11
        result['same_state_energy_relative_change']=result['candidate_energy_N_mm']/result['original_energy_N_mm']-1
        result['same_state_internal_force_relative_change']=result['candidate_internal_Fz_N']/result['original_internal_Fz_N']-1
    for r in result['rotation'].values():
        r['superposed_relative_energy_change']=r['superposed_energy_N_mm']/result['candidate_energy_N_mm']-1
        assert abs(r['superposed_relative_energy_change'])<1e-10
        assert abs(r['pure_rotation_energy_N_mm'])<1e-10
        assert r['stress_covariance_max_scaled_error']<1e-10
    result['candidate_full_energy_defined']=True
    result['original_full_energy_defined']=result['raw_nonpositive_points']==0
    result['same_state_only']=True;rules[str(qorder)]=result;write(O/f'rule_{qorder}.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['domain_bins','rotation']},indent=2),flush=True)
assert before=={str(p.relative_to(R)):sha(p) for p in inputs}
record={'status':'objective_kernel_frozen_probe_passed','rules':rules,'body_seconds':time.perf_counter()-start,
    'frozen_inputs_unchanged':True,'new_forward_jobs':0,'new_full_path_AD_jobs':0,'new_independent_physical_cases':0,
    'scope':'Full candidate material energy/P on one saved field. Actual virtual J remains inverted. No new equilibrium, no new compression curve and no trajectory gradient.'}
write(O/'result.json',record)
print(json.dumps({k:v for k,v in record.items() if k!='rules'},indent=2),flush=True)

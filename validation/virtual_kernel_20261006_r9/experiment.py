"""One virtual-energy candidate on frozen fields; no new trajectory solve."""
from pathlib import Path
import hashlib, json, os, shutil, sys, time
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np
import basix
from scipy.special import expit
import jax
import jax.numpy as jnp
R=Path('/home/xuehu/projects/tpms_jax');sys.path.insert(0,str(R))
from jax_fem.basis import get_elements
from hyperelastic_fem import (neo_hookean_energy, first_piola, MU, KAPPA,
    interpolated_neo_hookean_energy, interpolated_first_piola, void_interpolation_gamma)
from surface_distance import PeriodicSurfaceDistance
O=R/'validation/virtual_kernel_20261006_r9'
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

def main():
    start=time.perf_counter()
    assert not (O/'result.json').exists()
    inputs=[R/'hyperelastic_fem.py',R/'scripts/thin_target_explicit.py',R/'surface_distance.py',R/'pixi.lock',
            Q/'gauss_field.npz',STATE/'field.npz',STATE/'result.json',G]
    before={str(p.relative_to(R)):sha(p) for p in inputs}
    write(O/'probe_manifest.json',{'input_sha256':before,'experiment_sha256':sha(Path(__file__)),
        'environment':{'python':sys.version,'jax':jax.__version__,'basix':basix.__version__,'numpy':np.__version__},
        'new_equilibrium_or_path':False,'new_Abaqus_jobs':0,'new_full_AD_jobs':0,
        'prespecified_rotation_angles_degrees':ANGLES,
        'interpretation':'Pure rotation/superposed rotation are material-frame diagnostics, not additional physical cases.'})
    shutil.copy2(__file__,O/'experiment.py')
    with np.load(G) as f:vertices=f['surface_vertices'];triangles=f['surface_triangles']
    distance=PeriodicSurfaceDistance(vertices,triangles)
    with np.load(STATE/'field.npz') as f:q=f['q'];tsaved=float(f['time'])
    baseline=json.loads((STATE/'result.json').read_text());row=baseline['path'][-1]
    assert abs(tsaved-row['time'])<1e-12
    with np.load(Q/'gauss_field.npz') as f:oldphi=f['rho'];oldqp=f['physical_quad_points'];oldwt=f['JxW']
    family,cell,_,_,degree,order=get_elements('HEX27');el=basix.create_element(family,cell,degree)
    local=np.rint(2*el.points[order]).astype(int)
    Wfun=jax.jit(jax.vmap(interpolated_neo_hookean_energy))
    Pfun=jax.jit(jax.vmap(interpolated_first_piola))
    Wold=jax.jit(jax.vmap(neo_hookean_energy));Pold=jax.jit(jax.vmap(first_piola))
    gammafun=jax.jit(jax.vmap(void_interpolation_gamma))
    rules={}
    for qorder in [4,8]:
        xi,wtref=basix.make_quadrature(cell,qorder)
        vals=el.tabulate(1,xi).take(order,axis=2)
        shape=vals[0,:,:,0];grad=vals[1:4,:,:,0].transpose(1,2,0)*N
        weights=wtref/N**3
        result={'points_per_cell':len(xi),'Vf':0.,'raw_J_min':1e20,'mapped_J_min':1e20,
                'raw_nonpositive_points':0,'mapped_nonpositive_points':0,
                'raw_nonpositive_phi_max':0.,'candidate_energy_N_mm':0.,'candidate_internal_Fz_N':0.,
                'original_energy_N_mm':0. if qorder==4 else None,
                'original_internal_Fz_N':0. if qorder==4 else None,
                'candidate_energy_by_occupancy':{},'rotation':{str(a):{
                    'pure_rotation_energy_N_mm':0.,'superposed_energy_N_mm':0.,
                    'superposed_mapped_J_min':1e20,'superposed_invalid_points':0} for a in ANGLES}}
        bins=[('deep_void_lt_0p001',0.,.001),('weak_0p001_to_0p01',.001,.01),
              ('weak_0p01_to_0p05',.01,.05),('transition_0p05_to_0p95',.05,.95),('solid_ge_0p95',.95,1.000001)]
        for name,lo,hi in bins:result['candidate_energy_by_occupancy'][name]={'points':0,'reference_volume_mm3':0.,'energy_N_mm':0.}
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
            gamma=np.asarray(gammafun(jnp.asarray(rho.ravel()))).reshape(rho.shape)
            mapped=np.eye(3)+gamma[...,None,None]*(F-np.eye(3));Jm=np.linalg.det(mapped)
            result['Vf']+=float((rho*wt).sum())
            result['raw_J_min']=min(result['raw_J_min'],float(J.min()))
            result['mapped_J_min']=min(result['mapped_J_min'],float(Jm.min()))
            result['raw_nonpositive_points']+=int(bad.sum())
            result['mapped_nonpositive_points']+=int((Jm<=0).sum())
            if bad.any():result['raw_nonpositive_phi_max']=max(result['raw_nonpositive_phi_max'],float(rho[bad].max()))
            # Only mapped-admissible evaluations enter a total candidate sum.
            # Unlike a positive-raw-J subset, the declared extension includes
            # inverted deep-void points where its mapped gradient is valid.
            if np.min(Jm)<=0:
                result['candidate_energy_N_mm']=None;result['candidate_internal_Fz_N']=None
            else:
                W=np.asarray(Wfun(jnp.asarray(F.reshape((-1,3,3))),jnp.asarray(rho.ravel()))).reshape(rho.shape)
                P=np.asarray(Pfun(jnp.asarray(F.reshape((-1,3,3))),jnp.asarray(rho.ravel()))).reshape((*rho.shape,3,3))
                assert np.isfinite(W).all() and np.isfinite(P).all()
                if result['candidate_energy_N_mm'] is not None:
                    result['candidate_energy_N_mm']+=float((W*wt).sum())*L**3
                    result['candidate_internal_Fz_N']+=float((P[:,:,2,2]*wt).sum())*L**2
                for name,lo,hi in bins:
                    mask=(rho>=lo)&(rho<hi);b=result['candidate_energy_by_occupancy'][name]
                    b['points']+=int(mask.sum());b['reference_volume_mm3']+=float(wt[mask].sum())*L**3
                    b['energy_N_mm']+=float((W*wt)[mask].sum())*L**3
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
                assert np.isfinite(Wr).all()
                r['pure_rotation_energy_N_mm']+=float((Wr*wt).sum())*L**3
                rotated=np.einsum('ij,cqjk->cqik',rot,F)
                modeled=np.eye(3)+gamma[...,None,None]*(rotated-np.eye(3))
                Jr=np.linalg.det(modeled);r['superposed_mapped_J_min']=min(r['superposed_mapped_J_min'],float(Jr.min()))
                r['superposed_invalid_points']+=int((Jr<=0).sum())
                if np.min(Jr)<=0:r['superposed_energy_N_mm']=None
                elif r['superposed_energy_N_mm'] is not None:
                    Wr=np.asarray(Wfun(jnp.asarray(rotated.reshape((-1,3,3))),jnp.asarray(rho.ravel()))).reshape(rho.shape)
                    assert np.isfinite(Wr).all()
                    r['superposed_energy_N_mm']+=float((Wr*wt).sum())*L**3
        if qorder==4:
            assert abs(result['original_energy_N_mm']/row['energy_N_mm']-1)<1e-11
            assert abs(result['original_internal_Fz_N']/row['internal_macro_Fz_N']-1)<1e-11
        for angle in ANGLES:
            r=result['rotation'][str(angle)]
            r['pure_rotation_energy_over_frozen_compression_energy']=r['pure_rotation_energy_N_mm']/row['energy_N_mm']
            r['superposed_relative_energy_change']=None if r['superposed_energy_N_mm'] is None or result['candidate_energy_N_mm'] is None else r['superposed_energy_N_mm']/result['candidate_energy_N_mm']-1
        result['candidate_full_energy_defined']=result['mapped_nonpositive_points']==0
        result['original_full_energy_defined']=result['raw_nonpositive_points']==0
        result['same_state_only']=True
        rules[str(qorder)]=result;write(O/f'rule_{qorder}.json',result)
        print(json.dumps({'rule':qorder,'raw_bad':result['raw_nonpositive_points'],
                          'mapped_bad':result['mapped_nonpositive_points'],'mapped_J_min':result['mapped_J_min'],
                          'candidate_energy_N_mm':result['candidate_energy_N_mm'],
                          'rotated_90_relative_energy_change':result['rotation']['90.0']['superposed_relative_energy_change']},indent=2),flush=True)
    assert before=={str(p.relative_to(R)):sha(p) for p in inputs}
    record={'status':'kernel_and_frozen_state_probe_complete','rules':rules,
            'body_seconds':time.perf_counter()-start,'frozen_inputs_unchanged':True,
            'new_forward_jobs':0,'new_full_path_AD_jobs':0,'new_independent_physical_cases':0,
            'scope':'Material domain and finite-rotation diagnostics at one frozen 20% displacement. Candidate energy/internal force is not a new equilibrium or trajectory response. Mapped J is not the actual wall volume ratio.'}
    write(O/'result.json',record)
    print(json.dumps({k:v for k,v in record.items() if k!='rules'},indent=2),flush=True)

if __name__=='__main__':main()

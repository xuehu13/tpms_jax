"""Same-state HEX27 quadrature probe; not a new equilibrium/path solve."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,time
os.environ.setdefault('JAX_PLATFORMS','cpu')
import numpy as np
import basix
from scipy.special import expit
import jax
import jax.numpy as jnp
R=Path('/home/xuehu/projects/tpms_jax');sys.path.insert(0,str(R))
from jax_fem.basis import get_elements
from hyperelastic_fem import neo_hookean_energy,first_piola
from surface_distance import PeriodicSurfaceDistance
E=R/'validation/jax_improvement_20261006_r8';O=E/'integration_probe'
Q=R/'validation/large_compression_20261005_r6/quadratic_candidate'
G=R/'validation/thin_target_20261004_r5/gauss_field.npz'
L,T,ETA,N=10.,.5,1e-4,32
ELL=.05/(2*np.log(9))/L
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def main():
    started=time.perf_counter();O.mkdir(parents=True,exist_ok=False)
    with np.load(G) as f:v=f['surface_vertices'];tri=f['surface_triangles']
    distance_query=PeriodicSurfaceDistance(v,tri)
    with np.load(Q/'gauss_field.npz') as f:oldqp=f['physical_quad_points'];oldphi=f['rho'];oldwt=f['JxW']
    family,cell,_,_,degree,order=get_elements('HEX27');el=basix.create_element(family,cell,degree)
    local=np.rint(2*el.points[order]).astype(int)
    states={}
    for name in ['T0p004_compact','T0p008_compact']:
        record=json.loads((Q/name/'result.json').read_text())
        with np.load(Q/name/'field.npz') as f:q=f['q'];time_saved=float(f['time'])
        assert abs(time_saved-record['path'][-1]['time'])<1e-12
        states[name]=(q,record['path'][-1])
    inputs=[G,Q/'gauss_field.npz',R/'hyperelastic_fem.py',R/'surface_distance.py',R/'scripts/thin_target_explicit.py',R/'pixi.lock']
    inputs += [Q/name/file for name in states for file in ['field.npz','result.json']]
    before={str(p.relative_to(R)):sha(p) for p in inputs}
    manifest={'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),
              'prior_document_edits':subprocess.check_output(['git','diff','--name-only'],cwd=R,text=True).splitlines(),
              'input_sha256':before,'experiment_sha256':sha(Path(__file__)),
              'environment':{'python':sys.version,'numpy':np.__version__,'basix':basix.__version__,'jax':jax.__version__},
              'physics':{'L_mm':L,'t_mm':T,'eta':ETA,'interface_10_90_mm':.05,'periodicity':'XYZ','macro_lateral_strain':0}}
    write(O/'manifest.json',manifest);shutil.copy2(__file__,O/'experiment.py')
    summaries={};Wfun=jax.jit(jax.vmap(neo_hookean_energy));Pfun=jax.jit(jax.vmap(first_piola))
    for qorder in [4,8]:
        xi,weights=basix.make_quadrature(cell,qorder)
        vals=el.tabulate(1,xi).take(order,axis=2)
        shape=vals[0,:,:,0];grad=vals[1:4,:,:,0].transpose(1,2,0)*N
        weights=weights/N**3
        result={'points_per_cell':len(xi),'Vf':0.,'distance_square_occupancy_moment_mm5':0.,'states':{}}
        for name in states:
            result['states'][name]={'J_min':1e20,'nonpositive_points':0,'nonpositive_cells':0,
                'invalid_reference_volume_fraction':0.,'invalid_occupied_volume_mm3':0.,'invalid_phi_max':0.,
                'invalid_phi_min':None,'worst_point':None,'valid_domain_energy_N_mm':0.,'valid_domain_internal_Fz_N':0.,
                'nonpositive_by_phi':{'soft_lt_0p01':0,'interface':0,'solid_ge_0p9':0}}
        for start in range(0,N**3,512):
            k=np.arange(start,min(start+512,N**3));orig=np.stack([k//N**2,(k//N)%N,k%N],axis=1)
            full=2*orig[:,None,:]+local[None,:,:];nodes=full/(2*N)
            qp=np.einsum('qn,cnd->cqd',shape,nodes)
            if qorder==4:
                assert np.max(abs(qp-oldqp[k]))<2e-15
                qp=oldqp[k];rho=oldphi[k];wt=oldwt[k]
                d=distance_query.query(qp)
                assert np.max(abs(expit((T/(2*L)-d)/ELL)-rho))<5e-12
            else:
                d=distance_query.query(qp);rho=expit((T/(2*L)-d)/ELL)
                wt=np.broadcast_to(weights,rho.shape)
            result['Vf']+=float((rho*wt).sum())
            result['distance_square_occupancy_moment_mm5']+=float((rho*wt*d*d).sum())*L**5
            grid=full%(2*N);ids=(grid[...,0]*(2*N)+grid[...,1])*(2*N)+grid[...,2]
            for name,(q,row) in states.items():
                F=np.eye(3)+np.diag([0.,0.,-row['compression']])+np.einsum('cni,qnj->cqij',q[ids],grad)
                J=np.linalg.det(F);bad=J<=0;s=result['states'][name];worst=np.unravel_index(np.argmin(J),J.shape)
                if J[worst]<s['J_min']:
                    s['J_min']=float(J[worst]);s['worst_point']={'cell_index':int(k[worst[0]]),'position_mm':(qp[worst]*L).tolist(),
                        'phi':float(rho[worst]),'distance_mm':float(d[worst]*L)}
                s['nonpositive_points']+=int(bad.sum());s['nonpositive_cells']+=int(np.any(bad,axis=1).sum())
                if bad.any():
                    s['invalid_reference_volume_fraction']+=float(wt[bad].sum())
                    s['invalid_occupied_volume_mm3']+=float((rho*wt)[bad].sum())*L**3
                    s['invalid_phi_max']=max(s['invalid_phi_max'],float(rho[bad].max()))
                    s['invalid_phi_min']=min(s['invalid_phi_min'] or 1.,float(rho[bad].min()))
                    for group,mask in [('soft_lt_0p01',rho<.01),('interface',(rho>=.01)&(rho<.9)),('solid_ge_0p9',rho>=.9)]:
                        s['nonpositive_by_phi'][group]+=int((bad&mask).sum())
                # Never evaluate the material at bad points. Partial sums are
                # explicitly diagnostic only and cannot stand for total energy.
                good=~bad;flat=jnp.asarray(F[good])
                wg=np.asarray(Wfun(flat));pg=np.asarray(Pfun(flat))[:,2,2]
                factor=((ETA+(1-ETA)*rho)*wt)[good]
                s['valid_domain_energy_N_mm']+=float(np.dot(wg,factor))*L**3
                s['valid_domain_internal_Fz_N']+=float(np.dot(pg,factor))*L**2
        for name,(q,row) in states.items():
            s=result['states'][name];valid=s['nonpositive_points']==0
            s['total_material_response_defined']=valid
            s['total_energy_N_mm']=s['valid_domain_energy_N_mm'] if valid else None
            s['total_internal_Fz_N']=s['valid_domain_internal_Fz_N'] if valid else None
            if qorder==4:
                assert abs(s['total_energy_N_mm']/row['energy_N_mm']-1)<1e-11
                assert abs(s['total_internal_Fz_N']/row['internal_macro_Fz_N']-1)<1e-11
                assert abs(s['J_min']-row['J_min'])<1e-12
            elif valid:
                s['relative_energy_change_from_original_rule']=s['total_energy_N_mm']/row['energy_N_mm']-1
                s['relative_internal_force_change_from_original_rule']=s['total_internal_Fz_N']/row['internal_macro_Fz_N']-1
        summaries[str(qorder)]=result
        write(O/f'rule_{qorder}.json',result)
        print(json.dumps({'quadrature_order':qorder,**result},indent=2),flush=True)
    assert before=={str(p.relative_to(R)):sha(p) for p in inputs}
    result={'status':'same_state_probe_complete','rules':summaries,'body_seconds':time.perf_counter()-started,
            'frozen_inputs_unchanged':True,'new_forward_solve':False,'gradient_validation':False,
            'scope':'Quadrature sampling of frozen HEX27 states. Partial positive-J sums are not a physical total. No extrapolation to a reequilibrated higher-quadrature response.'}
    write(O/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k!='rules'},indent=2),flush=True)

if __name__=='__main__':main()

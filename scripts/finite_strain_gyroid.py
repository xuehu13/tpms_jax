"""Historical middle-thickness Gyroid continuation, not the new thin target.

The default c=.541062 is not physical shell thickness. This entry point preserves
the completed preflight definition; new target inputs require a separate case.
"""
from pathlib import Path
import argparse,hashlib,json,resource,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def capture(n,increment,limit,out,eta=1e-4):
    import jax
    import jax.numpy as jnp
    from petsc4py import PETSc
    from fem import solve,hex_jacobian_check
    from hyperelastic_fem import make_density_hyperelastic_problem,reduced_guess,finite_response,neo_hookean_energy
    if out.exists() or out.with_suffix('.progress.json').exists():raise FileExistsError(out)
    if n not in (16,32,48,64) or increment not in (.005,.0025) or limit not in (.05,.10,.20):raise ValueError('Outside locked case domain')
    start=time.perf_counter();p=make_density_hyperelastic_problem(n,eta=eta);built=time.perf_counter()
    minmap,V0=hex_jacobian_check(p);assert minmap>0 and abs(V0-1)<1e-10
    for key,val in {'ksp_rtol':1e-11,'ksp_atol':1e-13,'ksp_max_it':3000,'ksp_error_if_not_converged':True}.items():PETSc.Options()[key]=val
    points=np.asarray(p.fe.points);weights=np.asarray(p.JxW)[:,0,:];rho=np.asarray(p.rho)
    sol=[jnp.zeros_like(jnp.asarray(points))];rows=[];fields=[];started_a=0.;failure=None
    old=ROOT/'validation/mechanics_trust_20261003_r4/step1/audit.json'
    anchor=next(r for r in json.loads(old.read_text())['comparisons'] if r['label']=='baseline' and r['lateral']=='fixed')
    linearK=anchor['K_background'] if n==64 else None
    def write(f,v):f.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
    def save_fields():
        np.savez_compressed(out.with_suffix('.displacement.npz'),points=points,compression=[r['compression'] for r in rows],u=np.asarray(fields),N=n)
        np.savez_compressed(out.with_suffix('.last_state.npz'),points=points,w=np.asarray(sol[0]),H=np.asarray(p.H_macro),rho=rho,N=n)
    def constraints(w):
        maximum=0.
        for axis in (0,1):
            other=[d for d in range(3) if d!=axis]
            low=np.flatnonzero(points[:,axis]==0);high=np.flatnonzero(points[:,axis]==1)
            low=low[np.lexsort((points[low,other[1]],points[low,other[0]]))]
            high=high[np.lexsort((points[high,other[1]],points[high,other[0]]))]
            maximum=max(maximum,float(np.abs(w[high]-w[low]).max()))
        loaded=(points[:,2]==0)|(points[:,2]==1)
        return maximum,float(np.abs(w[loaded,2]).max())
    out.parent.mkdir(parents=True,exist_ok=True)
    try:
        for a in np.r_[0.,.0001,np.arange(1,round(limit/increment)+1)*increment]:
            started_a=float(a);p.set_params(np.diag([0.,0.,-a]),p.rho,eta);p.trial_calls=0
            write(out.with_suffix('.current.json'),{'compression':float(a),'state_start_monotonic':time.monotonic()})
            print('START_STATE '+str(a),flush=True);t=time.perf_counter()
            if a:
                sol=solve(p,{'newton':{'tol':1e-10,'rel_tol':1e-10,'initial_guess':reduced_guess(p,sol[0]),
                                     'linear':{'petsc_solver':{'ksp_type':'gmres','pc_type':'gamg'}}}})
            row=finite_response(p,sol);stats=p.detF_stats(sol[0]);w=np.asarray(sol[0]);u=w+points@np.asarray(p.H_macro).T
            F=jnp.eye(3)+p.H_macro+p.fe.sol_to_grad(sol[0]);W=np.asarray(jax.vmap(neo_hookean_energy)(F.reshape((-1,3,3))).reshape(rho.shape))
            material_energy=float(np.sum((1-eta)*rho*W*weights));floor_energy=float(np.sum(eta*W*weights))
            pure_void_energy=float(np.sum((eta+(1-eta)*rho)*W*weights*(rho<=.05)))
            per,loaded=constraints(w);energy=row['energy'];voidfrac=pure_void_energy/energy if energy>1e-14 else 0.
            row.update(compression=float(a),solve_seconds=time.perf_counter()-t,Newton_corrections=max(p.trial_calls-1,0),
                detF_by_occupancy=stats,energy_projected_material=material_energy,energy_uniform_floor=floor_energy,
                energy_pure_void_rho_le005=pure_void_energy,pure_void_energy_fraction=voidfrac,
                periodic_error=per,loaded_face_error=loaded,host_RSS_peak_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
            if a==.0001 and linearK is not None:row['small_strain_relative_difference']=abs(-row['Fz_top']/a-linearK)/linearK
            row['checks']={'finite':all(np.isfinite(v) for v in (row['energy'],row['Fz_top'],row['J_min'],row['reduced_residual_l2'])),
                'positive_detF':row['J_min']>0,'soft_domain_not_severely_collapsed':stats['void']['min'] is None or stats['void']['min']>=.1,
                'reduced_equilibrium':row['reduced_residual_l2']<=1e-8,'balance':abs(row['Fz_top']+row['Fz_bottom'])<=1e-8,
                'macro_reaction_vs_Piola':abs(row['Fz_top']-row['mean_first_piola'][2][2])<=1e-8,
                'energy_partition':abs(material_energy+floor_energy-energy)<=1e-10*max(1.,energy),
                'periodic':per<=1e-10,'flat_axial_faces':loaded<=1e-10,
                'small_strain_limit':row.get('small_strain_relative_difference',0.)<=.01}
            rows.append(row);fields.append(u);write(out.with_suffix('.progress.json'),rows);save_fields()
            print('COMPLETE_STATE '+json.dumps({'a':float(a),'Fz':row['Fz_top'],'energy':energy,'minJ':row['J_min'],'void_energy_fraction':voidfrac,'seconds':row['solve_seconds']}),flush=True)
            if not all(row['checks'].values()):raise ValueError('Physical/state criteria failed: '+str([k for k,v in row['checks'].items() if not v]))
    except Exception as exc:
        failure={'type':type(exc).__name__,'message':str(exc),'attempted_compression':started_a,
                 'last_completed_compression':rows[-1]['compression'] if rows else None,'trial_detF':getattr(p,'trial_detF',None)}
        if hasattr(p,'invalid_trial_w'):
            trial=out.with_suffix('.invalid_trial.npz');np.savez_compressed(trial,points=points,w=p.invalid_trial_w,H=np.asarray(p.H_macro),rho=rho,N=n)
            failure['invalid_trial_file']=str(trial)
        write(out.with_suffix('.failure.json'),failure);save_fields()
    a=np.array([r['compression'] for r in rows]);force=np.array([r['Fz_top'] for r in rows]);stored=rows[-1]['energy'] if rows else 0.
    work=float(np.sum(-.5*(force[1:]+force[:-1])*np.diff(a)));gap=abs(work-stored)/stored if stored>1e-14 else 0.
    result={'status':'stopped' if failure else 'ok','N':n,'increment':increment,'requested_limit':limit,'eta':eta,'rows':rows,'failure':failure,
        'geometry':{'c':.541062,'beta':40.,'vf_projected':float(np.sum(rho*weights)),'map':'actual reference Gauss coordinates'},
        'mesh':{'cells':n**3,'Gauss_points':8*n**3,'reduced_dofs':p.P_mat.shape[1],'reference_volume':V0,'min_map_det':minmap},
        'work':{'path_trapezoid':work,'stored_energy':stored,'relative_gap':gap,'uses_half_Fu':False,'checked_interval_end':float(a[-1]) if a.size else None},
        'runtime':{'build_seconds':built-start,'path_seconds':time.perf_counter()-built,'wall_seconds':time.perf_counter()-start,
                   'peak_host_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'devices':[str(d) for d in jax.devices()]},
        'solver':'installed Newton + existing PETSc GMRES/GAMG; unmodified FEM/PBC kernels',
        'precision_claim':False if n<48 else 'requires independent comparison',
        'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('hyperelastic_fem.py','scripts/finite_strain_gyroid.py','fem.py','pbc.py','geometry.py')}}
    write(out,result);print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)
    return not failure


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--N',type=int,required=True)
    p.add_argument('--increment',type=float,default=.005);p.add_argument('--limit',type=float,default=.05)
    p.add_argument('--eta',type=float,default=1e-4);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    raise SystemExit(0 if capture(a.N,a.increment,a.limit,a.out,a.eta) else 2)

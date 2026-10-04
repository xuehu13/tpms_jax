"""Bounded full-solid finite-strain benchmark: capture JAX paths or prepare INP."""
from pathlib import Path
import argparse,hashlib,json,resource,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def compressions(increment):
    if increment not in (.01,.005):
        raise ValueError('Locked benchmark increments are .01 or .005')
    return np.r_[0.,.0001,np.arange(1,round(.2/increment)+1)*increment]


def analytic(compression):
    mu=10/2.6;kappa=10/(3*.4);s=1-compression
    if not 0<s<=1:raise ValueError('Compression outside the benchmark range')
    pz=2*mu/3*(s**(1/3)-s**(-5/3))+kappa*(s-1)
    px=mu*s**(-2/3)*(1-(2+s*s)/3)+kappa*(s-1)*s
    W=mu/2*(s**(-2/3)*(2+s*s)-3)+kappa/2*(s-1)**2
    return {'compression':float(compression),'Fz':float(pz),'energy':float(W),'J':float(s),
            'first_piola_diagonal':[float(px),float(px),float(pz)],
            'cauchy_diagonal':[float(px/s),float(px/s),float(pz)]}


def capture(n,increment,out):
    import jax
    import jax.numpy as jnp
    from fem import solve,hex_jacobian_check
    from hyperelastic_fem import make_hyperelastic_problem,reduced_guess,finite_response
    if out.exists():raise FileExistsError(out)
    start=time.perf_counter();problem=make_hyperelastic_problem(n);built=time.perf_counter()
    min_map,V0=hex_jacobian_check(problem)
    assert min_map>0 and abs(V0-1)<1e-12
    points=np.asarray(problem.fe.points);sol=[jnp.zeros_like(jnp.asarray(points))]
    rows=[];fields=[];fluctuations=[]
    out.parent.mkdir(parents=True,exist_ok=True)
    for compression in compressions(increment):
        H=np.diag([0.,0.,-compression]);problem.set_params(H)
        initial=np.asarray(sol[0]).copy()
        seeded=compression==0 or np.isclose(compression,.2,atol=1e-14,rtol=0)
        if seeded:
            wave=np.sin(np.pi*points[:,2])
            initial[:,0]+=.003*(np.cos(2*np.pi*points[:,0])-1)*wave
            initial[:,1]+=.002*(np.cos(2*np.pi*points[:,1])-1)*wave
            initial[:,2]+=.002*(np.cos(2*np.pi*points[:,0])+np.cos(2*np.pi*points[:,1])-2)*wave
        q=reduced_guess(problem,initial)
        compatible=[jnp.asarray((problem.P_mat@np.asarray(q)).reshape((-1,3)))]
        initial_residual=finite_response(problem,compatible)['reduced_residual_l2']
        t=time.perf_counter()
        sol=solve(problem,{'newton':{'tol':1e-11,'rel_tol':1e-11,'initial_guess':q,'linear':{'spsolve_solver':{}}}})
        row=finite_response(problem,sol);expected=analytic(compression)
        w=np.asarray(sol[0]);u=w+points@H.T
        pair_error=0.
        for axis in (0,1):
            other=[d for d in range(3) if d!=axis]
            low=np.flatnonzero(points[:,axis]==0);high=np.flatnonzero(points[:,axis]==1)
            low=low[np.lexsort((points[low,other[1]],points[low,other[0]]))]
            high=high[np.lexsort((points[high,other[1]],points[high,other[0]]))]
            assert np.allclose(points[high][:,other],points[low][:,other],atol=1e-12)
            pair_error=max(pair_error,float(np.abs(w[high]-w[low]).max()))
        loaded=np.isclose(points[:,2],0)|np.isclose(points[:,2],1)
        row.update(compression=float(compression),solve_seconds=time.perf_counter()-t,
                   initial_residual_l2=initial_residual,nonzero_seed=bool(seeded),
                   periodic_error=pair_error,loaded_face_error=float(np.abs(w[loaded,2]).max()),
                   max_affine_displacement_error=float(np.abs(w).max()),analytic=expected)
        scale=max(1.,abs(expected['Fz']))
        row['checks']={'finite':all(np.isfinite(v) for v in (row['energy'],row['Fz_top'],row['J_min'],row['reduced_residual_l2'])),
            'positive_detF':row['J_min']>0,'reduced_equilibrium':row['reduced_residual_l2']<=1e-9,
            'top_bottom_balance':abs(row['Fz_top']+row['Fz_bottom'])<=1e-9*scale,
            'reaction_vs_first_piola':abs(row['Fz_top']-row['mean_first_piola'][2][2])<=1e-9*scale,
            'periodic':pair_error<=1e-10,'loaded_faces':row['loaded_face_error']<=1e-10,
            'affine_solution':row['max_affine_displacement_error']<=1e-9,
            'analytic_force':abs(row['Fz_top']-expected['Fz'])<=1e-7*scale,
            'analytic_energy':abs(row['energy']-expected['energy'])<=1e-10,
            'analytic_detF':max(abs(row['J_min']-expected['J']),abs(row['J_max']-expected['J']))<=1e-9,
            'nonzero_seed_exercised':not seeded or initial_residual>1e-4}
        rows.append(row);fields.append(u);fluctuations.append(w)
        out.with_suffix('.progress.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
        if not all(row['checks'].values()):raise ValueError('Finite-strain checks failed; progress retained')
    a=np.array([r['compression'] for r in rows]);force=np.array([r['Fz_top'] for r in rows])
    work=np.trapezoid(-force,a);energy=rows[-1]['energy']
    fields_path=out.with_suffix('.displacement.npz')
    np.savez_compressed(fields_path,points=points,compression=a,u=np.array(fields),w=np.array(fluctuations),N=n)
    result={'N':n,'increment':increment,'status':'ok','rows':rows,
            'material':{'E0':10.,'nu0':.3,'mu':10/2.6,'kappa':10/(3*.4),'energy_reference':'isochoric Neo-Hookean plus quadratic (J-1) bulk'},
            'mesh':{'cells':n**3,'Gauss_points':8*n**3,'reduced_dofs':problem.P_mat.shape[1],'reference_min_map_det':min_map,'reference_volume':V0},
            'work':{'path_trapezoid':float(work),'stored_energy':energy,'relative_gap':float(abs(work-energy)/energy),'uses_half_Fu':False},
            'runtime':{'build_seconds':built-start,'path_seconds':time.perf_counter()-built,'wall_seconds':time.perf_counter()-start,
                       'peak_host_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'jax':jax.__version__,'devices':[str(d) for d in jax.devices()]},
            'displacement':{'path':str(fields_path),'sha256':hashlib.sha256(fields_path.read_bytes()).hexdigest()},
            'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('hyperelastic_fem.py','fem.py','pbc.py','scripts/finite_strain_uniform.py')}}
    with out.open('x') as stream:stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    if result['work']['relative_gap']>.001:raise ValueError('Nonlinear work check failed')
    return result


def prepare(n,increment,out):
    from prepare_uniform_baseline import build_model,validate_model,render
    out.mkdir(parents=True,exist_ok=False)
    model=build_model(n,'fixed');audit=validate_model(n,model)
    nodes,cells,controls,equations,bcs=model
    prefix=render(n,'fixed',model).split('*STEP,')[0]
    prefix=prefix.replace('** E=10, nu=0.3, eps_z=-0.01, pbc macro DOFs at QX/QY/QZ',
                          '** Full solid finite strain; reference energy and matched Neo-Hookean constants')
    mu=10/2.6;kappa=10/(3*.4)
    assert '*ELASTIC\n10., 0.3' in prefix
    prefix=prefix.replace('*ELASTIC\n10., 0.3',f'*HYPERELASTIC, NEO HOOKE\n{mu/2:.17g}, {2/kappa:.17g}')
    lines=[prefix.rstrip(),'** Initial constraints; only QZ changes during loading','*BOUNDARY']
    lines += [f'{label}, {comp}, {comp}, 0.' for label,comp,_ in bcs]
    steps=[]
    for i,a in enumerate(compressions(increment)[1:],1):
        name=f'COMP_{i:03d}';steps.append({'name':name,'compression':float(a),'analytic':analytic(a)})
        lines += [f'*STEP, NAME={name}, NLGEOM=YES, INC=100','*STATIC, DIRECT','1., 1.','*BOUNDARY',f'{controls[2]}, 3, 3, {-a:.17g}',
                  '*OUTPUT, FIELD, FREQUENCY=1','*NODE OUTPUT','U, RF','*ELEMENT OUTPUT, ELSET=SOLID','S, LE, IVOL, SENER',
                  '*OUTPUT, HISTORY, FREQUENCY=1','*ENERGY OUTPUT','ALLSE, ALLWK, ALLAE',
                  '*NODE OUTPUT, NSET=QX','U1, RF1','*NODE OUTPUT, NSET=QY','U2, RF2','*NODE OUTPUT, NSET=QZ','U3, RF3','*END STEP']
    tag=f'uniform_finite_N{n}_d{round(increment*1000):03d}'
    inp=out/(tag+'.inp');inp.write_text('\n'.join(lines)+'\n',encoding='ascii')
    expected={'case':tag,'N':n,'increment':increment,'material':{'E0':10.,'nu0':.3,'C10':mu/2,'D1':2/kappa},
              'controls':list(controls),'nodes':nodes,'steps':steps,'offline_checks':audit,'reference_volume':1.,
              'displacement_tolerance':1e-7,'relative_physics_tolerance':1e-6,
              'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('scripts/finite_strain_uniform.py','scripts/prepare_uniform_baseline.py')},
              'input_sha256':hashlib.sha256(inp.read_bytes()).hexdigest()}
    (out/(tag+'.expected.json')).write_text(json.dumps(expected,indent=2)+'\n')
    print(json.dumps({k:v for k,v in expected.items() if k not in ('nodes','steps')},indent=2))
    return expected


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('capture','prepare'))
    parser.add_argument('--N',type=int,required=True);parser.add_argument('--increment',type=float,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    (capture if args.action=='capture' else prepare)(args.N,args.increment,args.out)

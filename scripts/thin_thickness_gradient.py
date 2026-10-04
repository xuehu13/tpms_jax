"""Step 4: thickness equilibrium-force gradients at accepted 1%/5% compression.

Reuse the saved forward state and installed tangent/linear solver. JAX computes
material and residual partials; one transpose solve supplies the implicit term.
No differentiation of the Newton iterations, shape map, contact or training.
"""
from pathlib import Path
import argparse,hashlib,json,os,resource,shutil,sys,time
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import jax
import jax.numpy as jnp
from fem import solve
from hyperelastic_fem import make_density_hyperelastic_problem,reduced_guess,finite_response,first_piola
from surface_distance import thickness_occupancy
from jax_fem.solver import get_A,linear_solver


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):
    with Path(p).open('x') as f:f.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
def tag(t):return 't'+format(t,'.6f').replace('.','p')


def force_partial(problem,distance,t_mm,L,ell,w):
    """Positive axial reaction Q=-L^2 mean(Pzz), differentiable at fixed w.

    At XYZ periodic equilibrium this equals the end-face reaction amplitude.
    Gradients use physical millimetres for thickness and a fixed interface ell.
    """
    rho=thickness_occupancy(distance,t_mm/L,ell)
    F=jnp.eye(3)+problem.H_macro+problem.fe.sol_to_grad(w)
    P=jax.vmap(first_piola)(F.reshape((-1,3,3))).reshape(F.shape)
    weights=jnp.asarray(problem.JxW)[:,0,:]
    return -L**2*jnp.sum((problem.eta+(1-problem.eta)*rho)*P[...,2,2]*weights)/weights.sum()


def equilibrium_gradient(problem,distance,t_mm,L,ell,a,w,linear_options):
    """Q_t - lambda^T R_t, where K^T lambda=Q_w in admissible XYZ DOFs.

    R=U_w and Q=L^2 U_a/V0 for this conservative displacement-controlled
    energy. Therefore Q_w=L^2 R_a/V0, computed by JAX JVP. This avoids a
    retained full reverse tape, not an approximation of the equilibrium rule.
    The reduced matrix is transposed in place only for this isolated adjoint;
    a later solve rebuilds it through the installed get_A interface.
    """
    t_mm=jnp.asarray(t_mm);a=jnp.asarray(a);w=jnp.asarray(w)
    H=jnp.diag(jnp.array([0.,0.,-a]));eta=problem.eta
    def residual(t,compression):
        problem._set_params_jax(jnp.diag(jnp.array([0.,0.,-compression])),
                               thickness_occupancy(distance,t/L,ell),eta)
        return problem.compute_residual([w])[0]
    start=time.perf_counter()
    try:
        raw,Rt=jax.jvp(lambda t:residual(t,a),(t_mm,),(jnp.ones_like(t_mm),))
        raw_host=np.asarray(raw);Rt_host=np.asarray(Rt)
        _,Ra=jax.jvp(lambda x:residual(t_mm,x),(a,),(jnp.ones_like(a),))
        Ra_host=np.asarray(Ra)
    finally:
        problem.set_params(H,thickness_occupancy(distance,t_mm/L,ell),eta)
    q,direct=jax.jvp(lambda t:force_partial(problem,distance,t,L,ell,w),(t_mm,),(jnp.ones_like(t_mm),))
    q=float(q);direct=float(direct)
    volume=float(jnp.asarray(problem.JxW)[:,0,:].sum())
    qw=Ra_host*(L**2/volume)
    partial_seconds=time.perf_counter()-start
    print('PARTIALS_DONE '+str(partial_seconds),flush=True)
    # Only host vectors are needed for the transpose solve; discard JVP arrays.
    del raw,Rt,Ra
    jax.clear_caches()
    problem.trial_calls=0;start=time.perf_counter()
    problem.newton_update([w]);A=get_A(problem)
    if hasattr(problem,'V'):del problem.V  # installed PETSc cache consumed COO values
    A.transpose()
    b=np.asarray(problem.P_mat.T@qw.ravel())
    assembled=time.perf_counter();print('ADJOINT_MATRIX_DONE '+str(assembled-start),flush=True)
    x=np.asarray(linear_solver(A,b,None,linear_options))
    vec=A.createVecRight();vec.setArray(x);out=A.createVecLeft();A.mult(vec,out)
    error=np.linalg.norm(out.getArray()-b);relative=error/max(np.linalg.norm(b),1e-30)
    full=np.asarray(problem.P_mat@x)
    correction=-float(np.dot(full,Rt_host.ravel()))
    result={'reaction_amplitude_N':q,'fixed_state_partial_N_per_mm':direct,
            'equilibrium_correction_N_per_mm':correction,'total_gradient_N_per_mm':direct+correction,
            'adjoint_residual_l2':float(error),'adjoint_relative_residual':float(relative),
            'equilibrium_reduced_residual_l2':float(np.linalg.norm(problem.P_mat.T@raw_host.ravel())),
            'partials_seconds':partial_seconds,'tangent_seconds':assembled-start,
            'adjoint_linear_seconds':time.perf_counter()-assembled,'reduced_dofs':len(b),
            'tangent_nonzeros':int(A.getInfo()['nz_used'])}
    vec.destroy();out.destroy();A.destroy()
    return result


def two_point_loss(forces,reference_forces):
    """Dimensionless diagnostic objective; fixed targets, no inverse optimization."""
    return .5*jnp.sum((jnp.asarray(forces)/jnp.asarray(reference_forces)-.98)**2)


def prepare(case,a=.01):
    assert a in (.01,.05)
    atag='a'+format(a,'.6f').replace('.','p')
    out=case/f'step4_thickness_a{int(round(a*100)):02d}';out.mkdir(exist_ok=False)
    old=case/'step3_xyz'/atag;cfg=json.loads((old/'input.json').read_text())
    assert json.loads((old/'comparison.json').read_text())['force_energy_screen_pass']
    cfg.pop('diagnostic_xyz',None)
    cfg.update(step=4,epsilon_z=-a,compression=a,thickness_mm=.5,
        boundary_role='user-selected XYZ compression',
        gradient_parameter='physical thickness in mm; fixed reference midsurface/distance/mesh and physical interface width',
        source_sha256={n:sha(ROOT/n) for n in ('scripts/thin_thickness_gradient.py','scripts/extract_thin_finite.py',
            'hyperelastic_fem.py','surface_distance.py','fem.py','pbc.py','pixi.lock')},
        input_sha256={n:sha(case/n) for n in ('input.json','gauss_field.npz','step1_decision.json',
            f'step3_xyz/{atag}/input.json',f'step3_xyz/{atag}/jax.json',f'step3_xyz/{atag}/jax_field.npz')},
        finite_difference_steps_mm=[.002,.001] if a==.01 else [.001],shell_difference_step_mm=.002 if a==.01 else None,
        acceptance={'numerical_fine_difference_relative_error_le':.01,'central_difference_step_consistency_le':.01,
                    'adjoint_relative_residual_le':1e-6,'shell_test':'same force-change direction; report magnitude difference without claiming shape/curve certification'},
        no_training=True,no_shape_derivative=True)
    cfg['mode_comparison']='Thickness response and equilibrium total derivative only'
    for name in cfg['source_sha256']:
        p=out/'source_at_run'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,p)
    write(out/'input.json',cfg)
    (out/'adjoint').mkdir()
    original=(old/'thin_shell.inp').read_text();marker='*Shell Section, elset=TPMS, material=MAT, section integration=SIMPSON\n0.5, 5'
    if marker not in original:
        # Preserve the actual source keyword formatting; only change its thickness line.
        rows=original.splitlines();indices=[i for i,row in enumerate(rows) if row.lower().startswith('*shell section')]
        assert len(indices)==1;index=indices[0]+1;assert float(rows[index].split(',')[0])==.5
    else:index=original.splitlines().index(marker.splitlines()[0])+1
    for t in sorted({.5+direction*h for h in cfg['finite_difference_steps_mm'] for direction in (-1,1)}):
        folder=out/tag(t);folder.mkdir();p_cfg=dict(cfg,thickness_mm=t)
        if abs(t-.5)>.0015:
            rows=original.splitlines();parts=rows[index].split(',');parts[0]=format(t,'.16g');rows[index]=','.join(parts)
            (folder/'thin_shell.inp').write_text('\n'.join(rows)+'\n')
            p_cfg['shell_inp_sha256']=sha(folder/'thin_shell.inp')
        else:p_cfg.pop('shell_inp_sha256',None)
        write(folder/'input.json',p_cfg)
    print(json.dumps({'out':str(out),'finite_difference_steps_mm':cfg['finite_difference_steps_mm'],'shell_thicknesses_mm':[.498,.502] if a==.01 else []}))


def run(case,t_mm,adjoint,a=.01):
    start=time.perf_counter();atag='a'+format(a,'.6f').replace('.','p');root=case/f'step4_thickness_a{int(round(a*100)):02d}';cfg=json.loads((root/'input.json').read_text())
    out=root/('adjoint' if adjoint else tag(t_mm));assert not (out/'jax.json').exists()
    for name,h in cfg['source_sha256'].items():assert sha(ROOT/name)==h,name
    for name,h in cfg['input_sha256'].items():assert sha(case/name)==h,name
    L=cfg['cell_size_mm'];ell=cfg['interface_10_90_mm']/L/(2*np.log(9));a=cfg['compression']
    with np.load(case/'gauss_field.npz') as c:
        distance=jnp.asarray(c['distance']);qp=c['physical_quad_points'];qw=c['JxW'];rho0=c['rho']
    rho=thickness_occupancy(distance,t_mm/L,ell)
    if adjoint:np.testing.assert_array_equal(np.asarray(rho),rho0)
    def field(p):
        assert np.array_equal(np.asarray(p.physical_quad_points),qp) and np.array_equal(np.asarray(p.JxW)[:,0,:],qw)
        return rho
    p=make_density_hyperelastic_problem(cfg['N'],rho_quad=field,eta=cfg['eta'],periodic_axes=(0,1,2));del qp,qw,rho0
    H=np.diag([0.,0.,-a]);p.set_params(H,rho,cfg['eta']);p.trial_calls=0
    with np.load(case/f'step3_xyz/{atag}/jax_field.npz') as f:w=jnp.asarray(f['w']);np.testing.assert_array_equal(f['H'],H)
    from petsc4py import PETSc
    options={'ksp_rtol':1e-8,'ksp_atol':1e-12,'ksp_max_it':3000,'ksp_error_if_not_converged':True,
             'ksp_gmres_modifiedgramschmidt':True,'ksp_pc_side':'right','ksp_norm_type':'unpreconditioned'}
    for k,v in options.items():PETSc.Options()[k]=v
    built=time.perf_counter();print('BUILD_DONE '+str(built-start),flush=True)
    try:
        if adjoint:
            result=equilibrium_gradient(p,distance,t_mm,L,ell,a,w,{'petsc_solver':{'ksp_type':'gmres','pc_type':'gamg'}})
            checks={'finite':bool(all(np.isfinite(v) for v in result.values())),
                    'base_equilibrium_residual_le_1e-8':result['equilibrium_reduced_residual_l2']<=1e-8,
                    'adjoint_relative_residual_le_1e-6':result['adjoint_relative_residual']<=1e-6}
            old=json.loads((case/f'step3_xyz/{atag}/jax.json').read_text())
            checks['same_base_reaction_le_1e-6']=abs(result['reaction_amplitude_N']/(-old['Fz_N'])-1)<=1e-6
        else:
            sol=solve(p,{'newton':{'tol':1e-10,'rel_tol':1e-10,'initial_guess':reduced_guess(p,w),
                                 'linear':{'petsc_solver':{'ksp_type':'gmres','pc_type':'gamg'}}}})
            jax.block_until_ready(sol);solved=time.perf_counter();r=finite_response(p,sol);stats=p.detF_stats(sol[0])
            points=np.asarray(p.fe.points);idx=np.rint(points*cfg['N']).astype(int)
            grid=np.empty((cfg['N']+1,)*3+(3,));grid[idx[:,0],idx[:,1],idx[:,2]]=np.asarray(sol[0])+points@H.T
            per=max(float(np.max(np.abs(np.take(grid,-1,axis=d)-np.take(grid,0,axis=d)-H[:,d]))) for d in range(3))
            balance=abs(r['Fz_top']+r['Fz_bottom'])/max(abs(r['Fz_top']),1e-30)
            ferr=abs(r['Fz_top']-r['mean_first_piola'][2][2])/max(abs(r['Fz_top']),1e-30)
            result={'reaction_amplitude_N':-r['Fz_top']*L**2,'energy_N_mm':r['energy']*L**3,
                    'finite_response_normalized':r,'detF_by_occupancy':stats,'periodic_error_over_L':per,
                    'relative_force_balance':balance,'relative_macro_force_error':ferr,
                    'solve_seconds':solved-built,'Newton_corrections':max(p.trial_calls-1,0),
                    'relative_fluctuation_change':float(np.linalg.norm(np.asarray(sol[0]-w))/np.linalg.norm(np.asarray(w)))}
            np.savez_compressed(out/'jax_field.npz',w=np.asarray(sol[0]),H=H,N=cfg['N']);result['field_sha256']=sha(out/'jax_field.npz')
            checks={'finite':bool(np.isfinite(np.asarray(sol[0])).all() and np.isfinite(r['energy'])),
                    'positive_detF':r['J_min']>0,'void_min_detF_ge_0p1':stats['void']['min']>=.1,
                    'reduced_residual_l2_le_1e-8':r['reduced_residual_l2']<=1e-8,
                    'relative_force_balance_le_1e-6':balance<=1e-6,'relative_macro_reaction_vs_P_le_1e-6':ferr<=1e-6,
                    'XYZ_periodic_le_1e-10':per<=1e-10}
        result.update(status='ok' if all(checks.values()) else 'check_failed',checks=checks,thickness_mm=t_mm,compression=a,
            build_seconds=built-start,body_seconds=time.perf_counter()-start,petsc_options=options,
            peak_host_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
            input_sha256=sha(root/'input.json') if adjoint else sha(out/'input.json'))
        write(out/'jax.json',result);print(json.dumps(result,indent=2),flush=True)
        if not all(checks.values()):raise ValueError('Gradient/state checks failed; output retained')
    except Exception as exc:
        write(out/'failure.json',{'type':type(exc).__name__,'message':str(exc),'body_seconds':time.perf_counter()-start});raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','adjoint','forward'))
    parser.add_argument('case',type=Path);parser.add_argument('--thickness-mm',type=float,default=.5);parser.add_argument('--compression',type=float,default=.01)
    args=parser.parse_args();case=args.case.resolve()
    if args.action=='prepare':prepare(case,args.compression)
    else:run(case,args.thickness_mm,args.action=='adjoint',args.compression)

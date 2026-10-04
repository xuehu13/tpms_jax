"""Segmented XYZ compression of the locked 5% wall; reuse the existing FEM.

Prepare a shell segment, solve each JAX state once, then compare common nodes.
Occupancy remains attached to reference Gauss points, including the soft void.
No contact, plasticity, stabilization, new Newton method, or mesh scan is added.
"""
from pathlib import Path
import argparse, hashlib, json, os, resource, shutil, sys, time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with p.open('x') as f:f.write(json.dumps(v,indent=2,allow_nan=False,default=lambda x:x.item())+'\n')
def base(case):return case/'step3_xyz'
def tag(a):return 'a'+format(a,'.6f').replace('.','p')
def state(case,a):return base(case)/tag(a)


def prepare(case,a):
    if a not in (.01,.05,.10,.20):raise ValueError('Locked observation positions: 1,5,10,20 percent')
    old=case/'step2/diagnostic_xyz';cfg=json.loads((old/'input.json').read_text())
    assert json.loads((old/'comparison.json').read_text())['stiffness_work_screen_pass']
    assert cfg['N']==64 and cfg['thickness_mm']==.5 and cfg['eta']==1e-4
    out=state(case,a);out.mkdir(parents=True,exist_ok=False)
    E=cfg['E_MPa'];nu=cfg['nu'];mu=E/(2*(1+nu));kappa=E/(3*(1-2*nu))
    prefix=(old/'thin_shell.inp').read_text().split('*Step,',1)[0]
    prefix=prefix.replace('matched linear elastic shell','matched Neo-Hookean finite-strain shell')
    elastic=f"*Elastic\n{E}, {nu}"
    assert elastic in prefix
    prefix=prefix.replace(elastic,f'*Hyperelastic, Neo Hooke\n{mu/2:.16g}, {2/kappa:.16g}')
    # Separate jobs retain lower checkpoints; no unrequested higher load occurs.
    targets=[x for x in (.01,.05,.10,.20) if x<=a]
    lines=[prefix.rstrip()]
    for i,target in enumerate(targets):
        dt=.1 if target==.01 else .02
        lines+= [f'*Step, name={tag(target)}, nlgeom=YES, inc=200','*Static',f'{min(.05,dt)}, 1., 1e-7, {dt}',
                 '*Boundary',f"QZ, 3, 3, {-target*cfg['cell_size_mm']:.16g}",
                 '*Output, field, frequency=1','*Node Output','U, RF, UR',
                 '*Element Output','STH',
                 '*Output, history, frequency=1','*Energy Output','ALLSE, ALLAE, ALLWK',
                 '*Node Output, nset=QZ','U3, RF3','*End Step']
    (out/'thin_shell.inp').write_text('\n'.join(lines)+'\n')
    cfg.pop('diagnostic_xyz',None)  # XYZ is now the user's selected loading.
    cfg.update(step=3,compression=a,epsilon_z=-a,step_name=tag(a),
        boundary_role='user-selected XYZ compression',
        mode_comparison='area-weighted midsurface translations after relative and common gauge translation removal',
        mechanical_periodic_axes=[0,1,2],macro_F='diag(1,1,1-a); lateral macro strain fixed',
        material={'name':'compressible Neo-Hookean','mu_MPa':mu,'kappa_MPa':kappa,'C10_MPa':mu/2,'D1_per_MPa':2/kappa,
                  'energy':'mu/2*(J^(-2/3)*tr(F.T F)-3)+kappa/2*(J-1)^2'},
        geometry_update='rho fixed at reference material Gauss points; F=I+H+grad(w)',
        shell_units='mm,N,MPa; Standard static nlgeom=YES; S3R, five section points; no contact/plasticity',
        finite_work='integrate force with displacement; half(F*delta) is not a nonlinear work identity',
        source_sha256={n:sha(ROOT/n) for n in ('scripts/thin_target_finite.py','scripts/extract_thin_finite.py','hyperelastic_fem.py','fem.py','pbc.py','pixi.lock')},
        input_sha256={n:sha(case/n) for n in ('input.json','step1_decision.json','gauss_field.npz','step2/diagnostic_xyz/input.json','step2/diagnostic_xyz/jax.json','step2/diagnostic_xyz/jax_field.npz')},
        shell_inp_sha256=sha(out/'thin_shell.inp'))
    for n in cfg['source_sha256']:
        snapshot=out/'source_at_run'/n;snapshot.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,snapshot)
    write(out/'input.json',cfg)
    print(json.dumps({k:cfg[k] for k in ('compression','macro_F','material','shell_inp_sha256')},indent=2))


def capture(case,a,previous=None):
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    import jax;import jax.numpy as jnp
    from petsc4py import PETSc
    from fem import solve
    from hyperelastic_fem import make_density_hyperelastic_problem,reduced_guess,finite_response,neo_hookean_energy
    out=state(case,a);cfg=json.loads((out/'input.json').read_text());assert not (out/'jax.json').exists()
    for n,h in cfg['source_sha256'].items():assert sha(ROOT/n)==h,n
    for n,h in cfg['input_sha256'].items():assert sha(case/n)==h,n
    start=time.perf_counter();N=cfg['N'];L=cfg['cell_size_mm'];H=np.diag([0.,0.,-a])
    with np.load(case/'gauss_field.npz') as c:rho=c['rho'];qp=c['physical_quad_points'];qw=c['JxW']
    def field(p):
        assert np.array_equal(np.asarray(p.physical_quad_points),qp)
        assert np.array_equal(np.asarray(p.JxW)[:,0,:],qw)
        return rho
    p=make_density_hyperelastic_problem(N,rho_quad=field,eta=cfg['eta'],periodic_axes=(0,1,2))
    del qp,qw
    points=np.asarray(p.fe.points);p.set_params(H,p.rho,cfg['eta']);p.trial_calls=0
    grididx=np.rint(points*N).astype(int)
    if previous is None:
        with np.load(case/'step2/diagnostic_xyz/jax_field.npz') as f:
            total=f['total_u'];oldH=f['H'];oldw=total[grididx[:,0],grididx[:,1],grididx[:,2]]-points@oldH.T
        predictor=oldw*(a/abs(oldH[2,2]));seed='scaled frozen linear XYZ fluctuation'
    else:
        prev=state(case,previous);assert json.loads((prev/'jax.json').read_text())['status']=='ok'
        with np.load(prev/'jax_field.npz') as f:oldw=f['w']
        predictor=oldw*(a/previous);seed=f'scaled previous converged state {previous}'
    initial=p.detF_stats(jnp.asarray(predictor))
    if not initial['finite'] or initial['all_min']<=0:
        predictor=oldw;seed+='; invalid scaled predictor replaced by unscaled previous fluctuation'
        initial=p.detF_stats(jnp.asarray(predictor))
    if not initial['finite'] or initial['all_min']<=0:raise ValueError('No valid initial predictor at this load')
    # Use the original equations' norm. A left-preconditioned residual has
    # a different scale and hit restart roundoff before these equations
    # needed further correction. Independently recheck nonlinear equilibrium.
    options={'ksp_rtol':1e-8,'ksp_atol':1e-12,'ksp_max_it':3000,'ksp_error_if_not_converged':True,
             'ksp_gmres_modifiedgramschmidt':True,'ksp_pc_side':'right','ksp_norm_type':'unpreconditioned'}
    for k,v in options.items():PETSc.Options()[k]=v
    built=time.perf_counter();print('BUILD_DONE '+str(built-start),flush=True)
    write(out/'initial.json',{'compression':a,'seed':seed,'previous':previous,'detF':initial,'input_sha256':sha(out/'input.json')})
    try:
        sol=solve(p,{'newton':{'tol':1e-10,'rel_tol':1e-10,'initial_guess':reduced_guess(p,predictor),
                             'linear':{'petsc_solver':{'ksp_type':'gmres','pc_type':'gamg'}}}})
        jax.block_until_ready(sol);solved=time.perf_counter();print('SOLVE_DONE '+str(solved-built),flush=True)
        r=finite_response(p,sol);stats=p.detF_stats(sol[0]);w=np.asarray(sol[0]);u=w+points@H.T
        grid=np.empty((N+1,)*3+(3,));grid[grididx[:,0],grididx[:,1],grididx[:,2]]=u
        np.savez_compressed(out/'jax_field.npz',total_u=grid,w=w,N=N,H=H)
        F=jnp.eye(3)+p.H_macro+p.fe.sol_to_grad(sol[0]);weights=jnp.asarray(p.JxW)[:,0,:]
        W=jax.vmap(neo_hookean_energy)(F.reshape((-1,3,3))).reshape(rho.shape)
        mat=float(jnp.sum((1-cfg['eta'])*p.rho*W*weights));floor=float(jnp.sum(cfg['eta']*W*weights))
        void=float(jnp.sum(jnp.where(p.rho<=.05,p.stiffness_scale*W*weights,0)))
        per=max(float(np.max(np.abs(np.take(grid,-1,axis=d)-np.take(grid,0,axis=d)-H[:,d]))) for d in range(3))
        energy=r['energy'];balance=abs(r['Fz_top']+r['Fz_bottom'])/max(abs(r['Fz_top']),1e-30)
        ferr=abs(r['Fz_top']-r['mean_first_piola'][2][2])/max(abs(r['Fz_top']),1e-30)
        checks={'finite':bool(np.isfinite(u).all() and np.isfinite(energy)),
                'actual_reference_Gauss_mapping_exact':True,'positive_detF':r['J_min']>0,
                'void_min_detF_ge_0p1':stats['void']['min'] is None or stats['void']['min']>=.1,
                'reduced_residual_l2_le_1e-8':r['reduced_residual_l2']<=1e-8,
                'relative_force_balance_le_1e-6':balance<=1e-6,'relative_macro_reaction_vs_P_le_1e-6':ferr<=1e-6,
                'energy_partition':abs(mat+floor-energy)<=1e-10*max(1.,energy),'XYZ_periodic_le_1e-10':per<=1e-10}
        result={'status':'ok' if all(checks.values()) else 'check_failed','compression':a,'checks':checks,
                'Fz_N':r['Fz_top']*L**2,'energy_N_mm':energy*L**3,'secant_stiffness_N_per_mm':-r['Fz_top']*L/a,
                'finite_response_normalized':r,'detF_by_occupancy':stats,'periodic_error_over_L':per,
                'energy_material_N_mm':mat*L**3,'energy_uniform_floor_N_mm':floor*L**3,
                'high_void_energy_fraction':void/energy,'relative_macro_force_error':ferr,
                'build_seconds':built-start,'solve_seconds':solved-built,'body_seconds':time.perf_counter()-start,
                'Newton_corrections':max(p.trial_calls-1,0),'peak_host_RSS_GiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
                'seed':seed,'previous':previous,'petsc_options':options,'devices':[str(d) for d in jax.devices()],
                'field_sha256':sha(out/'jax_field.npz'),'input_sha256':sha(out/'input.json')}
        write(out/'jax.json',result);print(json.dumps(result,indent=2),flush=True)
        if not all(checks.values()):raise ValueError('State checks failed; field and output retained')
    except Exception as exc:
        fail={'type':type(exc).__name__,'message':str(exc),'attempted_compression':a,'previous_converged':previous,
              'trial_detF':getattr(p,'trial_detF',None),'body_seconds':time.perf_counter()-start}
        if hasattr(p,'invalid_trial_w'):
            np.savez_compressed(out/'invalid_trial.npz',w=p.invalid_trial_w,H=H,N=N)
            fail['invalid_trial_file']='invalid_trial.npz'
        write(out/'failure.json',fail);raise


def compare(case,a):
    out=state(case,a);cfg=json.loads((out/'input.json').read_text())
    j=json.loads((out/'jax.json').read_text());s=json.loads((out/'shell.json').read_text())
    assert j['status']=='ok' and s['status'] in ('ok','check_failed')
    assert all(s['checks'][k] for k in ('complete_step','finite','node_count','load_delta_stored_float_precision'))
    with np.load(case/'gauss_field.npz') as f:v,tri,ids=[f[k] for k in ('surface_vertices','surface_triangles','node_ids')]
    with np.load(out/'jax_field.npz') as f:grid=f['total_u']
    with np.load(out/'shell_field.npz') as f:
        lookup={int(n):u for n,u in zip(f['labels'],f['u_mm'])};su=np.array([lookup[int(n)] for n in ids])
    from scipy.interpolate import RegularGridInterpolator
    ju=RegularGridInterpolator([np.arange(cfg['N']+1)/cfg['N']]*3,grid)(v)*cfg['cell_size_mm']
    areas=np.linalg.norm(np.cross(v[tri[:,1]]-v[tri[:,0]],v[tri[:,2]]-v[tri[:,0]]),axis=1)/2
    weights=np.bincount(tri.ravel(),weights=np.repeat(areas/3,3),minlength=len(v))
    shift=np.average(ju-su,weights=weights,axis=0);ju-=shift
    affine=v*np.array([0.,0.,-a])*cfg['cell_size_mm']
    # Remove their shared arbitrary translation as well: otherwise a shell
    # gauge inside the cell can inflate the fluctuation-mode cosine.
    common_shift=np.average(su-affine,weights=weights,axis=0)
    ju-=common_shift;su-=common_shift;jw=ju-affine;sw=su-affine
    dot=lambda x,y:float(np.sum(weights[:,None]*x*y))
    cosine=dot(jw,sw)/np.sqrt(dot(jw,jw)*dot(sw,sw))
    fd=abs(j['Fz_N']/s['Fz_N']-1);ed=abs(j['energy_N_mm']/s['energy_N_mm']-1)
    result={'compression':a,'relative_force_difference':fd,'relative_stored_energy_difference':ed,
            'relative_total_shell_energy_difference':abs(j['energy_N_mm']/s['total_energy_N_mm']-1),
            'fluctuation_vector_cosine':cosine,'area_weighted_relative_fluctuation_error':np.sqrt(dot(jw-sw,jw-sw)/dot(sw,sw)),
            'area_weighted_relative_total_u_error':np.sqrt(dot(ju-su,ju-su)/dot(su,su)),
            'gauge_translation_mm':shift.tolist(),'shared_gauge_translation_removed_mm':common_shift.tolist(),
            'reference_quality_pass':s['status']=='ok',
            'force_energy_screen_pass':bool(fd<=.1 and ed<=.1 and s['status']=='ok'),
            'visual_review_required':True,'mode_note':'midsurface translations only; no fit of rotation or scale; not local-stress certification',
            'input_sha256':sha(out/'input.json'),'comparison_source_sha256':sha(Path(__file__))}
    write(out/'comparison.json',result)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig=plt.figure(figsize=(12,5));vmax=max(np.max(np.linalg.norm(jw,axis=1)),np.max(np.linalg.norm(sw,axis=1)))
    for i,(title,u,w) in enumerate([('Abaqus S3R',su,sw),('JAX HEX8 N64',ju,jw)]):
        ax=fig.add_subplot(1,2,i+1,projection='3d');coll=Poly3DCollection((v+u/cfg['cell_size_mm'])[tri],linewidths=0)
        coll.set_array(np.linalg.norm(w,axis=1)[tri].mean(axis=1));coll.set_cmap('viridis');coll.set_clim(0,vmax)
        ax.add_collection3d(coll);ax.set(xlim=(0,1),ylim=(0,1),zlim=(0,1),title=f'{title}: {a:.0%}, actual deformation')
        ax.set_box_aspect((1,1,1));fig.colorbar(coll,ax=ax,shrink=.65,label='non-affine displacement (mm)')
    fig.tight_layout();fig.savefig(out/'modes.png',dpi=160);plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','jax','compare'))
    p.add_argument('case',type=Path);p.add_argument('--compression',type=float,required=True);p.add_argument('--previous',type=float)
    a=p.parse_args();c=a.case.resolve()
    if a.action=='jax':capture(c,a.compression,a.previous)
    else:{'prepare':prepare,'compare':compare}[a.action](c,a.compression)

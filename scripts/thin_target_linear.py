"""Step 2 only: matched N64 background / elastic S3R shell comparison.

Run prepare, jax, then compare on validation/thin_target_20261004_r5.
The archived Step 1 Gauss cache is reused only after exact mapping checks.
No constitutive, geometry, stabilization or solver implementation is changed.
"""
from pathlib import Path
import argparse, hashlib, json, os, resource, sys, time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    payload=json.dumps(value, indent=2, allow_nan=False, default=lambda x:x.item())+'\n'
    with p.open('x') as stream:
        stream.write(payload)


def output(case, diagnostic_xyz=False):
    return case/'step2'/'diagnostic_xyz' if diagnostic_xyz else case/'step2'


def prepare(case, diagnostic_xyz=False):
    cfg = json.loads((case/'input.json').read_text())
    decision = json.loads((case/'step1_decision.json').read_text())
    assert decision['status'] == 'geometry_and_Gauss_representation_ready_for_step2'
    out = output(case,diagnostic_xyz); out.mkdir(exist_ok=False)
    with np.load(case/'gauss_field.npz') as field:
        v, f, ids, eids = [field[k] for k in ('surface_vertices','surface_triangles','node_ids','element_ids')]
    relations = json.loads((case/'input/reference_xy_pbc.json').read_text())['relations']
    if diagnostic_xyz:
        groups={}
        for i,p in enumerate(v):groups.setdefault(tuple(np.round(np.where(np.isclose(p,1,atol=1e-9),0,p),9)),[]).append(i)
        relations=[]
        for members in groups.values():
            root=min(members)
            for i in members:
                if i!=root:relations.append({'node':i,'root':root,'shift':np.rint(v[i]-v[root]).astype(int).tolist()})
    slave = {r['node'] for r in relations}; roots = {r['root'] for r in relations}
    assert len(slave) == len(relations) and not slave & roots
    assert len(ids) == len(set(ids)) and len(eids) == len(set(eids))
    error = max(np.max(np.abs(v[r['node']]-v[r['root']]-(np.array(r['shift']) if diagnostic_xyz else np.r_[r['shift'],0]))) for r in relations)
    assert error < 1e-9
    top = [i for i in range(len(v)) if np.isclose(v[i,2],1,atol=1e-10) and i not in slave]
    bottom = [i for i in range(len(v)) if np.isclose(v[i,2],0,atol=1e-10) and i not in slave]
    assert top and bottom
    pin = min([i for i in range(len(v)) if i not in slave],key=lambda i:int(ids[i])) if diagnostic_xyz else min(bottom, key=lambda i:int(ids[i]))
    qz = int(ids.max())+1
    inp = ['*Heading','5 percent thickness; matched linear elastic shell; no contact/plasticity', '*Node']
    L = cfg['cell_size_mm']; eps = -1e-4
    inp += [f'{n}, '+', '.join(f'{x:.16g}' for x in p*L) for n,p in zip(ids,v)]
    inp += [f'{qz}, 0., 0., {L}', '*Element, type=S3R, elset=WALL']
    inp += [f'{e}, '+', '.join(str(ids[i]) for i in tri) for e,tri in zip(eids,f)]
    inp += ['*Nset, nset=PHYSICAL']
    inp += [', '.join(str(n) for n in ids[i:i+16]) for i in range(0,len(ids),16)]
    inp += ['*Nset, nset=BOTTOM_ROOTS']
    inp += [', '.join(str(ids[n]) for n in bottom[i:i+16]) for i in range(0,len(bottom),16)]
    inp += ['*Nset, nset=QZ', str(qz),'*Material, name=BASE','*Elastic',f"{cfg['E_s']}, {cfg['nu']}",
            '*Shell Section, elset=WALL, material=BASE',f"{cfg['thickness_mm']}, 5"]
    # Slave variables are eliminated once. Roots are never earlier slaves;
    # top-root loading equations come last, after their periodic uses.
    for r in relations:
        for dof in range(1,7):
            sz=r['shift'][2] if diagnostic_xyz else 0
            line=f"{ids[r['node']]}, {dof}, 1., {ids[r['root']]}, {dof}, -1."
            terms=3 if dof==3 and sz else 2
            if terms==3:line+=f', {qz}, 3, {-sz}.'
            inp += ['*Equation',str(terms),line]
    if not diagnostic_xyz:
        for i in top:inp += ['*Equation','2',f'{ids[i]}, 3, 1., {qz}, 3, -1.']
    boundary=[f'{ids[pin]}, 1, 3, 0.'] if diagnostic_xyz else ['BOTTOM_ROOTS, 3, 3, 0.',f'{ids[pin]}, 1, 2, 0.']
    inp += ['*Boundary',*boundary,
            '*Step, name=COMPRESSION, nlgeom=NO','*Static','1., 1., 1e-5, 1.',
            '*Boundary',f'QZ, 3, 3, {eps*L:.16g}',
            '*Output, field, frequency=1','*Node Output','U, RF, UR',
            '*Output, history, frequency=1','*Energy Output','ALLSE, ALLAE, ALLWK', '*End Step']
    (out/'thin_shell.inp').write_text('\n'.join(inp)+'\n')
    manifest = {'step':2,'case_id':cfg['case_id'],'N':cfg['N'],'epsilon_z':eps,
        'diagnostic_xyz':diagnostic_xyz,'mechanical_periodic_axes':[0,1,2] if diagnostic_xyz else [0,1],
        'cell_size_mm':L,'thickness_mm':cfg['thickness_mm'],'E_MPa':cfg['E_s'],'nu':cfg['nu'],'eta':cfg['eta'],
        'interface_10_90_mm':L*cfg['interface_10_90_over_L'],
        'jax_units':'L=1; displacement times L, force times L^2, energy times L^3 give mm,N,N mm',
        'shell_units':'mm,N,MPa; Standard static nlgeom=NO; S3R, five section points',
        'bc':'XYZ periodic with macro Hzz; three translation gauges; no flat ends' if diagnostic_xyz else 'XY periodic, zero macro lateral strain; flat z end planes; two in-plane translation gauges',
        'shell_end_rotation':'periodic in XYZ' if diagnostic_xyz else 'free except XY-periodic ties; solid cap constrains pointwise uz, shell constrains midsurface uz',
        'pin_node_label':int(ids[pin]),'pin_xyz_mm':(v[pin]*L).tolist(),
        'physical_node_count':len(v),'element_count':len(f),'periodic_relations':len(relations),
        'xy_coordinate_error_over_L':float(error),'top_roots':len(top),'bottom_roots':len(bottom),
        'macro_control_label':qz,'screen_relative_stiffness':.10,
        'mode_comparison':'area-weighted midsurface translations after in-plane translation alignment; shell rotations have no direct HEX8 nodal counterpart',
        'input_sha256':{name:sha(case/name) for name in ('input.json','step1_decision.json','gauss_field.npz','input/reference_xy_pbc.json')},
        'source_sha256':{name:sha(ROOT/name) for name in ('scripts/thin_target_linear.py','scripts/extract_thin_shell.py','density_fem.py','fem.py','pbc.py','pixi.lock')},
        'shell_inp_sha256':sha(out/'thin_shell.inp')}
    write(out/'input.json',manifest)
    write(out/'periodic_relations.json',relations)
    print(json.dumps(manifest,indent=2))


def capture(case, diagnostic_xyz=False):
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64',True)
    from petsc4py import PETSc
    from density_fem import make_density_problem
    from fem import solve
    from pbc import xy_compression_fixed_dofs
    out=output(case,diagnostic_xyz); cfg=json.loads((out/'input.json').read_text())
    assert not (out/'jax.json').exists()
    for name,digest in cfg['input_sha256'].items(): assert sha(case/name)==digest,name
    for name,digest in cfg['source_sha256'].items(): assert sha(ROOT/name)==digest,name
    t0=time.perf_counter(); N=cfg['N']; H=np.diag([0.,0.,cfg['epsilon_z']])
    with np.load(case/'gauss_field.npz') as cache:
        rho=cache['rho']; old_points=cache['physical_quad_points']; old_weights=cache['JxW']
    def actual_field(problem):
        assert np.array_equal(np.asarray(problem.physical_quad_points),old_points)
        assert np.array_equal(np.asarray(problem.JxW)[:,0,:],old_weights)
        return rho
    pins={} if diagnostic_xyz else {'fixed_dofs':lambda points:xy_compression_fixed_dofs(points,N,N,N)}
    if diagnostic_xyz:pins['fixed_class']=(0,0,0)
    problem=make_density_problem(N,N,N,H,actual_field,E_s=cfg['E_MPa'],E_min=cfg['E_MPa']*cfg['eta'],
        periodic_axes=tuple(cfg.get('mechanical_periodic_axes',[0,1])),**pins)
    del old_points,old_weights
    t1=time.perf_counter(); print('BUILD_DONE',t1-t0,flush=True)
    options={'ksp_rtol':1e-11,'ksp_atol':1e-13,'ksp_max_it':5000,'ksp_error_if_not_converged':True}
    for key,val in options.items(): PETSc.Options()[key]=val
    sol=solve(problem,{'petsc_solver':{'ksp_type':'cg','pc_type':'gamg'}})
    jax.block_until_ready(sol); t2=time.perf_counter(); print('SOLVE_DONE',t2-t1,flush=True)
    points=np.asarray(problem.fe.points); w=np.asarray(sol[0]); r=np.asarray(problem.compute_residual(sol)[0])
    top=np.isclose(points[:,2],1,atol=1e-10); bottom=np.isclose(points[:,2],0,atol=1e-10)
    F=float(r[top,2].sum()); Fb=float(r[bottom,2].sum())
    red=float(np.max(np.abs(problem.P_mat.T@r.ravel())))
    # Device reductions avoid multiple host copies of 2M tensors.
    grad=problem.fe.sol_to_grad(sol[0])+jnp.asarray(H)
    eps=(grad+jnp.swapaxes(grad,-1,-2))/2
    lam,mu=problem.internal_vars[1:]
    sigma=lam[...,None,None]*jnp.trace(eps,axis1=-2,axis2=-1)[...,None,None]*jnp.eye(3)+2*mu[...,None,None]*eps
    weights=jnp.asarray(problem.JxW)[:,0,:]; V=float(weights.sum())
    avg=np.asarray(jnp.sum(sigma*weights[...,None,None],axis=(0,1))/V)
    U=float(.5*jnp.sum(sigma*eps*weights[...,None,None])); Um=float(.5*V*np.sum(avg*H))
    eq=np.sum(sigma*eps,axis=(-1,-2))*weights/2
    voidE=float(jnp.sum(jnp.where(problem.rho<=.05,eq,0)))
    grid=np.empty((N+1,N+1,N+1,3)); index=np.rint(points*N).astype(int)
    grid[index[:,0],index[:,1],index[:,2]]=w+points@H.T
    periodic=max(float(np.max(np.abs(np.take(grid,-1,axis=a)-np.take(grid,0,axis=a)-H[:,a]))) for a in cfg.get('mechanical_periodic_axes',[0,1]))
    loaded=float(np.max(np.abs(w[top|bottom,2])))
    L=cfg['cell_size_mm']; force_error=abs(F-avg[2,2])/max(abs(F),1e-30)
    work_error=abs(U-Um)/max(abs(U),1e-30)
    checks={'finite':bool(np.isfinite(grid).all() and np.isfinite([F,Fb,U,Um,red]).all()),
        'actual_Gauss_mapping_exact':True,'reduced_residual_le_1e-8':red<=1e-8,
        'relative_force_balance_le_1e-6':abs(F+Fb)<=1e-6*abs(F),
        'relative_reaction_stress_le_1e-6':force_error<=1e-6,
        'relative_energy_work_le_1e-6':work_error<=1e-6,'periodic_le_1e-10':periodic<=1e-10}
    if not diagnostic_xyz:checks['flat_z_faces_le_1e-10']=loaded<=1e-10
    checks={name:bool(val) for name,val in checks.items()}
    field=out/'jax_field.npz'; np.savez_compressed(field,total_u=grid,N=N,H=H)
    result={'status':'ok' if all(checks.values()) else 'check_failed','checks':checks,
        'Fz_top_N':F*L**2,'Fz_bottom_N':Fb*L**2,'stiffness_N_per_mm':F*L/cfg['epsilon_z'],
        'effective_axial_MPa':F/cfg['epsilon_z'],'energy_N_mm':U*L**3,'macro_energy_N_mm':Um*L**3,
        'average_stress_MPa':avg.tolist(),'reduced_residual_normalized':red,
        'relative_reaction_stress_error':force_error,'relative_energy_work_error':work_error,
        'high_void_energy_fraction':voidE/U,'periodic_error_over_L':periodic,
        'build_seconds':t1-t0,'solve_seconds':t2-t1,'body_seconds':time.perf_counter()-t0,
        'peak_host_RSS_GiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        'solver':'Existing PETSc CG/GAMG, unchanged JAX-FEM operator','petsc_options':options,
        'devices':[str(d) for d in jax.devices()],'jax_version':jax.__version__,
        'field_sha256':sha(field),'input_sha256':sha(out/'input.json')}
    write(out/'jax.json',result); print(json.dumps(result,indent=2),flush=True)
    if not all(checks.values()):raise RuntimeError('JAX consistency failed; result retained')


def compare(case, diagnostic_xyz=False):
    out=output(case,diagnostic_xyz); cfg=json.loads((out/'input.json').read_text())
    j=json.loads((out/'jax.json').read_text()); s=json.loads((out/'shell.json').read_text())
    assert j['status']==s['status']=='ok'
    with np.load(case/'gauss_field.npz') as cache:
        v,f,ids=[cache[k] for k in ('surface_vertices','surface_triangles','node_ids')]
    with np.load(out/'jax_field.npz') as field:grid=field['total_u']
    from scipy.interpolate import RegularGridInterpolator
    N=cfg['N']; coords=[np.arange(N+1)/N]*3
    ju=RegularGridInterpolator(coords,grid)(v)*cfg['cell_size_mm']
    lookup={row['label']:row['U_mm'] for row in s['nodes']}; su=np.array([lookup[int(n)] for n in ids])
    area=np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=1)/2
    nodal_area=np.bincount(f.ravel(),weights=np.repeat(area/3,3),minlength=len(v))
    # Remove the two different gauge translations only, never fit rotation/scale.
    shift=np.average(ju-su,weights=nodal_area,axis=0)
    if not diagnostic_xyz:shift[2]=0.
    ju-=shift
    affine=v*np.array([0,0,cfg['epsilon_z']])*cfg['cell_size_mm']
    jw,sw=ju-affine,su-affine
    dot=lambda a,b:float(np.sum(nodal_area[:,None]*a*b))
    relative=lambda a,b:np.sqrt(dot(a-b,a-b)/dot(b,b))
    corr=dot(jw,sw)/np.sqrt(dot(jw,jw)*dot(sw,sw))
    kd=abs(j['stiffness_N_per_mm']/s['stiffness_N_per_mm']-1)
    result={'step':2,'relative_stiffness_difference':kd,'relative_energy_difference':abs(j['energy_N_mm']/s['energy_N_mm']-1),
        'jax_stiffness_N_per_mm':j['stiffness_N_per_mm'],'shell_stiffness_N_per_mm':s['stiffness_N_per_mm'],
        'area_weighted_relative_total_displacement_error':relative(ju,su),
        'area_weighted_relative_fluctuation_error':relative(jw,sw),'fluctuation_vector_cosine':corr,
        'in_plane_gauge_shift_mm':shift.tolist(),'component_fluctuation_cosines':[],
        'mode_note':'Cosines and weighted errors describe translations on the common triangular midsurface, not shell/solid local stress or rotation equivalence',
        'stiffness_work_screen_pass':bool(kd<=cfg['screen_relative_stiffness']),
        'visual_mode_review_required':True,'input_sha256':sha(out/'input.json')}
    for a in range(3):
        result['component_fluctuation_cosines'].append(float(np.sum(nodal_area*jw[:,a]*sw[:,a])/np.sqrt(np.sum(nodal_area*jw[:,a]**2)*np.sum(nodal_area*sw[:,a]**2))))
    write(out/'comparison.json',result)
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig=plt.figure(figsize=(12,8)); vmax=max(np.max(np.linalg.norm(sw,axis=1)),np.max(np.linalg.norm(jw,axis=1)))
    for i,(title,u,w) in enumerate([('Abaqus S3R',su,sw),('JAX HEX8 N64',ju,jw)]):
        ax=fig.add_subplot(2,2,i+1,projection='3d'); deformed=v+u/cfg['cell_size_mm']*600
        val=np.linalg.norm(w,axis=1)[f].mean(axis=1)
        coll=Poly3DCollection(deformed[f],linewidths=0,alpha=1);coll.set_array(val);coll.set_cmap('viridis');coll.set_clim(0,vmax)
        ax.add_collection3d(coll);ax.set(xlim=(0,1),ylim=(0,1),zlim=(0,1),title=title+' (displacement x600)');ax.set_box_aspect((1,1,1))
        fig.colorbar(coll,ax=ax,shrink=.6,label='non-affine |u| (mm)')
    ax=fig.add_subplot(2,2,3);labels=['ux fluctuation','uy fluctuation','uz fluctuation']
    for a,label in enumerate(labels):ax.scatter(sw[:,a]/.001,jw[:,a]/.001,s=1,alpha=.3,label=label)
    ax.set(xlabel='Shell fluctuation / imposed compression',ylabel='JAX fluctuation / imposed compression',title='Common midsurface nodes; translation aligned');ax.legend(markerscale=4)
    ax=fig.add_subplot(2,2,4);err=np.linalg.norm(ju-su,axis=1)/.001
    ax.scatter(v[:,2],err,s=2,alpha=.25);ax.set(xlabel='z/L',ylabel='displacement mismatch / imposed compression',title='Mismatch versus height')
    fig.tight_layout();fig.savefig(out/'modes.png',dpi=160);plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','jax','compare'));parser.add_argument('case',type=Path)
    parser.add_argument('--diagnostic-xyz',action='store_true',help='One boundary diagnostic, not the original flat-end acceptance case')
    args=parser.parse_args();{'prepare':prepare,'jax':capture,'compare':compare}[args.action](args.case.resolve(),args.diagnostic_xyz)

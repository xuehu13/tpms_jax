"""Build matched geometry/Gauss/HRZ/XYZ shell inputs, with no displacement solve."""
from pathlib import Path
import hashlib,importlib.metadata,json,os,resource,shutil,sys,time
os.environ['JAX_PLATFORMS']='cpu'
R=Path('/home/xuehu/projects/tpms_jax');sys.path.insert(0,str(R))
O=R/'validation/geometry_transfer_20261006_r15'
import numpy as np
from scipy.special import expit
from scipy.spatial import cKDTree
import jax,jax.numpy as jnp
from scripts.prepare_thin_target import read_shell_mesh,periodic_components
from scripts.thin_target_explicit import ExplicitXYZ
from surface_distance import PeriodicSurfaceDistance
from hyperelastic_fem import make_density_hyperelastic_problem
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item())+'\n')
cfg=json.loads((O/'input.json').read_text());assert not (O/'preparation.json').exists()
for k,v in cfg['solver_source_sha256'].items():assert sha(R/k)==v
start=time.perf_counter();L=cfg['cell_size_mm'];N=cfg['N'];t=cfg['thickness_mm']/L
ell=cfg['interface_10_90_mm']/L/(2*np.log(9))
vmm,f,ids,eids=read_shell_mesh(O/'input/shell_mesh.inc');v=vmm/L
assert len(v)==cfg['reused_screen']['nodes'] and len(f)==cfg['reused_screen']['triangles']
np.savez_compressed(O/'surface_geometry.npz',surface_vertices=v,surface_triangles=f,node_ids=ids,element_ids=eids)
surface=PeriodicSurfaceDistance(v,f)
cross=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])
areas=np.linalg.norm(cross,axis=1)/2;assert np.min(areas)>0
normal=cross/(2*areas[:,None]);centers=v[f].mean(axis=1)
ratios=np.r_[surface.query(centers+t/2*normal),surface.query(centers-t/2*normal)]/(t/2)
rng=np.random.default_rng(20261006);sample=rng.choice(len(f),512,replace=False,p=areas/areas.sum())
roots=[];bracket_failures=0
for sign in [-1,1]:
    p=centers[sample];direction=sign*normal[sample];lo=np.zeros(len(p));hi=np.full(len(p),t)
    bracket_failures+=int(np.sum(surface.query(p+hi[:,None]*direction)<=t/2))
    for _ in range(22):
        mid=(lo+hi)/2;inside=surface.query(p+mid[:,None]*direction)<=t/2
        lo=np.where(inside,mid,lo);hi=np.where(inside,hi,mid)
    roots.append((lo+hi)/2)
width=(roots[0]+roots[1])*L
band={'nominal_thickness_mm':t*L,'sampled_normal_width_mm_quantiles':np.quantile(width,[0,.01,.5,.99,1]).tolist(),
 'all_facet_endpoint_distance_ratio_quantiles':np.quantile(ratios,[0,.01,.5,1]).tolist(),
 'normal_bracket_failures':bracket_failures,'normal_samples':512,
 'scope':'Local sampled thickness-band diagnostics; not global offset injectivity or contact certification'}
print('Local band diagnostics completed',flush=True)
collected={}
def field(problem):
    X=np.asarray(problem.physical_quad_points);weights=np.asarray(problem.fe.JxW)
    qstart=time.perf_counter();distance=surface.query(X);phi=expit((t/2-distance)/ell)
    collected.update(X=X,weights=weights,distance=distance,phi=phi,query_seconds=time.perf_counter()-qstart)
    print('Actual HEX27 Gauss occupancy: '+str(distance.size)+' points',flush=True)
    return phi
problem=make_density_hyperelastic_problem(N,rho_quad=field,eta=cfg['eta'],periodic_axes=(0,1,2),
                                        element_degree=2,quadrature_order=4,material_model='objective_void')
np.testing.assert_array_equal(np.asarray(problem.rho),collected['phi'])
X=collected['X'];weights=collected['weights'];phi=collected['phi']
assert X.shape==(32**3,27,3) and weights.shape==phi.shape==(32**3,27)
assert np.isfinite(weights).all() and weights.min()>0
points=np.asarray(problem.fe.points);cells=np.asarray(problem.fe.cells)
cell_centers=points[cells].mean(axis=1)
mid_cells=cKDTree(cell_centers).query(centers)[1];mid_max=phi[mid_cells].max(axis=1)
ijk=np.minimum((cell_centers*N).astype(int),N-1);grid=np.zeros((N,)*3);grid[tuple(ijk.T)]=phi.mean(axis=1)
ex=ExplicitXYZ(problem,L=L,density=cfg['solid_density_tonne_per_mm3'],force_batch_cells=2048)
nodem=np.asarray(ex.nodem);mass=np.asarray(ex.mass)
weighted_vf=float(np.sum(phi*weights)/weights.sum());shell_vf=float(areas.sum()*t)
expected_mass=cfg['solid_density_tonne_per_mm3']*L**3*np.sum((cfg['eta']+(1-cfg['eta'])*phi)*weights)
physical_mass=float(nodem.sum()*L)
assert np.isfinite(mass).all() and np.min(mass)>0 and np.min(nodem)>0
np.testing.assert_allclose(physical_mass,expected_mass,rtol=1e-12,atol=0)
np.testing.assert_allclose(nodem.sum(),mass.sum(),rtol=1e-12,atol=0)
energy=jax.jit(jax.vmap(lambda s:problem.material_energy(jnp.eye(3),s)))(problem.stiffness_scale.ravel())
energy=np.asarray(energy);assert np.isfinite(energy).all() and np.max(np.abs(energy))<1e-12
qs=rng.random((1024,3));periodic_errors=[float(np.max(np.abs(surface.query(qs)-surface.query(qs+np.eye(3)[k])))) for k in range(3)]
np.savez_compressed(O/'gauss_field.npz',physical_quad_points=X,JxW=weights,distance=collected['distance'],rho=phi,
                    surface_vertices=v,surface_triangles=f,node_ids=ids,element_ids=eids)
np.savez_compressed(O/'hrz_mass.npz',nodal_mass_normalized=nodem,periodic_class_mass_normalized=mass,
                    class_ids=np.asarray(ex.ids),physical_mass_multiplier=L)
geometry={'N':N,'cells':len(cells),'nodes':len(points),'Gauss_points_per_cell':27,'actual_Gauss_points':phi.size,
 'reference_volume_normalized':float(weights.sum()),'Gauss_weights_positive':True,
 'occupancy_exactly_in_own_problem':True,'weighted_projected_Vf':weighted_vf,
 'same_Gauss_binary_Vf':float(np.sum((collected['distance']<=t/2)*weights)/weights.sum()),
 'area_times_thickness_nominal_shell_Vf':shell_vf,'projected_Vf_vs_shell_area_thickness_relative':weighted_vf/shell_vf-1,
 'volume_note':'Same physical midsurface/thickness target; different geometric representations need not have identical integrated volume.',
 'mid_facet_cell_max_phi_min':float(mid_max.min()),'mid_facet_cells_with_no_phi_ge_half':int(np.sum(mid_max<.5)),
 'cell_mean_phi_ge_half_periodic_components':periodic_components(grid>=.5),
 'connectivity_note':'Thresholded-cell diagnostic only, not a mechanical connectivity certificate.',
 'periodic_distance_errors_over_L':periodic_errors,'initial_actual_detF':1.,'initial_all_material_energy_finite':True,
 'initial_energy_density_absolute_max_MPa':float(np.max(np.abs(energy))),
 'mass_lumping':ex.mass_lumping,'minimum_HRZ_nodal_mass_normalized':float(nodem.min()),
 'minimum_periodic_class_mass_normalized':float(mass.min()),'cell_mass_conservation_relative_error':ex.mass_conservation_error,
 'negative_unmodified_row_mass_entries':ex.negative_row_mass_entries,
 'JAX_total_physical_mass_tonne':physical_mass,'shell_area_thickness_mass_tonne':shell_vf*L**3*cfg['solid_density_tonne_per_mm3'],
 'stable_dt_estimate_seconds':ex.dt_estimate,'query_seconds':collected['query_seconds']}
# Same XYZ quotient/root policy as the maintained thin_target_linear.prepare.
groups={}
for i,p in enumerate(v):groups.setdefault(tuple(np.round(np.where(np.isclose(p,1,atol=1e-9),0,p),9)),[]).append(i)
relations=[]
for members in groups.values():
    root=min(members)
    for i in members:
        if i!=root:relations.append({'node':i,'root':root,'shift':np.rint(v[i]-v[root]).astype(int).tolist()})
slaves={r['node'] for r in relations};roots_set={r['root'] for r in relations}
assert len(slaves)==len(relations) and not slaves&roots_set
error=max(np.max(np.abs(v[r['node']]-v[r['root']]-r['shift'])) for r in relations)
assert error<1e-9
pin=min((i for i in range(len(v)) if i not in slaves),key=lambda i:int(ids[i]));qz=int(ids.max())+1
assert qz not in ids and int(ids[pin]) not in {int(ids[r['node']]) for r in relations}
lines=['*Heading','diverse_04 matched 0.50mm NH XYZ shell; no contact/plasticity','*Node']
lines += [f'{n}, '+', '.join(f'{x:.16g}' for x in p) for n,p in zip(ids,vmm)]
lines += [f'{qz}, 0., 0., {L}','*Element, type=S3R, elset=WALL']
lines += [f'{e}, '+', '.join(str(ids[i]) for i in tri) for e,tri in zip(eids,f)]
lines += ['*Nset, nset=PHYSICAL']+[', '.join(str(n) for n in ids[i:i+16]) for i in range(0,len(ids),16)]
lines += ['*Nset, nset=QZ',str(qz),'*Material, name=BASE','*Density','1e-9',
          '*Hyperelastic, Neo Hooke','1.923076923076923, 0.24','*Shell Section, elset=WALL, material=BASE','0.5, 5']
for r in relations:
    for dof in range(1,7):
        shift=r['shift'][2];terms=3 if dof==3 and shift else 2
        text=f"{ids[r['node']]}, {dof}, 1., {ids[r['root']]}, {dof}, -1."
        if terms==3:text+=f', {qz}, 3, {-shift}.'
        lines+=['*Equation',str(terms),text]
lines+=['*Boundary',f'{ids[pin]}, 1, 3, 0.',
 '*Amplitude, name=MACRO, definition=SMOOTH STEP, time=TOTAL TIME','0., 0., 0.04, 1., 0.044, 1.',
 '*Step, name=compress20, nlgeom=YES','*Dynamic, Explicit',', 0.044',
 '*Boundary, amplitude=MACRO','QZ, 3, 3, -2.',
 '*Output, field, number interval=40','*Node Output','U, UR, RF','*Element Output','S, LE, STH',
 '*Output, history, time interval=4.400000000000001e-05','*Energy Output',
 'ALLIE, ALLSE, ALLKE, ALLAE, ALLVD, ALLWK, ETOTAL','*Node Output, nset=QZ','U3, RF3','*End Step']
native=Path('/mnt/e/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus/geometry_transfer_20261006_r15_diverse04_explicit_T0p040')
native.mkdir(exist_ok=False);pack=O/'abaqus/explicit_T0p040';pack.mkdir(parents=True)
(native/'thin_shell.inp').write_text('\n'.join(lines)+'\n')
shellcfg={k:cfg[k] for k in ['case_id','N','cell_size_mm','thickness_mm','E_MPa','nu','eta','interface_10_90_mm','mechanical_periodic_axes','solid_density_tonne_per_mm3']}
shellcfg.update(step_name='compress20',load_time_seconds=.04,hold_time_seconds=.004,total_time_seconds=.044,
 physical_node_count=len(v),element_count=len(f),periodic_relations=len(relations),macro_control_label=qz,
 pin_node_label=int(ids[pin]),pin_xyz_mm=vmm[pin].tolist(),no_contact=True,no_plasticity=True,no_mass_scaling=True,
 shell_units='mm,N,MPa,tonne,s; S3R five section points; Explicit with default bulk viscosity',
 macro_F='diag(1,1,1-a); XYZ periodic fluctuations and rotations; lateral macro strain zero',
 shell_inp_sha256=sha(native/'thin_shell.inp'),original_mesh_sha256=sha(O/'input/shell_mesh.inc'),
 material={'C10_MPa':1.923076923076923,'D1_per_MPa':.24},
 boundary_role='Periodic cell; no pressure plates or periodic image contact',source_case=str(O),
 extractor_sha256=sha(R/'scripts/extract_thin_explicit.py'))
write(native/'input.json',shellcfg);shutil.copy2(R/'scripts/extract_thin_explicit.py',native/'extract_thin_explicit.py')
for name in ['thin_shell.inp','input.json','extract_thin_explicit.py']:shutil.copy2(native/name,pack/name)
write(O/'input/XYZ_periodic_relations.json',relations)
checks={'same_original_midsurface':True,'same_physical_thickness_material_density':True,
 'Gauss_field_in_own_HEX27_Problem':True,'positive_HRZ_mass':True,'HRZ_mass_conserved':ex.mass_conservation_error<1e-12,
 'XYZ_quotient_equations_consistent':error<1e-9,'local_normal_band_brackets':bracket_failures==0,
 'mid_facet_cells_Gauss_wall_coverage':bool(np.all(mid_max>=.5)),
 'initial_material_domain_valid':True,'periodic_distance_consistent':max(periodic_errors)<1e-10}
result={'main_plan_step':1,'step1_input_ready':all(checks.values()),'checks':checks,
 'band':band,'geometry_mass':geometry,
 'shell':{'node_count':len(v),'elements':len(f),'XYZ_relations':len(relations),'equations':6*len(relations),
  'periodic_coordinate_error_over_L':float(error),'pin_label':int(ids[pin]),'macro_control_label':qz,'native_job_directory':str(native)},
 'source_sha256':cfg['solver_source_sha256'],'input_sha256':{p:sha(O/p) for p in ['input.json','input/shell_mesh.inc','surface_geometry.npz','gauss_field.npz','hrz_mass.npz']},
 'generator_sha256':sha(Path(__file__)),'reused_parser_and_periodic_policy_sha256':{p:sha(R/p) for p in ['scripts/prepare_thin_target.py','scripts/thin_target_linear.py']},
 'environment':{name:importlib.metadata.version(name) for name in ['numpy','scipy','jax','jax-fem','fenics-basix','libigl']},
 'body_seconds':time.perf_counter()-start,'peak_RSS_GiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
 'new_displacement_solves':0,'new_Abaqus_jobs':0,'new_full_AD_jobs':0,
 'interpretation':'Matched-input readiness only; no response, equilibrium, contact, or gradient validation.'}
write(O/'preparation.json',result);shutil.copy2(__file__,O/'tools/prepare_inputs.py')
print((O/'preparation.json').read_text(),flush=True)
if not result['step1_input_ready']:raise ValueError('Input readiness failed; preserve evidence and do not submit the next solve')

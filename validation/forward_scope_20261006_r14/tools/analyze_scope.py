"""Read-only four-step closure and a small existing-midsurface inventory.

No FEM, new geometry, gradient execution or Abaqus job. Geometry summaries are
screening evidence only; they do not predict mechanical response or certify AD.
"""
from pathlib import Path
import ast, hashlib, json, subprocess, sys
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

R=Path('/home/xuehu/projects/tpms_jax');O=R/'validation/forward_scope_20261006_r14'
BASE='10d7ce3ecf53abe1e7e450f5d592c7a958cc5618'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()==BASE
assert not subprocess.check_output(['git','status','--porcelain'],cwd=R,text=True).strip()
assert not O.exists();O.mkdir()
tracked=subprocess.check_output(['git','ls-files','-z'],cwd=R).decode().split('\0')
write(O/'before_manifest.json',{s:sha(R/s) for s in tracked if s})
prior=R/'validation/void_continuation_20261006_r12';thick=R/'validation/thickness_range_20261006_r13'
files=[prior/'decision.json',prior/'rate_check.json',prior/'T0p004_analysis/comparison.json',
       prior/'T0p008_analysis/comparison.json',thick/'decision.json',
       thick/'t0p45/analysis/comparison.json',thick/'t0p55/analysis/comparison.json',
       R/'validation/large_compression_20261005_r6/gradient20_path_20261005/gradient_validation.json',
       R/'validation/large_compression_20261005_r6/gradient20_path_20261005/path_ad_full/result.json',
       R/'scripts/prepare_thin_target.py',R/'tests/test_objective_explicit.py',R/'hyperelastic_fem.py',
       R/'scripts/thin_target_explicit.py',R/'surface_distance.py',R/'pixi.lock']
input_hashes={str(p):sha(p) for p in files}
metrics={.45:read(files[5]),.50:read(files[2]),.55:read(files[6])}
rows=[]
for t,m in metrics.items():
    shell=read(R/('validation/large_compression_20261005_r6/abaqus/explicit_T0p040/shell.json' if t==.5 else
                 'validation/thickness_range_20261006_r13/'+('t0p45' if t==.45 else 't0p55')+'/abaqus/explicit_T0p040/shell.json'))
    shellpath=Path(shell['odb_path'].replace('\\','/'))
    j={'thickness_mm':t,'JAX_hold_magnitude_N':abs(m['hold_mean_Fz_N']),
       'shell_hold_magnitude_N':abs(shell['hold_mean_Fz_N']),
       'hold_force_difference':m['terminal_force_difference_vs_shell'],
       'curve_RMS_over_fixed_peak':m['curve_RMS_over_Standard_peak'],
       'work_difference':m['input_work_difference_vs_shell'],'body_minutes':m['body_seconds']/60,
       'energy_work_gap':m['energy_work_gap'],'loading_fraction_KE_below_5pct':m['loading_time_fraction_KE_below_5pct'],
       'terminal_KE_over_U':m['terminal_KE_over_U'],
       'final_dense_required_positive_J_min':m['dense_sampling_required_positive_J_min'],
       'final_dense_invalid_material_points':m['dense_sampling_invalid_material_points'],
       'final_dense_full_material_defined':m['dense_sampling_material_defined'],
       'actual_virtual_nonpositive_J_points':m['dense_sampling_nonpositive_points'],
       'mode_cosine':m['mode']['shell']['fluctuation_vector_cosine'],
       'mode_relative_difference':m['mode']['shell']['area_weighted_relative_fluctuation_difference'],
       'shell_energy_drift':shell['global_energy_drift_relative_work'],
       'shell_strict_quality_pass':all(shell['checks'].values()),'rejected_blocks':m['rejected_blocks'],
       'separate_rate_check':t==.5}
    assert all(j[k]<=.1 for k in ['hold_force_difference','curve_RMS_over_fixed_peak','work_difference'])
    assert j['energy_work_gap']<.01 and j['loading_fraction_KE_below_5pct']>=.95 and j['terminal_KE_over_U']<.05
    assert j['final_dense_required_positive_J_min']>0 and j['final_dense_invalid_material_points']==0 and j['final_dense_full_material_defined']
    rows.append(j)
write(O/'forward_scope.json',{'rows':rows,'range':'Three sampled thicknesses only; not continuous interval certification',
      'scope':'diverse_28, L10mm, NH E10MPa nu.3, XYZ zero lateral macro strain, no contact/plasticity, HEX27 N32/27 points, 0-20%',
      'maximum_hold_force_difference':max(j['hold_force_difference'] for j in rows),
      'maximum_curve_metric':max(j['curve_RMS_over_fixed_peak'] for j in rows),
      'maximum_work_difference':max(j['work_difference'] for j in rows),
      'reference_quality_certified':False,'all_morphologies_certified':False,'gradient20_certified':False,
      'decision':'Continue scoped forward research; do not claim full differentiable replacement of Abaqus.'})

# Only reuse the generic read-only triangular mesh parser, not the old audit
# which hardcodes diverse_28's implicit expression.
source=(R/'scripts/prepare_thin_target.py').read_text();tree=ast.parse(source)
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='read_shell_mesh')
exec(compile(ast.Module(body=[node],type_ignores=[]),str(R/'scripts/prepare_thin_target.py'),'exec'),globals())
native=Path('/mnt/f/auto_abaqus/work/para_aly/Fine/T0p02/MS9');inventory={}
for name in ['diverse_28','diverse_04','diverse_05']:
    case=native/name;cfgp=case/'case_used.json';mesh=case/'abaqus/ingredients/shell_mesh.inc'
    cfg=read(cfgp);L=float(cfg['unit_cell_size_mm']);v,f,labels,elems=read_shell_mesh(mesh);v=v/L
    c=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]);area=np.linalg.norm(c,axis=1)/2
    assert np.isfinite(v).all() and area.min()>0
    normal=c/(2*area[:,None]);moment=np.einsum('f,fi,fj->ij',area,normal,normal)/area.sum()
    # Use the same periodic-quotient precision as the established surface audit.
    _,inverse=np.unique(np.round(v%1,9),axis=0,return_inverse=True);glued=inverse[f]
    directed=np.vstack([glued[:,[0,1]],glued[:,[1,2]],glued[:,[2,0]]])
    edges,which,counts=np.unique(np.sort(directed,axis=1),axis=0,return_inverse=True,return_counts=True)
    orient=np.bincount(which,weights=np.where(directed[:,0]<directed[:,1],1,-1))
    graph=coo_matrix((np.ones(2*len(edges)),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])))
    components=int(connected_components(graph,directed=False,return_labels=False))
    seams=[]
    for axis in range(3):
        lo=v[np.isclose(v[:,axis],0,atol=1e-9)].copy();hi=v[np.isclose(v[:,axis],1,atol=1e-9)].copy()
        lo[:,axis]=0;hi[:,axis]=0
        if len(lo) and len(hi):gap=float(max(cKDTree(lo).query(hi)[0].max(),cKDTree(hi).query(lo)[0].max()))
        else:gap=None
        seams.append({'axis':axis,'low_nodes':len(lo),'high_nodes':len(hi),'coordinate_gap_over_L':gap})
    passes=bool(components==1 and np.all(counts==2) and np.all(orient==0) and
                all(s['coordinate_gap_over_L'] is not None and s['coordinate_gap_over_L']<1e-8 for s in seams))
    j={'case':name,'source_directory':str(case),'implicit_expression':cfg['surface_expression'],
       'case_sha256':sha(cfgp),'mesh_sha256':sha(mesh),'L_mm':L,'nodes':len(v),'triangles':len(f),
       'coordinate_bounds_over_L':[v.min(axis=0).tolist(),v.max(axis=0).tolist()],
       'area_over_L2':float(area.sum()),'periodic_components':components,
       'edges_not_two':int((counts!=2).sum()),'orientation_conflicts':int((orient!=0).sum()),
       'seams':seams,'normal_second_moment':moment.tolist(),'basic_periodic_surface_screen_pass':passes,
       'new_XYZ_shell_reference_exists':name=='diverse_28',
       'strict_mathematical_minimal_surface_certified':False,
       'mechanical_response_predicted':False,'constant_thickness_overlap_or_contact_certified':False}
    inventory[name]=j;input_hashes[str(cfgp)]=sha(cfgp);input_hashes[str(mesh)]=sha(mesh)
write(O/'geometry_inventory.json',inventory)
write(O/'input_manifest.json',{'source_commit':BASE,'input_sha256':input_hashes,
      'analyzer_sha256':sha(Path(__file__)),'new_FEM_paths':0,'new_Abaqus_jobs':0,'new_full_AD_jobs':0,
      'geometry_screen_scope':'Read only three available original midsurfaces; no new mesh/occupancy generation or mechanics.',
      'environment':{'python':sys.version,'numpy':np.__version__}})
assert input_hashes=={s:sha(Path(s)) for s in input_hashes}
print(json.dumps({'forward_maxima':read(O/'forward_scope.json'),'geometry':{k:{s:v[s] for s in
      ['nodes','triangles','periodic_components','edges_not_two','orientation_conflicts','basic_periodic_surface_screen_pass','normal_second_moment']} for k,v in inventory.items()}},indent=2))

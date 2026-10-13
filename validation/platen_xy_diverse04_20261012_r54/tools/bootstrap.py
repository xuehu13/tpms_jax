"""Prepare one authorized diverse_04 platen comparison; never edit old evidence."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
import numpy as np
R=Path('/home/xuehu/projects/tpms_jax')
W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
B=R/'validation/platen_xy_diverse04_20261012_r54'
O=W/'output/r54_platen_xy_diverse04_20261012'
G=R/'validation/platen_xy_G_20261010_r51'
PD=R/'validation/platen_xy_PD_20261010_r52'
OLD=R/'validation/geometry_transfer_20261006_r15'
assert not B.exists() and not O.exists(),'Do not overwrite an existing attempt'
for p in [B/'tools',B/'checks',O]:p.mkdir(parents=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
protected=[R/f for f in ['hyperelastic_fem.py','surface_distance.py','fem.py','pbc.py','scripts/thin_target_explicit.py']]
protected+=list((G/'tools').glob('*.py'))+list((PD/'tools').glob('*.py'))
protected+=[OLD/'surface_geometry.npz',OLD/'gauss_field.npz',OLD/'input.json',
 G/'experiments/midsurface_rim_20pct_20261010/abaqus/platen_20pct.inp',
 G/'experiments/midsurface_rim_20pct_20261010/tools/run_jax_20.py',
 G/'experiments/midsurface_rim_20pct_20261010/tools/analyze_rim20.py',
 G/'experiments/midsurface_rim_20pct_20261010/tools/extract_rim20_frames.py',
 R/'validation/platen_xy_gradients_20261011_r53/status.json']
guard={str(p.relative_to(R)):sha(p) for p in protected};write(B/'checks/source_guard_before.json',guard)
with np.load(OLD/'surface_geometry.npz') as z:v=z['surface_vertices'];f=z['surface_triangles']
groups={}
for i,x in enumerate(v):
 key=tuple(np.round(np.r_[np.where(np.isclose(x[:2],1,atol=1e-10),0,x[:2]),x[2]],9));groups.setdefault(key,[]).append(i)
rel=[(i,min(ns)) for ns in groups.values() for i in ns if i!=min(ns)]
assert len({i for i,j in rel})==len(rel)
for i,j in rel:np.testing.assert_allclose(v[i]-v[j],np.r_[np.rint(v[i,:2]-v[j,:2]),0],atol=1e-9)
audit={'diverse_04':{'shell_surface':str(OLD/'surface_geometry.npz'),
 'source_sha256':{p.name:sha(p) for p in [OLD/'surface_geometry.npz',OLD/'gauss_field.npz',OLD/'input.json']},
 'xy_independent_relations':len(rel),'nodes':len(v),'triangles':len(f),
 'rim_node_count':[int(np.sum(np.isclose(v[:,2],s,atol=1e-10))) for s in [0,1]],
 'new_remeshing':False,'same_distance_surface_and_shell_vertices':True}}
write(B/'checks/geometry_precheck.json',audit)
protocol=json.loads((PD/'protocol_planned.json').read_text())
protocol.update(stage='authorized_execution',date='2026-10-12',execution_authorized=True,
 authorization='User explicitly requested diverse_04 20% comparison under latest platen/XY/midsurface-end model',
 request='Only diverse_04 paired 20% forward simulation; compare peak location, force and full response',
 cases=['diverse_04'],future_dynamic_jobs=2,jobs_submitted=0,execution_order=['diverse_04_Abaqus','diverse_04_JAX'],
 geometry_precheck='checks/geometry_precheck.json',
 implementation='Data/path-only adapters reuse r52 preparation and bounded runner, r51 shared mechanics and Abaqus controls; one attempt each')
protocol['metrics']['peak_rules']={
 'primary':'Raw global maximum upper-platen compressive force in actual loading window; report force, engineering compression and time',
 'location_difference':'JAX minus new Abaqus in compression percentage points; report raw sample spacing',
 'diagnostic_smoothing':'Centered moving mean on fixed 1001-point compression grid: width25 samples (~0.5 percentage points); no time warping',
 'first_prominent_peak':'On smoothed curve, scipy find_peaks prominence >=10% of its own full-loading absolute peak and distance>=25 samples; list all qualifying peaks, compare first if both have one',
 'endpoint_rule':'If global maximum is at end or no qualifying interior peak, mark no captured interior peak rather than claim a buckling peak',
 'incomplete_rule':'Compare common achieved window only; no full20 acceptance, no unseen peak extrapolation'}
protocol['metrics'].pop('contextual_fixed_normalizer_N',None)
protocol['metrics'].pop('contextual_normalizer_use',None)
protocol['cost_basis_minutes']={'prior_G_total':87.08,'prior_P_total':44.59,'prior_D_total':80.12}
write(B/'protocol.json',protocol);write(B/'protocol_planned.json',protocol);write(O/'protocol.json',protocol)
status={'stage':'authorized_prepared_for_input_generation','execution_authorized':True,'active_jobs':[],
 'case':'diverse_04','new_dynamic_jobs':0,'datachecks':0,'new_design_AD':False,'training':False,
 'heartbeat':'PAUSED; not re-enabled','original_r53_paused':True,'complete_pair':False}
write(B/'status.json',status);write(O/'status.json',status)

# Small adapters use the original tested code without copying mechanics.
prepare=r'''from pathlib import Path
import hashlib,json
B=Path(__file__).resolve().parents[1];R=B.parents[1]
source=R/'validation/platen_xy_PD_20261010_r52/tools/prepare_case.py'
original=source.read_text();code=original
replacements=[("choices=['P','D']","choices=['diverse_04']"),
 ("old=R/'validation/canonical_tpms_20261009_r50'/a.kind","old=R/'validation/geometry_transfer_20261006_r15'"),
 ("'surface.npz'","'surface_geometry.npz'"),('r52','r54')]
for old,new in replacements:
 assert old in code,old
 code=code.replace(old,new)
(B/'checks/prepare_recipe.json').write_text(json.dumps({'source':str(source),'source_sha256':hashlib.sha256(original.encode()).hexdigest(),'substitutions':replacements,'adapted_sha256':hashlib.sha256(code.encode()).hexdigest()},indent=2)+'\n')
exec(compile(code,str(source),'exec'),{'__name__':'__main__','__file__':str(Path(__file__).resolve())})
'''
run=r'''from pathlib import Path
import hashlib,json
B=Path(__file__).resolve().parents[1];R=B.parents[1]
source=R/'validation/platen_xy_PD_20261010_r52/tools/run_jax.py'
original=source.read_text();assert original.count("choices=['P','D']")==1
code=original.replace("choices=['P','D']","choices=['diverse_04']")
(B/'checks/run_jax_recipe.json').write_text(json.dumps({'source':str(source),'source_sha256':hashlib.sha256(original.encode()).hexdigest(),'substitution':"choices=['diverse_04']",'adapted_sha256':hashlib.sha256(code.encode()).hexdigest(),'same_mechanics_and_100min_budget':True},indent=2)+'\n')
exec(compile(code,str(source),'exec'),{'__name__':'__main__','__file__':str(Path(__file__).resolve())})
'''
analyze=r'''from pathlib import Path
import hashlib,json
B=Path(__file__).resolve().parents[1];R=B.parents[1];E=B/'diverse_04';G=R/'validation/platen_xy_G_20261010_r51'
cfg=json.loads((E/'input/protocol.json').read_text())
source=G/'experiments/midsurface_rim_20pct_20261010/tools/analyze_rim20.py';original=source.read_text();code=original
substitutions=[("E=Path(__file__).resolve().parents[1];D=E.parents[1];R=D.parents[1]",f"E=Path({str(E)!r});D=Path({str(G)!r});R=Path({str(R)!r})"),
 ("R/'validation/canonical_tpms_20261009_r50/G/surface_shell_quality.npz'",f"Path({cfg['source_shell_surface']!r})"),
 ("D/'input/cut_face_contact.npz'","E/'input/cut_face_contact.npz'"),
 ("fig.suptitle('G | L10 mm","fig.suptitle('diverse_04 | L10 mm")]
for old,new in substitutions:
 assert code.count(old)==1,(old,code.count(old))
 code=code.replace(old,new)
(E/'checks/postprocess_recipe.json').write_text(json.dumps({'source':str(source),'source_sha256':hashlib.sha256(original.encode()).hexdigest(),'substitutions':substitutions,'adapted_sha256':hashlib.sha256(code.encode()).hexdigest(),'new_FEM_calls':False},indent=2)+'\n')
exec(compile(code,str(source),'exec'),{'__name__':'__main__','__file__':str(source)})
'''
for name,content in [('prepare.py',prepare),('run_jax.py',run),('analyze.py',analyze)]:
 (B/'tools'/name).write_text(content)
write(B/'checks/authorization.json',{'UTC':datetime.now(timezone.utc).isoformat(),'scope':protocol['request'],
 'heartbeat_reenabled':False,'no_design_AD_training_or_Git':True,'source_guard_count':len(guard)})
write(O/'geometry_precheck.json',audit)
print(json.dumps({'B':str(B),'geometry':audit['diverse_04'],'wall_budget_minutes':100},ensure_ascii=False))

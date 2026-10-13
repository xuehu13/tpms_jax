"""Classify returned candidate modes; no eigensolve or time advance."""
from pathlib import Path
import json,time,shutil,sys,platform,importlib.metadata as meta
import numpy as np
R=Path('/home/xuehu/projects/tpms_jax');D=R/'validation/fine_mode_20261007_r22';old=R/'validation/critical_mode_20261007_r21'
W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
receipt=json.loads((D/'launch_receipt.json').read_text());start=time.perf_counter()
env={'Python':sys.version,'platform':platform.platform(),'packages':{x:meta.version(x) for x in ['jax','jaxlib','numpy','scipy','fenics-basix']},'GPU_preallocation_off_in_actual_source':True}
(D/'environment.json').write_text(json.dumps(env,indent=2))
if not (D/'result.json').exists():
 (D/'analysis_summary.json').write_text(json.dumps({'complete_result':False,'receipt':receipt,'no_mode_acceptance_or_scientific_cause_inferred_from_incomplete_call':True},indent=2));print(json.dumps(receipt));sys.exit(0)
result=json.loads((D/'result.json').read_text())
shell=R/'validation/shell_rate_20261007_r20/abaqus/explicit_T0p004/results/shell_frames'
with np.load(shell/'reference.npz') as f:xyz=f['xyz_mm'];tri=f['triangles'];normals=f['reference_vertex_normals'];area=f['reference_area_weights_mm2'];labels=f['labels']
with np.load(R/'validation/geometry_transfer_20261006_r15/surface_geometry.npz') as f:ids=f['node_ids']
assert np.array_equal(labels,ids)
sr=json.loads((shell/'modes.json').read_text())['rows']
def frame(a):
 row=min(sr,key=lambda r:abs(r['compression']-a))
 with np.load(shell/row['file']) as f:w=f['fluctuation_mm']
 return row,w
b,wb=frame(.1186);e,we=frame(.1382);fold=we-wb
with np.load(old/'physical_metric/accepted_a0.1200_deformation_candidate.npz') as f:initial=f['surface_mode_mm']
def cosine(x,y):return float(np.sum(area[:,None]*x*y)/np.sqrt(np.sum(area[:,None]*x*x)*np.sum(area[:,None]*y*y)))
def tris(t):
 t=np.sort(t,axis=1);return t[np.lexsort(t.T[::-1])]
fields=[];rows=[]
for row in result['rows']:
 i=row['rank_lowest_block']
 with np.load(D/f'mode{i}_surface.npz') as f:w=f['surface_mode_mm'];v=f['surface'];tt=f['triangles']
 assert np.array_equal((v*10).astype(np.float32).astype(float),xyz) and np.array_equal(tris(tt),tris(tri))
 curv=row['curvature_for_1mm_max_surface_mode_N_mm'];row=dict(row)
 row['absolute_cosine_with_fast_shell_event_increment']=abs(cosine(w,fold))
 row['absolute_cosine_with_initial_r21_candidate']=abs(cosine(w,initial))
 row['deformation_inertia_greater_than_mean_translation']=row['uniform_translation_inertia_fraction']<.5
 row['continued_void_curvature_N_mm']=row['parts_N_mm']['deep_void']+row['parts_N_mm']['mixed_tail']
 row['continued_void_curvature_fraction']=row['continued_void_curvature_N_mm']/curv if curv!=0 else None
 rows.append(row);sign=1 if cosine(w,fold)>=0 else -1;fields.append((row,np.einsum('ni,ni->n',w*sign,normals)))
summary={'complete_result':True,'compression':result['compression'],'rows':rows,'receipt':receipt,
 'shell_finite_increment_compressions':[b['compression'],e['compression']],'shell_increment_not_eigenmode':True,
 'cause_not_uniquely_identified':True,'new_time_advance':False,'new_design_AD':False}
(D/'analysis_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
fig=plt.figure(figsize=(10,9));lim=max(np.abs(w).max() for _,w in fields);norm=Normalize(-lim,lim);cmap=plt.get_cmap('coolwarm')
for i,(row,w) in enumerate(fields):
 ax=fig.add_subplot(2,2,i+1,projection='3d');ax.add_collection3d(Poly3DCollection(xyz[tri],facecolors=cmap(norm(w[tri].mean(axis=1))),edgecolors='none'))
 ax.set(xlim=(0,10),ylim=(0,10),zlim=(0,10));ax.set_box_aspect((1,1,1));ax.view_init(24,-58);ax.tick_params(labelsize=8)
 ax.set_title(f"Candidate {i+1}: curvature {row['curvature_for_1mm_max_surface_mode_N_mm']:.4g} N mm\nFine residual {row['original_fine_relative_residual']:.3g}; translation {100*row['uniform_translation_inertia_fraction']:.1f}%",fontsize=10,pad=10)
fig.suptitle('Original N32 space: fixed 12.1904% dynamic state\n1 mm maximum surface mode normalization. Colors: reference-normal component. Not actual deformation.',fontsize=11)
fig.subplots_adjust(left=.03,right=.87,top=.88,bottom=.05,hspace=.18,wspace=.1)
cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=fig.add_axes([.90,.18,.018,.58]));cb.set_label('Normal mode component (mm)')
fig.savefig(D/'modes.png',dpi=150,bbox_inches='tight');plt.close(fig)
shutil.copy2(D/'modes.png',W/'output/figures/FINE_MODE_modes.png')
shutil.copy2(W/'work/fine_mode_20261007/analyze.py',D/'analyze.py')
(D/'analysis_receipt.json').write_text(json.dumps({'wall_seconds':time.perf_counter()-start,'new_jobs':0,'new_design_AD':False},indent=2))
print(json.dumps(summary,indent=2))

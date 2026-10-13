"""Read retained Ritz modes and existing shell fields; no mechanical solve."""
from pathlib import Path
import json,time,shutil,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
R=Path('/home/xuehu/projects/tpms_jax');D=R/'validation/critical_mode_20261007_r21'
W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
A=D/'analysis';A.mkdir(exist_ok=False);start=time.perf_counter()
source=D/'results/result.json'
if not source.exists():source=D/'results/partial_result.json'
result=json.loads(source.read_text());rows=result['rows']
shell=R/'validation/shell_rate_20261007_r20/abaqus/explicit_T0p004/results/shell_frames'
with np.load(shell/'reference.npz') as f:xyz=f['xyz_mm'];tri=f['triangles'];area=f['reference_area_weights_mm2'];normals=f['reference_vertex_normals']
shellrows=json.loads((shell/'modes.json').read_text())['rows']
def frame(a):
 row=min(shellrows,key=lambda r:abs(r['compression']-a))
 with np.load(shell/row['file']) as f:w=f['fluctuation_mm']
 return row,w
before,wb=frame(.1186);after,wa=frame(.1382);fold=wa-wb
def cosine(x,y):
 return float(np.sum(area[:,None]*x*y)/np.sqrt(np.sum(area[:,None]*x*x)*np.sum(area[:,None]*y*y)))
selection=[];fields=[];physical=[];physical_fields=[]
for row in rows:
 f=D/'results'/f"{row['state'][:-4]}_N{row['Ncoarse']}_selected.npz"
 if not f.exists():continue
 with np.load(f) as z:w=z['surface_mode_mm'];surf=z['surface'];tri0=z['triangles']
 assert np.allclose(surf*10,xyz,atol=1e-12) and np.array_equal(tri0,tri)
 sign=1 if cosine(w,fold)>=0 else -1;w=w*sign
 ref=np.einsum('ni,ni->n',w,normals)
 current=dict(row['modes'][0]);current.update({'state':row['state'],'compression':row['compression'],'Ncoarse':row['Ncoarse'],
   'basis_eigenvalue':row['basis_eigenvalues'][0],'eigenpair_relative_residual':row['eigenpair_relative_residuals'][0],
   'absolute_cosine_with_fast_shell_event_increment':abs(cosine(w,fold)),
   'plot_sign':sign,'scope':'Shell event increment is an observed finite change, not an eigenmode. Sign alignment changes no geometry or event position.'})
 selection.append(current);fields.append((row,ref))
summary={'rows':selection,'shell_increment_compression':[before['compression'],after['compression']],
 'shell_increment_is_not_an_eigenmode':True,'all_original_domains_retained':True,
 'original_mesh_N32_unchanged':True,'partial_or_complete_source':str(source.relative_to(R)),
 'new_time_advance':False,'new_design_AD':False}
interpret=json.loads((D/'physical_metric/deformation_candidate.json').read_text())
for row in interpret['rows']:
 with np.load(D/'physical_metric'/f"{row['state'][:-4]}_deformation_candidate.npz") as z:w=z['surface_mode_mm'];surf=z['surface'];tri0=z['triangles']
 assert np.allclose(surf*10,xyz,atol=1e-12) and np.array_equal(tri0,tri)
 sign=1 if cosine(w,fold)>=0 else -1;w=w*sign
 current=dict(row);current['absolute_cosine_with_fast_shell_event_increment']=abs(cosine(w,fold));current['plot_sign']=sign
 virtual=row['parts_N_mm']['deep_void']+row['parts_N_mm']['mixed_tail']
 current['continued_virtual_curvature_fraction']=virtual/row['curvature_for_1mm_max_surface_mode_N_mm']
 physical.append(current);physical_fields.append((row,np.einsum('ni,ni->n',w,normals),w))
summary['physical_candidates']=physical
if len(physical_fields)==2:summary['two_candidate_surface_pattern_absolute_cosine']=abs(cosine(physical_fields[0][2],physical_fields[1][2]))
(A/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
fig=plt.figure(figsize=(12,9.5));limit=max([float(np.abs(w).max()) for _,w in fields]+[float(np.abs(w).max()) for _,w,_ in physical_fields]);norm=Normalize(-limit,limit);cmap=plt.get_cmap('coolwarm')
states=list(dict.fromkeys(row['state'] for row in rows))
for r in range(2):
 for c,name in enumerate(states):
  ax=fig.add_subplot(2,3,r*3+c+1,projection='3d')
  pair=next(((row,w) for row,w in fields if row['state']==name and row['Ncoarse']==4),None) if r==0 else next(((row,w) for row,w,_ in physical_fields if row['state']==name),None)
  if pair:
   row,w=pair;ax.add_collection3d(Poly3DCollection(xyz[tri],facecolors=cmap(norm(w[tri].mean(axis=1))),edgecolors='none'))
   if r==0:val=row['modes'][0]['curvature_for_1mm_max_surface_mode_N_mm'];label='Bare N4 minimum'
   else:val=row['curvature_for_1mm_max_surface_mode_N_mm'];label='HRZ N8 deformation candidate'
   ax.set_title(f"{label}: {100*row['compression']:.3f}%\nCurvature {val:.4g} N mm",fontsize=10,pad=12)
  else:ax.set_title('10% mass-metric call not requested',fontsize=10)
  ax.set(xlim=(0,10),ylim=(0,10),zlim=(0,10));ax.set_box_aspect((1,1,1));ax.view_init(24,-58);ax.tick_params(labelsize=8)
fig.suptitle('Restricted modes: bare void motion versus deformation-dominated candidate\n1 mm maximum surface normalization; colors show reference-normal component. Not actual deformation or certified full-space modes.',fontsize=11)
fig.subplots_adjust(left=.02,right=.90,bottom=.06,top=.88,wspace=.06,hspace=.30)
cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=fig.add_axes([.93,.20,.012,.52]));cb.set_label('Normal mode component (mm)')
fig.savefig(A/'modes.png',dpi=150);plt.close(fig)
shutil.copy2(A/'modes.png',W/'output/figures/CRITICAL_MODE_modes.png')
(A/'receipt.json').write_text(json.dumps({'wall_seconds':time.perf_counter()-start,'new_jobs':0,'new_design_AD':False,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()},indent=2))
shutil.copy2(W/'work/critical_mode_20261007/analyze.py',D/'analyze.py')
print(json.dumps(summary,indent=2))

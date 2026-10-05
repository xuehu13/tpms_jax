"""One bounded periodic small-edge welding attempt, with topology gates."""
from pathlib import Path
import hashlib,json,shutil,sys,time
import numpy as np
from scipy.spatial import cKDTree
R=Path('/home/xuehu/projects/tpms_jax');sys.path.insert(0,str(R))
from surface_distance import PeriodicSurfaceDistance
E=R/'validation/simulation_error_20261006_r7/step2';O=E/'surface_cleanup';O.mkdir(exist_ok=False)
started=time.perf_counter();tol=.003
with np.load(E/'sharp_surface.npz') as f:p=f['points_mm'];t=f['triangles'];cap=f['is_cap']
canonical=p.copy();high=np.isclose(p,10,atol=1e-10,rtol=0);canonical[high]=0
parent=np.arange(len(p));pairs=cKDTree(canonical).query_pairs(tol,output_type='ndarray')
def root(i):
    while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
    return i
for a,b in pairs:
    a=root(a);b=root(b)
    if a!=b:parent[max(a,b)]=min(a,b)
rep=np.array([root(i) for i in range(len(p))]);roots,inverse=np.unique(rep,return_inverse=True)
centers=np.column_stack([np.bincount(inverse,weights=canonical[:,a])/np.bincount(inverse) for a in range(3)])
for a in range(3):centers[np.unique(inverse[abs(canonical[:,a])<1e-10]),a]=0
# Retain separate translated nodes; welding does not wrap elements across cube.
keys=np.column_stack([inverse,high]);unique,iv=np.unique(keys,axis=0,return_inverse=True)
points=centers[unique[:,0].astype(int)]+10*unique[:,1:]
facets=iv[t];valid=np.all(np.diff(np.sort(facets,axis=1),axis=1)>0,axis=1);facets=facets[valid];cap=cap[valid]
edges=np.sort(facets[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1)
_,multiplicity=np.unique(edges,axis=0,return_counts=True)
xyz=points[facets];norm=np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0])
lengths=np.linalg.norm(xyz[:,[1,2,0]]-xyz[:,[0,1,2]],axis=2)
area=np.linalg.norm(norm,axis=1)/2;alt=2*area/lengths.max(axis=1)
shift=float(np.linalg.norm(points[iv]-p,axis=1).max());paired=[]
for a in range(3):
    other=[i for i in range(3) if i!=a]
    l={tuple(x) for x in np.round(points[abs(points[:,a])<1e-10][:,other],8)}
    h={tuple(x) for x in np.round(points[abs(points[:,a]-10)<1e-10][:,other],8)}
    paired.append(len(l.symmetric_difference(h)))
checks={'closed_manifold':bool(np.all(multiplicity==2)),'nondegenerate_faces':bool(area.min()>1e-14),
        'opposite_node_sets_match':paired==[0,0,0],'bounded_vertex_motion':shift<=.006}
with np.load(R/'validation/thin_target_20261004_r5/gauss_field.npz') as f:v=f['surface_vertices'];tr=f['surface_triangles']
query=PeriodicSurfaceDistance(v,tr);residual=(query.query(xyz[~cap].mean(axis=1)/10)*10-.25)
result={'status':'clean_surface_ready' if all(checks.values()) else 'cleanup_rejected','checks':checks,
        'weld_radius_mm':tol,'maximum_vertex_motion_mm':shift,'points':len(points),'triangles':len(facets),
        'surface_minimum_edge_mm':float(lengths.min()),'surface_minimum_altitude_mm':float(alt.min()),
        'surface_altitude_quantiles_mm':np.quantile(alt,[0,.01,.5,1]).tolist(),
        'boundary_distance_residual_quantiles_mm':np.quantile(residual,[0,.01,.5,.99,1]).tolist(),
        'periodic_unmatched_nodes':paired,'body_seconds':time.perf_counter()-started,
        'input_sha256':hashlib.sha256((E/'sharp_surface.npz').read_bytes()).hexdigest(),
        'scope':'Small-edge geometric repair, not thickness tuning or precision certification; meshing and Jacobian gates remain.'}
if all(checks.values()):np.savez_compressed(O/'sharp_surface.npz',points_mm=points,triangles=facets,is_cap=cap)
shutil.copy2(__file__,O/'experiment.py');(O/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

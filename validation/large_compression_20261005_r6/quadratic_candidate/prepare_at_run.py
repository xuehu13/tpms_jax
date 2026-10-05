"""One targeted HEX27 candidate at unchanged thin-wall physics and node count."""
from pathlib import Path
import sys,json,time,hashlib,shutil
import numpy as np
R=Path('/home/xuehu/projects/tpms_jax');sys.path.insert(0,str(R))
from hyperelastic_fem import make_density_hyperelastic_problem
from surface_distance import PeriodicSurfaceDistance,thickness_occupancy

old=R/'validation/thin_target_20261004_r5'
out=R/'validation/large_compression_20261005_r6/quadratic_candidate'
out.mkdir(exist_ok=False);start=time.perf_counter()
cfg=json.loads((old/'input.json').read_text())
with np.load(old/'gauss_field.npz') as f:
    vertices=f['surface_vertices'];triangles=f['surface_triangles']
surface=PeriodicSurfaceDistance(vertices,triangles)
captured={}
def field(p):
    # Explicit does not assemble tangents; discard these large unused indices.
    del p.I,p.J
    points=np.asarray(p.physical_quad_points);weights=np.asarray(p.fe.JxW)
    distance=surface.query(points)
    rho=np.asarray(thickness_occupancy(distance,cfg['thickness_over_L'],
                   cfg['interface_10_90_over_L']/(2*np.log(9))))
    captured.update(physical_quad_points=points,JxW=weights,distance=distance,rho=rho)
    return rho
p=make_density_hyperelastic_problem(32,rho_quad=field,eta=cfg['eta'],
                                   periodic_axes=(0,1,2),element_degree=2)
points=np.asarray(p.fe.points)
H=np.array([[.01,.02,.03],[-.01,.02,0],[0,-.02,-.01]])
grad=np.asarray(p.fe.sol_to_grad(points@H.T))
error=float(np.max(np.abs(grad-H)))
partition=float(np.max(np.abs(np.asarray(p.fe.shape_vals).sum(axis=1)-1)))
vf=float(np.sum(captured['rho']*captured['JxW']))
oldvf=.15164916224866734
passed=(error<1e-12 and partition<1e-12 and len(points)==65**3
        and np.max(p.class_ids)+1==64**3 and abs(captured['JxW'].sum()-1)<1e-12
        and abs(vf/oldvf-1)<.01)
np.savez_compressed(out/'gauss_field.npz',**captured)
result={'status':'geometry_basis_ready' if passed else 'preflight_failed',
        'element':'HEX27','cells_per_axis':32,'node_grid_intervals':64,
        'nodes':len(points),'elements':p.fe.num_cells,'Gauss_points_per_cell':p.fe.num_quads,
        'total_Gauss_points':captured['rho'].size,'quadrature_order':4,
        'thickness_mm':cfg['thickness_mm'],'cell_size_mm':cfg['cell_size_mm'],
        'interface_10_90_mm':cfg['interface_10_90_over_L']*cfg['cell_size_mm'],
        'eta':cfg['eta'],'projected_Vf':vf,'old_HEX8_Vf':oldvf,
        'relative_Vf_change':vf/oldvf-1,'affine_gradient_max_error':error,
        'shape_partition_max_error':partition,'volume':float(captured['JxW'].sum()),
        'seconds':time.perf_counter()-start,
        'method_scope':'quadratic background representation candidate; no solve or precision certification',
        'unchanged_physics':True,'old_cache_sha256':hashlib.sha256((old/'gauss_field.npz').read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256((R/'hyperelastic_fem.py').read_bytes()).hexdigest()}
(out/'preparation.json').write_text(json.dumps(result,indent=2)+'\n')
shutil.copy2(__file__,out/'prepare_at_run.py')
print(json.dumps(result,indent=2),flush=True)
if not passed:raise RuntimeError('Quadratic preflight failed; do not submit target')

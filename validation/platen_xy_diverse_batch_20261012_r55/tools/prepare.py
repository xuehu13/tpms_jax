"""New geometry cache at actual HEX27 Gauss points; reuse r52 input cards."""
from pathlib import Path
import argparse, hashlib,json,os,sys,time
os.environ['JAX_PLATFORMS']='cpu'
B=Path(__file__).resolve().parents[1];R=B.parents[1];sys.path.insert(0,str(R))
import numpy as np
from scipy.special import expit
from scipy.spatial import cKDTree
from surface_distance import PeriodicSurfaceDistance
from hyperelastic_fem import make_density_hyperelastic_problem
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
ap=argparse.ArgumentParser();ap.add_argument('kind');args=ap.parse_args();kind=args.kind
protocol=json.loads((B/'protocol.json').read_text());assert kind in protocol['cases'];E=B/kind
audit=json.loads((B/'checks/geometry_precheck.json').read_text());d=B/'sources'/kind
assert not (d/'gauss_field.npz').exists() and sha(d/'surface.npz')==audit[kind]['source_sha256']['surface.npz']
for name,h in audit[kind]['original_readonly_source_sha256'].items():
 p=Path('/mnt/'+name[0].lower()+name[2:].replace('\\','/'));assert sha(p)==h
with np.load(d/'surface.npz') as z:v=z['surface_vertices'];tri=z['surface_triangles']
start=time.time();distance=PeriodicSurfaceDistance(v,tri);ell=.05/10/(2*np.log(9));collected={}
def field(problem):
 X=np.asarray(problem.physical_quad_points);w=np.asarray(problem.fe.JxW);dist=distance.query(X);phi=expit((.025-dist)/ell)
 collected.update(physical_quad_points=X,JxW=w,distance=dist,rho=phi);return phi
p=make_density_hyperelastic_problem(32,rho_quad=field,eta=1e-4,periodic_axes=(0,1),element_degree=2,quadrature_order=4,material_model='objective_void')
np.testing.assert_array_equal(np.asarray(p.rho),collected['rho'])
X=collected['physical_quad_points'];w=collected['JxW'];phi=collected['rho']
assert X.shape==(32**3,27,3) and phi.shape==w.shape==(32**3,27) and np.isfinite(phi).all() and w.min()>0
np.savez_compressed(d/'gauss_field.npz',**collected)
cross=np.cross(v[tri[:,1]]-v[tri[:,0]],v[tri[:,2]]-v[tri[:,0]]);area=np.linalg.norm(cross,axis=1)/2
centers=v[tri].mean(1);cellcenters=np.asarray(p.fe.points)[np.asarray(p.fe.cells)].mean(1)
midmax=phi[cKDTree(cellcenters).query(centers)[1]].max(1)
rng=np.random.default_rng(20261012);q=rng.random((256,3));periodic=[float(np.max(abs(distance.query(q)-distance.query(q+np.eye(3)[k])))) for k in range(3)]
assert max(periodic)<1e-10
vf=float(np.sum(phi*w)/w.sum());shellvf=float(area.sum()*.05)
write(E/'checks/gauss_preparation.json',{'kind':kind,'actual_own_problem_Gauss_equality':True,'Gauss_points':phi.size,
 'weighted_occupancy_volume_fraction':vf,'shell_area_times_t_volume_fraction':shellvf,'volume_relative_difference':vf/shellvf-1,
 'no_volume_or_stiffness_fitting':True,'periodic_distance_errors':periodic,'midfacet_cells_max_phi_min':float(midmax.min()),
 'midfacet_cells_with_max_phi_below_half':int(np.sum(midmax<.5)),
 'coverage_scope':'Cell/Gauss sampling diagnostic; not global thickness, contact or mechanical-connectivity certification',
 'geometry_cache_seconds':time.time()-start})
audit[kind]['source_sha256']['gauss_field.npz']=sha(d/'gauss_field.npz');write(B/'checks/geometry_precheck.json',audit)
source=R/'validation/platen_xy_PD_20261010_r52/tools/prepare_case.py';original=source.read_text();code=original
replacements=[("choices=['P','D']",'choices='+repr(protocol['cases'])),
 ("old=R/'validation/canonical_tpms_20261009_r50'/a.kind","old=B/'sources'/a.kind"),('r52','r55')]
for old,new in replacements:assert old in code;code=code.replace(old,new)
write(E/'checks/prepare_recipe.json',{'source':str(source),'source_sha256':sha(source),'substitutions':replacements,'adapted_sha256':hashlib.sha256(code.encode()).hexdigest()})
exec(compile(code,str(source),'exec'),{'__name__':'__main__','__file__':str(Path(__file__).resolve())})

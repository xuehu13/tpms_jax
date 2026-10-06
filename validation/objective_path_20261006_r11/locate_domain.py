"""Locate dense-probe NH-invalid points in one saved state; no new solve."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import basix
from scipy.special import expit
R=Path('/home/xuehu/projects/tpms_jax')
O=R/'validation/objective_path_20261006_r11'
sys.path.insert(0,str(R))
from jax_fem.basis import get_elements
from surface_distance import PeriodicSurfaceDistance
G=R/'validation/thin_target_20261004_r5/gauss_field.npz'
Q=R/'validation/large_compression_20261005_r6/quadratic_candidate/gauss_field.npz'
with np.load(O/'T0p004/field.npz') as f:q=f['q']
with np.load(G) as f:distance=PeriodicSurfaceDistance(f['surface_vertices'],f['surface_triangles'])
with np.load(Q) as f:rho27=f['rho']
family,cell,_,_,degree,order=get_elements('HEX27')
el=basix.create_element(family,cell,degree);local=np.rint(2*el.points[order]).astype(int)
xi,_=basix.make_quadrature(cell,8)
tab=el.tabulate(1,xi).take(order,axis=2)
shape=tab[0,:,:,0];grad=tab[1:4,:,:,0].transpose(1,2,0)*32
xi27,_=basix.make_quadrature(cell,4)
g27=el.tabulate(1,xi27).take(order,axis=2)[1:4,:,:,0].transpose(1,2,0)*32
bad=[]
for begin in range(0,32**3,512):
    indices=np.arange(begin,min(begin+512,32**3))
    origins=np.stack([indices//32**2,(indices//32)%32,indices%32],axis=1)
    nodes=2*origins[:,None,:]+local[None,:,:];grid=nodes%64
    ids=(grid[...,0]*64+grid[...,1])*64+grid[...,2]
    F=np.diag([1.,1.,.8])+np.einsum('cni,qnj->cqij',q[ids],grad)
    J=np.linalg.det(F);ci,qi=np.where(J<=0)
    if not len(ci):continue
    points=np.einsum('qn,cnd->cqd',shape,nodes/64)[ci,qi]
    d=distance.query(points);phi=expit((.025-d)/(.05/(2*np.log(9))/10))
    for k in np.where(phi>.001)[0]:
        c,i=int(ci[k]),int(qi[k]);x=np.clip((phi[k]-.001)/.009,0,1)
        j27=np.linalg.det(np.diag([1.,1.,.8])+np.einsum('ni,qnj->qij',q[ids[c]],g27))
        active27=rho27[indices[c]]>.001
        bad.append({'cell_id':int(indices[c]),'dense_local_point':i,'reference_xyz_mm':(points[k]*10).tolist(),
            'actual_J':float(J[c,i]),'phi':float(phi[k]),'NH_weight':float(x**3*(10-15*x+6*x*x)),
            'stiffness_scale':float(1e-4+(1-1e-4)*phi[k]),'distance_from_midsurface_mm':float(d[k]*10),
            'distance_outside_nominal_wall_mm':float(d[k]*10-.25),
            'same_cell_27point_J_min':float(j27.min()),
            'same_cell_27point_NH_active_J_min':float(j27[active27].min()) if active27.any() else None})
assert len(bad)==6,len(bad)
report={'scope':'Locations only in current 20% saved field; no added force, path or material change',
    'points':bad,'cell_count':len({p['cell_id'] for p in bad}),
    'phi_range':[min(p['phi'] for p in bad),max(p['phi'] for p in bad)],
    'NH_weight_range':[min(p['NH_weight'] for p in bad),max(p['NH_weight'] for p in bad)],
    'outside_nominal_wall_mm_range':[min(p['distance_outside_nominal_wall_mm'] for p in bad),max(p['distance_outside_nominal_wall_mm'] for p in bad)],
    'saved_field_sha256':hashlib.sha256((O/'T0p004/field.npz').read_bytes()).hexdigest()}
target=O/'domain_probe/invalid_points.json';assert not target.exists()
target.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

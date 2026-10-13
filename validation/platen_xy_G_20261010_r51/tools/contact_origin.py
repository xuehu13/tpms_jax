"""r51-only contact-support diagnostic; no production mechanics are changed."""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.spatial import cKDTree
D=Path(__file__).resolve().parents[1]
B=D/'diagnostics/contact_origin_20261010'
R=D.parents[1]
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def shapes_at(points,n):
    nn=2*n+1
    cell=np.minimum(np.floor(points[:,:2]*n).astype(int),n-1)
    local=points[:,:2]*n-cell
    shape=lambda x:np.stack([2*(x-.5)*(x-1),4*x*(1-x),2*x*(x-.5)],axis=-1)
    sh=np.einsum('ni,nj->nij',shape(local[:,0]),shape(local[:,1])).reshape(-1,9)
    offsets=np.indices((3,3)).reshape(2,-1).T
    ij=2*cell[:,None,:]+offsets
    ids=(ij[:,:,0]*nn+ij[:,:,1])*nn+(np.rint(points[:,2]*2*n).astype(int))[:,None]
    np.testing.assert_allclose(sh.sum(axis=1),1,atol=1e-14)
    grid=np.indices((nn,nn,nn)).reshape(3,-1).T/(2*n)
    np.testing.assert_allclose(np.einsum('cn,cnj->cj',sh,grid[ids]),points,atol=1e-14)
    return ids,sh
def prepare():
    cfg=json.loads((D/'input/protocol.json').read_text())
    with np.load(D/'input/cut_face_contact.npz') as z:cap=dict(z)
    with np.load(R/'validation/canonical_tpms_20261009_r50/G/surface_shell_quality.npz') as z:v=z['surface_vertices']
    points=[];weights=[];sides=[];labels=[];assignment=np.zeros(len(cap['points']),int)
    distance=np.zeros(len(assignment))
    for side in (0,1):
        mask=cap['side']==side;rim=np.where(abs(v[:,2]-side)<1e-10)[0]
        # Use actual existing rim nodes, including periodic edge representatives.
        # XY field periodicity makes paired edge representatives equivalent.
        dist,idx=cKDTree(v[rim,:2]).query(cap['points'][mask,:2])
        w=np.bincount(idx,weights=cap['area_weight'][mask],minlength=len(rim))
        offset=len(points);assignment[mask]=offset+idx;distance[mask]=dist
        points.extend(v[rim]);weights.extend(w);sides.extend([side]*len(rim));labels.extend((rim+1).tolist())
    points=np.asarray(points);ids,sh=shapes_at(points,cfg['N'])
    cache=dict(node_ids=ids,points=points,shape=sh,area_weight=np.asarray(weights),side=np.asarray(sides),phi=np.ones(len(points)),shell_node_labels=np.asarray(labels))
    # Occupancy is retained in the aggregated weights; phi=1 here identifies
    # midsurface points for penetration reporting, not a new material field.
    dest=B/'input/rimline_contact.npz';np.savez_compressed(dest,**cache)
    for side in (0,1):np.testing.assert_allclose(np.sum(cache['area_weight'][cache['side']==side]),np.sum(cap['area_weight'][cap['side']==side]),rtol=1e-13)
    np.savez_compressed(B/'input/contact_assignment.npz',rim_assignment=assignment,distance_normalized=distance)
    line={**cfg,'case_id':'G_platen_xy_r51_rimline_diagnostic','contact_cache_path':str(dest),
          'stage':'Artificial midsurface-rim contact-support diagnostic, 2pct; not physical finite-thickness contact certification',
          'jax_contact':'Occupied cap weights aggregated at nearest existing shell rim node; Q2 interpolation and force distribution at that point; reference area and kn conserved',
          'contact_cache_sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
    write(B/'input/rimline_protocol.json',line)
    slow={**cfg,'case_id':'G_platen_xy_r51_T2ms_diagnostic','load_time_s':.002,
          'stage':'2pct cut-end rate diagnostic: T2ms, original .2ms hold and original contact definitions'}
    write(B/'input/slow_protocol.json',slow)
    text=(D/'abaqus/platen_2pct.inp').read_text()
    before='0.,0.,0.001,1.,0.0012000000000000001,1.'
    assert text.count(before)==1
    text=text.replace(before,'0.,0.,0.002,1.,0.0022,1.')
    assert text.count(', 0.0012000000000000001')==1
    text=text.replace(', 0.0012000000000000001',', 0.0022')
    (B/'abaqus/platen_T2ms.inp').write_text(text)
    write(B/'input/preparation.json',{'rim_nodes':len(points),'cap_points':len(cap['points']),
          'area_preserved_mm2':[float(np.sum(cache['area_weight'][cache['side']==side])*100) for side in (0,1)],
          'area_weighted_distance_to_rim_mm':float(np.sum(cap['area_weight']*distance)/np.sum(cap['area_weight'])*10),
          'max_shape_partition_error':float(np.max(abs(sh.sum(axis=1)-1))),
          'warning':'Changed contact support/discretization is a diagnostic, not a correction certified as physical.'})
    print(json.dumps(json.loads((B/'input/preparation.json').read_text()),indent=2))
if __name__=='__main__':prepare()

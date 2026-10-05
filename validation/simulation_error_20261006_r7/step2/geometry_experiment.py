"""One periodic clipped distance-band surface for plan step 2.

Marching tetrahedra on a periodic sample of the *original triangle distance*.
This is a geometric approximation, not a binary background mechanics model.
The six planar caps use the same tetrahedral face diagonals on opposite sides.
"""
from pathlib import Path
import hashlib, json, shutil, sys, time
import numpy as np
R=Path('/home/xuehu/projects/tpms_jax');sys.path.insert(0,str(R))
from surface_distance import PeriodicSurfaceDistance
E=R/'validation/simulation_error_20261006_r7/step2'
N=80;LEVEL=.025
G=R/'validation/thin_target_20261004_r5/gauss_field.npz'

def main():
    started=time.perf_counter();E.mkdir(parents=True,exist_ok=False)
    with np.load(G) as f:v=f['surface_vertices'];tri=f['surface_triangles']
    query=PeriodicSurfaceDistance(v,tri)
    axis=np.arange(N+1)/N;grid=np.indices((N+1,)*3).reshape(3,-1).T/N
    d=query.query(grid);print('Distance sample complete',flush=True)
    # Force copied periodic samples bit-identical (distance query wraps corners).
    dg=d.reshape((N+1,)*3)
    for a in range(3):
        low=[slice(None)]*3;high=low.copy();low[a]=0;high[a]=N
        dg[tuple(high)]=dg[tuple(low)]
    d=dg.ravel();count=len(grid);stride=N+1
    corner=np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[1,0,1],[0,1,1],[1,1,1]])
    shifts=(corner[:,0]*stride+corner[:,1])*stride+corner[:,2]
    # Binary corner indexing differs from flattening coordinates; explicitly list.
    tetra=np.array([[0,1,3,7],[0,3,2,7],[0,2,6,7],[0,6,4,7],[0,4,5,7],[0,5,1,7]])
    raw=[]
    def edgekey(a,b):return np.minimum(a,b)*count+np.maximum(a,b)
    def emit(t,sign):
        inside=t[sign];outside=t[~sign]
        if len(inside)==1:
            row=edgekey(inside[0],outside)[None,:]
        elif len(inside)==3:
            row=edgekey(outside[0],inside)[None,:]
        else:
            a,b=inside;c,e=outside
            ac,ae,bc,be=edgekey(a,c),edgekey(a,e),edgekey(b,c),edgekey(b,e)
            row=np.array([[ac,ae,be],[ac,be,bc]])
        return row
    # Vectorize each tetrahedron sign case, rather than Python per tetrahedron.
    origins=np.indices((N,)*3).reshape(3,-1).T
    base=(origins[:,0]*stride+origins[:,1])*stride+origins[:,2]
    for local in tetra:
        t=base[:,None]+shifts[local];inside=d[t]<=LEVEL
        code=np.sum(inside*(1<<np.arange(4)),axis=1)
        for c in range(1,15):
            selected=t[code==c];bits=((c>>np.arange(4))&1).astype(bool)
            i=selected[:,bits];o=selected[:,~bits]
            if i.shape[1]==1: faces=edgekey(i,o)[:,None,:]
            elif i.shape[1]==3: faces=edgekey(o,i)[:,None,:]
            else:
                ac=edgekey(i[:,0],o[:,0]);ae=edgekey(i[:,0],o[:,1]);bc=edgekey(i[:,1],o[:,0]);be=edgekey(i[:,1],o[:,1])
                faces=np.stack([np.stack([ac,ae,be],axis=1),np.stack([ac,be,bc],axis=1)],axis=1)
            # Orientation is fixed later by inside-to-outside direction.
            direction=grid[o].mean(axis=1)-grid[i].mean(axis=1)
            raw.append((faces.reshape(-1,3),np.repeat(direction,faces.shape[1],axis=0),False))
    caps=[]
    for a in range(3):
        other=[i for i in range(3) if i!=a]
        ij=np.indices((N,N)).reshape(2,-1).T
        for side in [0,N]:
            p=np.zeros((len(ij),3),dtype=int);p[:,a]=side;p[:,other]=ij
            b=(p[:,0]*stride+p[:,1])*stride+p[:,2]
            s1=stride**(2-other[0]);s2=stride**(2-other[1])
            # Same low-low to high-high diagonal as the Freudenthal tetrahedra.
            for off in [[0,s1,s1+s2],[0,s1+s2,s2]]:
                t=b[:,None]+off;bits=d[t]<=LEVEL;codes=np.sum(bits*(1<<np.arange(3)),axis=1)
                for c in range(1,8):
                    rows=t[codes==c];mask=((c>>np.arange(3))&1).astype(bool)
                    if len(rows)==0:continue
                    if c==7:faces=edgekey(rows,rows)
                    elif mask.sum()==1:
                        i=rows[:,mask];o=rows[:,~mask]
                        faces=np.column_stack([edgekey(i[:,0],i[:,0]),edgekey(i[:,0],o[:,0]),edgekey(i[:,0],o[:,1])])
                    else:
                        i=rows[:,mask];o=rows[:,~mask][:,0]
                        v0=edgekey(i[:,0],i[:,0]);v1=edgekey(i[:,1],i[:,1]);a0=edgekey(i[:,0],o);a1=edgekey(i[:,1],o)
                        faces=np.vstack([np.column_stack([v0,v1,a1]),np.column_stack([v0,a1,a0])])
                    direction=np.zeros_like(faces,dtype=float);direction[:,a]=-1 if side==0 else 1
                    raw.append((faces,direction,True))
    faces=np.vstack([z[0] for z in raw]);direction=np.vstack([z[1] for z in raw]);cap=np.concatenate([np.full(len(z[0]),z[2]) for z in raw])
    unique,inv=np.unique(faces,return_inverse=True);facets=inv.reshape(-1,3)
    lo=unique//count;hi=unique%count;alpha=np.zeros(len(unique));m=lo!=hi
    alpha[m]=(LEVEL-d[lo[m]])/(d[hi[m]]-d[lo[m]])
    points=grid[lo]+alpha[:,None]*(grid[hi]-grid[lo])
    xyz=points[facets];normal=np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0])
    flip=np.sum(normal*direction,axis=1)<0;facets[flip]=facets[flip][:,[0,2,1]]
    edges=np.sort(facets[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1)
    _,multiplicity=np.unique(edges,axis=0,return_counts=True)
    assert np.all(multiplicity==2),'Closed surface edge manifold failed'
    volume=float(np.sum(np.einsum('ij,ij->i',points[facets[:,0]],np.cross(points[facets[:,1]],points[facets[:,2]])))/6)
    assert volume>0
    # Pair cap geometry by canonical translations. This is not yet a volume mesh PBC.
    mismatch=[]
    for a in range(3):
        ids=np.unique(facets[cap]);low=ids[abs(points[ids,a])<1e-12];high=ids[abs(points[ids,a]-1)<1e-12]
        other=[i for i in range(3) if i!=a]
        l={tuple(x) for x in np.round(points[low][:,other],10)};h={tuple(x) for x in np.round(points[high][:,other],10)}
        mismatch.append(len(l.symmetric_difference(h)));assert l==h
    # Distance residual checks away from planar caps; facets represent linear interpolation.
    boundary=facets[~cap];centers=points[boundary].mean(axis=1)
    residual=(query.query(centers)-LEVEL)*10
    np.savez_compressed(E/'sharp_surface.npz',points_mm=points*10,triangles=facets,is_cap=cap)
    import meshio
    meshio.write(E/'sharp_surface.stl',meshio.Mesh(points*10,[('triangle',facets)]),binary=True)
    shutil.copy2(__file__,E/'geometry_experiment.py')
    result={'status':'closed_periodic_surface_ready_not_volume_reference','sampling_intervals':N,
            'points':len(points),'triangles':len(facets),'closed_edge_multiplicity':2,
            'periodic_cap_unmatched_nodes_by_axis':mismatch,'volume_mm3':volume*1000,
            'distance_band_boundary_centroid_error_mm_quantiles':np.quantile(residual,[0,.01,.5,.99,1]).tolist(),
            'thickness_target_mm':.5,'input_sha256':hashlib.sha256(G.read_bytes()).hexdigest(),
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'body_seconds':time.perf_counter()-started,
            'scope':'Clipped sharp distance band with sampled linear surface approximation. No material solve; quadratic volume mesh and quality gates outstanding.'}
    (E/'geometry.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()

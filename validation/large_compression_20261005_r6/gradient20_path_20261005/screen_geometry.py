"""Saved-state nonlocal midsurface proximity screen, not a contact law/certificate.

Periodic seam neighbors are excluded using five graph rings. Exact midsurface
triangle distances are compared with mapped half-thickness envelopes; this is
only a warning screen, not exact offsets or exhaustive continuous contact.
"""
from pathlib import Path
import argparse,json,time,os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import numpy as np
import basix
from scipy.spatial import cKDTree
from jax_fem.basis import get_elements
R=Path('/home/xuehu/projects/tpms_jax');root=R/'validation/large_compression_20261005_r6';out=root/'gradient20_path_20261005'
p=argparse.ArgumentParser();p.add_argument('name',choices=['center','minus','plus']);a=p.parse_args();started=time.perf_counter()
folder=root/'quadratic_candidate/T0p004_compact' if a.name=='center' else out/('adaptive_'+a.name)
with np.load(folder/'field.npz') as f:q=f['q']
t=json.loads((folder/'input.json').read_text())['thickness_mm']
with np.load(R/'validation/thin_target_20261004_r5/gauss_field.npz') as f:v=f['surface_vertices'];tri=f['surface_triangles']
family,cell,_,_,degree,order=get_elements('HEX27');element=basix.create_element(family,cell,degree)
local=np.rint(2*element.points[order]).astype(int)
def deform(x):
 wrapped=x%1;cellid=np.floor(wrapped*32).astype(int);xi=wrapped*32-cellid
 sh=element.tabulate(0,xi)[0,:,order,0]
 if sh.shape!=(len(x),27):sh=sh.T
 grid=(2*cellid[:,None,:]+local[None,:,:])%64;ids=(grid[...,0]*64+grid[...,1])*64+grid[...,2]
 return (x*np.array([1,1,.8])+np.einsum('pn,pni->pi',sh,q[ids]))*10
x=deform(v);T=x[tri];centers=T.mean(axis=1);radius=np.max(np.linalg.norm(T-centers[:,None,:],axis=2),axis=1)
initial_cross=np.cross(v[tri[:,1]]-v[tri[:,0]],v[tri[:,2]]-v[tri[:,0]])
node_normal=np.zeros(v.shape)
for col in range(3):np.add.at(node_normal,tri[:,col],initial_cross)
normal_lengths=np.linalg.norm(node_normal,axis=1)
if np.min(normal_lengths)<=1e-15:raise ValueError('Surface normal is undefined; cannot screen thickness envelopes')
node_normal/=normal_lengths[:,None]
face_cross=np.cross(T[:,1]-T[:,0],T[:,2]-T[:,0]);face_normal=face_cross/np.linalg.norm(face_cross,axis=1)[:,None]
xplus=deform(v+node_normal*t/20);xminus=deform(v-node_normal*t/20)
halfwidth=np.max(np.abs(np.einsum('fni,fi->fn',(xplus-xminus)[tri]/2,face_normal)),axis=1)
box=np.array([10.,10.,8.]);wrapped=centers%box
tree=cKDTree(wrapped,boxsize=box);threshold=float(2*halfwidth.max())
pairs=tree.query_pairs(threshold+2*radius.max(),output_type='ndarray')
delta=centers[pairs[:,0]]-centers[pairs[:,1]];delta-=box*np.rint(delta/box)
bound=np.linalg.norm(delta,axis=1)-radius[pairs[:,0]]-radius[pairs[:,1]]
pairs=pairs[bound<=halfwidth[pairs[:,0]]+halfwidth[pairs[:,1]]]
# Quantized periodic vertices join the seam graph before local exclusion.
periodic=np.round((v%1)*1e9).astype(np.int64);periodic[periodic==int(1e9)]=0
_,vertexclass=np.unique(periodic,axis=0,return_inverse=True)
vertex_faces=[set() for _ in range(vertexclass.max()+1)]
for i,face in enumerate(vertexclass[tri]):
 for node in face:vertex_faces[node].add(i)
neighbors=[set().union(*(vertex_faces[node] for node in face)) for face in vertexclass[tri]]
local_cache={}
def local_faces(i):
 if i not in local_cache:
  seen={i};front={i}
  for _ in range(5):
   front=set().union(*(neighbors[k] for k in front))-seen;seen|=front
  local_cache[i]=seen
 return local_cache[i]
nonlocal_pairs=np.array([(i,j) for i,j in pairs if j not in local_faces(int(i))],dtype=np.int64).reshape((-1,2))
def point_triangle(points,triangle):
 A,B,C=triangle;ab=B-A;ac=C-A;n=np.cross(ab,ac);n2=n@n
 signed=(points-A)@n;projection=points-signed[:,None]*n/n2;ap=projection-A
 d00=ab@ab;d01=ab@ac;d11=ac@ac;den=d00*d11-d01*d01
 vv=((ap@ab)*d11-(ap@ac)*d01)/den;ww=((ap@ac)*d00-(ap@ab)*d01)/den
 inside=(vv>=-1e-12)&(ww>=-1e-12)&(vv+ww<=1+1e-12)
 best=np.where(inside,signed**2/n2,np.inf)
 for aa,bb in [(A,B),(B,C),(C,A)]:
  edge=bb-aa;s=np.clip((points-aa)@edge/(edge@edge),0,1)
  best=np.minimum(best,np.sum((points-aa-s[:,None]*edge)**2,axis=1))
 return float(np.min(best))
def segment_face_crossing(triangle,other):
 A,B,C=other;n=np.cross(B-A,C-A);signed=(triangle-A)@n
 for k in range(3):
  l=(k+1)%3;den=signed[k]-signed[l]
  if abs(den)>1e-16 and signed[k]*signed[l]<=0:
   x=triangle[k]+signed[k]/den*(triangle[l]-triangle[k])
   if point_triangle(x[None,:],other)<1e-20:return True
 return False
def triangle_distance(A,B):
 if segment_face_crossing(A,B) or segment_face_crossing(B,A):return 0.
 d2=min(point_triangle(A,B),point_triangle(B,A))
 for k in range(3):
  aa=A[k];u=A[(k+1)%3]-aa
  for l in range(3):
   bb=B[l];vv=B[(l+1)%3]-bb;w=aa-bb
   uu=u@u;uv=u@vv;v2=vv@vv;uw=u@w;vw=vv@w;den=uu*v2-uv*uv
   if den>1e-14*uu*v2:
    ss=(uv*vw-v2*uw)/den;tt=(uu*vw-uv*uw)/den
    if 0<=ss<=1 and 0<=tt<=1:d2=min(d2,float(np.sum((w+ss*u-tt*vv)**2)))
 return np.sqrt(max(0.,d2))
near=[];minimum=np.inf;minpair=None
for i,j in nonlocal_pairs:
 shift=box*np.rint((centers[i]-centers[j])/box)
 distance=triangle_distance(T[i],T[j]+shift);gap=distance-halfwidth[i]-halfwidth[j]
 if distance<minimum:minimum=distance;minpair=[int(i),int(j)]
 if gap<=0:near.append({'triangle_ids':[int(i),int(j)],'midsurface_distance_mm':float(distance),
                      'half_thickness_envelope_gap_mm':float(gap),'periodic_shift_mm':shift.tolist()})
result={'state':a.name,'macro_compression':.2,'thickness_mm':t,'graph_rings_excluded':5,
 'broad_candidate_pairs':len(pairs),'nonlocal_exact_pairs':len(nonlocal_pairs),
 'minimum_distance_in_candidates_mm':float(minimum) if np.isfinite(minimum) else None,
 'minimum_pair':minpair,'envelope_overlap_candidates':len(near),'closest_candidates':sorted(near,key=lambda r:r['half_thickness_envelope_gap_mm'])[:20],
 'periodic_images_included':True,'contact_certified':False,
 'scope':'Five-ring excluded midsurface triangle proximity and approximate mapped normal halfwidth; no exact contact/offset or path-wide certification',
 'elapsed_seconds':time.perf_counter()-started}
(out/('geometry_screen_'+a.name+'.json')).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))

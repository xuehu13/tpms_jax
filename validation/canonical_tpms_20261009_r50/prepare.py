"""Canonical G/P/D input adapter. Reuses the existing material and explicit entry."""
from pathlib import Path
import sys, json, hashlib, shutil, time
import numpy as np
from scipy.optimize import brentq
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

R=Path('/home/xuehu/projects/tpms_jax')
D=R/'validation/canonical_tpms_20261009_r50'
sys.path[:0]=[str(D/'_geometry_deps'),str(R)]
from skimage.measure import marching_cubes

def write(p,v): p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def clean_mesh(v,f):
    _,first,inv=np.unique(np.round(v,11),axis=0,return_index=True,return_inverse=True)
    v=v[first];f=inv[f]
    keep=(f[:,0]!=f[:,1])&(f[:,0]!=f[:,2])&(f[:,1]!=f[:,2]);f=f[keep]
    _,first=np.unique(np.sort(f,axis=1),axis=0,return_index=True);f=f[np.sort(first)]
    used,inv=np.unique(f,return_inverse=True)
    return v[used],inv.reshape(-1,3)

def collapse_short_periodic_edges(v,f,tolerance=.001):
    """Remove marching-grid slivers, synchronizing all periodic copies.

    Representatives stay on analytic surface vertices. Boundary vertices take
    precedence so the one-cell cut remains on its original six planes.
    This is a bounded geometry mesh operation, not mechanical stabilization.
    """
    v,f=clean_mesh(v,f);old_vertices=len(v);max_move=0.
    for _ in range(3):
        wrapped=np.where(np.isclose(v,1,atol=1e-10),0,v)
        _,first,qids=np.unique(np.round(wrapped,9),axis=0,return_index=True,return_inverse=True)
        q=wrapped[first];parent=np.arange(len(q))
        priority=np.sum(np.isclose(q,0,atol=1e-10),axis=1)
        def root(a):
            while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
            return a
        edges=np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]])
        short=edges[np.linalg.norm(v[edges[:,0]]-v[edges[:,1]],axis=1)<tolerance]
        if not len(short):break
        for aa,bb in short:
            a,b=root(int(qids[aa])),root(int(qids[bb]))
            if a==b:continue
            if (priority[a],-a)<(priority[b],-b):a,b=b,a
            parent[b]=a
        representative=np.array([root(int(a)) for a in qids])
        new=q[representative]+np.rint(v-q[representative])
        max_move=max(max_move,float(np.max(np.linalg.norm(new-v,axis=1))))
        v,f=clean_mesh(new,f)
    return v,f,{'short_edge_threshold_mm':tolerance*10,'removed_vertices':old_vertices-len(v),
        'max_single_pass_vertex_move_mm':max_move*10,'passes_at_most':3}

def implicit(kind,v):
    x,y,z=np.moveaxis(2*np.pi*np.asarray(v),-1,0)
    if kind=='G': return np.sin(x)*np.cos(y)+np.sin(y)*np.cos(z)+np.sin(z)*np.cos(x)
    if kind=='P': return np.cos(x)+np.cos(y)+np.cos(z)
    if kind=='D': return np.cos(x)*np.cos(y)*np.cos(z)-np.sin(x)*np.sin(y)*np.sin(z)
    raise ValueError(kind)

def make_surface(kind):
    n=64; grid=np.stack(np.meshgrid(*([np.linspace(0,1,n+1)]*3),indexing='ij'),axis=-1)
    values=implicit(kind,grid);values[abs(values)<1e-13]=0
    for axis in range(3):
        index=[slice(None)]*3;index[axis]=-1
        values[tuple(index)]=np.take(values,0,axis=axis)
    v,f,_,_=marching_cubes(values,level=0,spacing=(1/n,)*3,allow_degenerate=False)
    v=v.astype(float)
    # Recover the analytic edge root in float64; both periodic faces use the
    # identical coordinate/root rule. This is geometry preprocessing, not AD.
    interior_vertices=0
    for p in v:
        rounded=np.rint(p*n)/n;on_grid=abs(p-rounded)<2e-7
        p[on_grid]=rounded[on_grid]
        fractional=np.flatnonzero(~on_grid)
        if len(fractional)==1:
            a=int(fractional[0]);lo=np.floor(p[a]*n)/n;hi=lo+1/n
            def value(x):
                q=p.copy();q[a]=x;return float(implicit(kind,q))
            vl,vh=value(lo),value(hi)
            if abs(vl)<1e-12:p[a]=lo
            elif abs(vh)<1e-12:p[a]=hi
            elif vl*vh<0:p[a]=brentq(value,lo,hi,xtol=5e-15)
            else: raise ValueError('Unbracketed marching-cube edge')
        elif len(fractional)>1:
            interior_vertices+=1
    v,f,collapse=collapse_short_periodic_edges(v,f)
    cross=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])
    area=np.linalg.norm(cross,axis=1)/2
    if np.min(area)<=1e-15:raise ValueError('Degenerate surface triangles remain')
    _,qids=np.unique(np.round(np.where(np.isclose(v,1,atol=1e-10),0,v),9),axis=0,return_inverse=True)
    quotient=qids[f];edges=np.vstack([quotient[:,[0,1]],quotient[:,[1,2]],quotient[:,[2,0]]])
    uniq,which,counts=np.unique(np.sort(edges,axis=1),axis=0,return_inverse=True,return_counts=True)
    orient=np.bincount(which,weights=np.where(edges[:,0]<edges[:,1],1,-1))
    graph=coo_matrix((np.ones(2*len(uniq)),(np.r_[uniq[:,0],uniq[:,1]],np.r_[uniq[:,1],uniq[:,0]])))
    components=int(connected_components(graph,directed=False,return_labels=False))
    raw_edges=np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]])
    raw_unique,raw_counts=np.unique(np.sort(raw_edges,axis=1),axis=0,return_counts=True)
    trace=v[np.unique(raw_unique[raw_counts==1])]
    seams=[]
    for axis in range(3):
        low=trace[np.isclose(trace[:,axis],0,rtol=0,atol=1e-10)].copy();high=trace[np.isclose(trace[:,axis],1,rtol=0,atol=1e-10)].copy()
        # The corner (0,0,0) may touch the surface without incident facets in
        # this positive octant. Compare face traces on their transverse torus;
        # the independent quotient-edge audit checks actual seam connectivity.
        low=np.where(np.isclose(low,1,rtol=0,atol=1e-10),0,low)
        high=np.where(np.isclose(high,1,rtol=0,atol=1e-10),0,high)
        low[:,axis]=high[:,axis]=0
        gap=max(cKDTree(low).query(high)[0].max(),cKDTree(high).query(low)[0].max())
        seams.append({'axis':axis,'low_nodes':len(low),'high_nodes':len(high),'max_gap_over_L':float(gap)})
    audit={'kind':kind,'geometry_intervals_per_axis':n,'background_N':32,'nodes':len(v),'triangles':len(f),
        'area_mm2':float(area.sum()*100),'nominal_shell_volume_mm3':float(area.sum()*50),
        'min_triangle_area_mm2':float(area.min()*100),'periodic_components':components,
        'nonmanifold_periodic_edges':int(np.sum(counts!=2)),
        'orientation_conflicts':int(np.sum(orient!=0)),'seams':seams,
        'analytic_vertex_residual_max':float(np.max(abs(implicit(kind,v)))),
        'marching_cube_interior_vertices':interior_vertices,'mesh_cleanup':collapse,
        'seam_definition':'actual open-edge traces, transverse periodic edge/corner equivalence; face tangencies excluded; quotient edges independently audited'}
    audit['pass']=bool(components==1 and np.all(counts==2) and np.all(orient==0)
        and all(s['max_gap_over_L']<1e-9 for s in seams))
    return v,f,audit

def shell_input(c,v,f,cfg,static=False):
    out=c/('initial_shell' if static else 'abaqus');out.mkdir()
    ids=np.arange(1,len(v)+1);qz=len(v)+1;groups={}
    for i,p in enumerate(v):
        key=tuple(np.round(np.where(np.isclose(p,1,atol=1e-10),0,p),9));groups.setdefault(key,[]).append(i)
    relations=[]
    for members in groups.values():
        root=min(members)
        for i in members:
            if i!=root:relations.append((i,root,np.rint(v[i]-v[root]).astype(int)))
    slaves={x[0] for x in relations};roots={x[1] for x in relations}
    assert len(slaves)==len(relations) and not slaves & roots
    pin=min(i for i in range(len(v)) if i not in slaves)
    lines=['*Heading',f'Canonical {cfg["case_id"]}; constant t=.5mm; XYZ periodic; no contact/plasticity','*Node']
    lines += [f'{i}, '+', '.join(f'{x:.16g}' for x in p*10) for i,p in zip(ids,v)]
    lines += [f'{qz}, 0., 0., 10.','*Element, type=S3R, elset=WALL']
    lines += [f'{i+1}, '+', '.join(str(ids[x]) for x in tri) for i,tri in enumerate(f)]
    lines += ['*Nset, nset=PHYSICAL']
    lines += [', '.join(str(x) for x in ids[i:i+16]) for i in range(0,len(ids),16)]
    lines += ['*Nset, nset=QZ',str(qz),'*Material, name=BASE']
    if static:lines += ['*Elastic','10., 0.3']
    else:lines += ['*Density','1e-9','*Hyperelastic, Neo Hooke','1.923076923076923, 0.24']
    lines += ['*Shell Section, elset=WALL, material=BASE','0.5, 5']
    for i,root,shift in relations:
        assert np.max(abs(v[i]-v[root]-shift))<1e-9
        for dof in range(1,7):
            line=f'{ids[i]}, {dof}, 1., {ids[root]}, {dof}, -1.'
            nt=2
            if dof==3 and shift[2]:line+=f', {qz}, 3, {-shift[2]}.';nt=3
            lines += ['*Equation',str(nt),line]
    lines += ['*Boundary',f'{ids[pin]}, 1, 3, 0.']
    if static:
        lines += ['*Step, name=COMPRESSION, nlgeom=NO','*Static','1., 1., 1e-5, 1.',
            '*Boundary','QZ, 3, 3, -0.001','*Output, field, frequency=1','*Node Output','U, UR, RF',
            '*Output, history, frequency=1','*Energy Output','ALLSE, ALLAE, ALLWK','*End Step']
    else:
        lines += ['*Amplitude, name=MACRO, definition=SMOOTH STEP, time=TOTAL TIME',
            '0., 0., 0.004, 1., 0.0044, 1.','*Step, name=compress20, nlgeom=YES',
            '*Dynamic, Explicit',', 0.0044','*Boundary, amplitude=MACRO','QZ, 3, 3, -2.',
            '*Output, field, number interval=40','*Node Output','U, UR, RF','*Element Output','S, LE, STH',
            '*Output, history, time interval=4.4e-6','*Energy Output','ALLIE, ALLSE, ALLKE, ALLAE, ALLVD, ALLWK, ETOTAL',
            '*Node Output, nset=QZ','U3, RF3','*End Step']
    (out/'shell.inp').write_text('\n'.join(lines)+'\n')
    meta={**cfg,'epsilon_z':-1e-4,'diagnostic_xyz':True,'physical_node_count':len(v),'element_count':len(f),
        'periodic_relations':len(relations),'macro_control_label':qz,'pin_node_label':int(ids[pin]),
        'step_name':'COMPRESSION' if static else 'compress20','no_contact':True,
        'load_time_seconds':.004,'hold_time_seconds':.0004,'total_time_seconds':.0044,
        'no_plasticity':True,'no_mass_scaling':True,'shell_inp_sha256':sha(out/'shell.inp')}
    write(out/'input.json',meta)
    src=R/('validation/initial_tangent_20261008_r30/abaqus/extract_shell.py' if static else 'scripts/extract_thin_explicit.py')
    shutil.copy2(src,out/'extract.py')

def main():
    start=time.perf_counter()
    for kind in (sys.argv[1:] or ['G','P','D']):
        c=D/kind;c.mkdir(exist_ok=False)
        v,f,audit=make_surface(kind);write(c/'geometry_audit.json',audit)
        print(json.dumps(audit),flush=True)
        assert audit['pass'],kind
        np.savez_compressed(c/'surface.npz',surface_vertices=v,surface_triangles=f)
        cfg={'case_id':kind+'_canonical_t005','N':32,'cell_size_mm':10.,'thickness_mm':.5,
            'E_MPa':10.,'nu':.3,'eta':1e-4,'interface_10_90_mm':.05,
            'mechanical_periodic_axes':[0,1,2],'solid_density_tonne_per_mm3':1e-9,
            'formula_level':0,'phase':[0,0,0],'compression_direction':[0,0,1],
            'surface_sha256':sha(c/'surface.npz')}
        write(c/'input.json',cfg)
        shell_input(c,v,f,cfg,True);shell_input(c,v,f,cfg,False)
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        fig=plt.figure(figsize=(6,5));ax=fig.add_subplot(projection='3d')
        ax.add_collection3d(Poly3DCollection(v[f]*10,facecolor='#6c91a8',edgecolor='none'))
        ax.set(xlim=(0,10),ylim=(0,10),zlim=(0,10),xlabel='x (mm)',ylabel='y (mm)',zlabel='z (mm)',
            title=f'{kind}: common midsurface; L=10 mm, t=0.50 mm');ax.set_box_aspect((1,1,1))
        fig.tight_layout();fig.savefig(c/'geometry.png',dpi=160);plt.close(fig)
    write(D/'geometry_preparation.json',{'wall_seconds':time.perf_counter()-start,'cases':['G','P','D'],
        'skimage_version':__import__('skimage').__version__,'new_dynamic_jobs':0})

if __name__=='__main__':main()

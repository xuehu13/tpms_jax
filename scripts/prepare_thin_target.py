"""Step 1: lock one periodic t/L=.05 case and inspect actual HEX8 Gauss coverage.

Builds an existing Problem but never solves displacement or submits Abaqus.
Usage: python scripts/prepare_thin_target.py validation/thin_target_20261004_r5
"""
from pathlib import Path
import argparse,hashlib,importlib.metadata,json,resource,sys,time
import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from surface_distance import PeriodicSurfaceDistance,thickness_occupancy


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_shell_mesh(path):
    """Read only Node and triangular Element blocks, retaining original labels."""
    nodes=[];elements=[];mode=None
    for line in Path(path).read_text().splitlines():
        text=line.strip()
        if not text or text.startswith('**'):continue
        if text.startswith('*'):
            mode='node' if text.lower().startswith('*node') else 'element' if text.lower().startswith('*element') else None
            continue
        row=text.split(',')
        if mode=='node':nodes.append([int(row[0]),*[float(s) for s in row[1:4]]])
        elif mode=='element':
            if len(row)!=4:raise ValueError('Expected triangular midsurface elements')
            elements.append([int(s) for s in row])
    node_ids=np.array([r[0] for r in nodes],dtype=np.int64)
    vertices=np.array([r[1:] for r in nodes])
    lookup={label:i for i,label in enumerate(node_ids)}
    triangles=np.array([[lookup[label] for label in r[1:]] for r in elements],dtype=np.int64)
    return vertices,triangles,node_ids,np.array([r[0] for r in elements],dtype=np.int64)


def mesh_audit(v,f):
    """Independent periodic-quotient edge/topology audit, not a volume proof."""
    cross=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])
    areas=np.linalg.norm(cross,axis=1)/2
    normal=cross/(2*areas[:,None])
    _,inverse=np.unique(np.round(v%1,9),axis=0,return_inverse=True)
    glued=inverse[f]
    directed=np.vstack([glued[:,[0,1]],glued[:,[1,2]],glued[:,[2,0]]])
    edges,which,counts=np.unique(np.sort(directed,axis=1),axis=0,return_inverse=True,return_counts=True)
    direction=np.where(directed[:,0]<directed[:,1],1,-1)
    orientation=np.bincount(which,weights=direction)
    graph=coo_matrix((np.ones(2*len(edges)),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])))
    components=connected_components(graph,directed=False,return_labels=False)
    seam=[]
    for axis in range(3):
        lo=v[np.isclose(v[:,axis],0,atol=1e-9)].copy();hi=v[np.isclose(v[:,axis],1,atol=1e-9)].copy()
        lo[:,axis]=0;hi[:,axis]=0
        forward=cKDTree(lo).query(hi)[0];back=cKDTree(hi).query(lo)[0]
        seam.append({'axis':axis,'low_nodes':len(lo),'high_nodes':len(hi),'max_coordinate_mismatch_over_L':float(max(forward.max(),back.max()))})
    # Validate the implicit expression only as a geometry approximation indicator.
    q=2*np.pi*v;x,y,z=q.T;a=3.4;b=4.6
    g=-a*np.cos(x)*np.cos(z)+b*np.sin(x)*np.sin(y)*np.sin(z)+2
    grad=2*np.pi*np.column_stack((a*np.sin(x)*np.cos(z)+b*np.cos(x)*np.sin(y)*np.sin(z),b*np.sin(x)*np.cos(y)*np.sin(z),a*np.cos(x)*np.sin(z)+b*np.sin(x)*np.sin(y)*np.cos(z)))
    error=np.abs(g)/np.linalg.norm(grad,axis=1)
    result={'nodes':len(v),'triangles':len(f),'area_over_L2':float(areas.sum()),
            'minimum_triangle_area_over_L2':float(areas.min()),'periodic_components':int(components),
            'periodic_edges_with_count_not_two':int(np.sum(counts!=2)),
            'periodic_orientation_conflicts':int(np.sum(orientation!=0)),
            'seams':seam,'estimated_node_normal_error_over_L_max':float(error.max()),
            'estimated_node_normal_error_over_L_p95':float(np.quantile(error,.95)),
            'error_definition':'abs(implicit value)/gradient norm; first-order nodal indicator, not a global bound'}
    result['pass']=bool(areas.min()>0 and components==1 and np.all(counts==2) and np.all(orientation==0) and all(s['max_coordinate_mismatch_over_L']<1e-8 for s in seam))
    return result,normal,areas


def periodic_components(mask):
    labels,count=ndimage.label(mask);parent=np.arange(count+1)
    def root(k):
        while parent[k]!=k:parent[k]=parent[parent[k]];k=parent[k]
        return k
    for axis in range(3):
        a=np.take(labels,0,axis=axis).ravel();b=np.take(labels,-1,axis=axis).ravel()
        for x,y in set(zip(a[(a>0)&(b>0)],b[(a>0)&(b>0)])):
            parent[root(x)]=root(y)
    return len({root(k) for k in range(1,count+1)})


def capture(directory):
    t0=time.perf_counter();cfg=json.loads((directory/'input.json').read_text())
    if (directory/'step1.json').exists():raise FileExistsError('Frozen capture exists')
    N=cfg['N'];thickness=cfg['thickness_over_L'];ell=cfg['interface_10_90_over_L']/(2*np.log(9))
    if thickness!=.05 or N!=64 or not 0<ell<thickness/2:raise ValueError('Outside locked thin case')
    v_mm,f,node_ids,element_ids=read_shell_mesh(directory/'input/shell_mesh.inc');v=v_mm/cfg['cell_size_mm']
    mesh,normal,areas=mesh_audit(v,f);assert mesh['pass'],mesh
    print('Periodic midsurface topology passed',flush=True)
    surface=PeriodicSurfaceDistance(v,f);t_tree=time.perf_counter()
    centers=v[f].mean(axis=1)
    # All triangle-centre offsets probe local encroachment; not a global injectivity proof.
    d_plus=surface.query(centers+thickness/2*normal);d_minus=surface.query(centers-thickness/2*normal)
    ratios=np.r_[d_plus,d_minus]/(thickness/2)
    # Area-weighted fixed sample: actual band intercepts along facet normals.
    rng=np.random.default_rng(20261004);sample=rng.choice(len(f),512,replace=False,p=areas/areas.sum())
    roots=[];bracket_fail=[]
    for sign in (-1,1):
        p=centers[sample];n=sign*normal[sample];lo=np.zeros(len(p));hi=np.full(len(p),thickness)
        bracket_fail.append(int(np.sum(surface.query(p+hi[:,None]*n)<=thickness/2)))
        for _ in range(22):
            mid=(lo+hi)/2;inside=surface.query(p+mid[:,None]*n)<=thickness/2
            lo=np.where(inside,mid,lo);hi=np.where(inside,hi,mid)
        roots.append((lo+hi)/2)
    measured=roots[0]+roots[1]
    offsets={'all_facet_normal_endpoint_distance_ratio_quantiles':np.quantile(ratios,[0,.01,.5,1]).tolist(),
             'endpoints_short_by_more_than_2_percent':int(np.sum(ratios<.98)),
             'normal_line_samples':512,'sampling':'area-weighted triangle centroids, fixed RNG; not a global thickness bound',
             'normal_line_band_width_over_L_quantiles':np.quantile(measured,[0,.01,.5,.99,1]).tolist(),
             'normal_line_bracket_failures':sum(bracket_fail),
             'global_offset_injectivity':'Not proved; local facet and connectivity diagnostics only'}
    print('Normal-offset diagnostics complete',flush=True)
    from density_fem import make_density_problem
    from pbc import xy_compression_fixed_dofs
    from fem import hex_jacobian_check
    import jax.numpy as jnp
    collected={}
    def field(problem):
        X=np.asarray(problem.physical_quad_points);weights=np.asarray(problem.JxW)[:,0,:]
        start=time.perf_counter();d=surface.query(X);rho=thickness_occupancy(d,thickness,ell)
        collected.update({'X':X,'weights':weights,'distance':d,'rho':np.asarray(rho),'query_seconds':time.perf_counter()-start})
        print(f'Actual Gauss distance/occupancy evaluated: {d.size} points',flush=True)
        return rho
    problem=make_density_problem(N,N,N,None,field,E_s=cfg['E_s'],E_min=cfg['eta']*cfg['E_s'],
        fixed_dofs=lambda pts:xy_compression_fixed_dofs(pts,N,N,N))
    min_det,volume=hex_jacobian_check(problem)
    X=collected['X'];rho=collected['rho'];d=collected['distance'];weights=collected['weights']
    np.testing.assert_array_equal(np.asarray(problem.rho),rho)
    cell_centers=np.asarray(problem.fe.points)[np.asarray(problem.fe.cells)].mean(axis=1)
    ijk=np.minimum((cell_centers*N).astype(int),N-1)
    grid=np.zeros((N,N,N));grid[tuple(ijk.T)]=rho.mean(axis=1)
    mid_cells=cKDTree(cell_centers).query(centers)[1]
    wall_cell_max=rho[mid_cells].max(axis=1)
    points_count=int(d.size)
    coverage={'N':N,'cells':int(problem.fe.num_cells),'Gauss_points_per_cell':int(problem.fe.num_quads),
              'Gauss_points':points_count,'thickness_over_cell_width':thickness*N,
              'solid_points_rho_ge_095':int(np.sum(rho>=.95)),'transition_points_rho_005_to_095':int(np.sum((rho>.05)&(rho<.95))),
              'void_points_rho_le_005':int(np.sum(rho<=.05)),
              'weighted_projected_Vf':float(np.sum(rho*weights)/weights.sum()),
              'same_Gauss_binary_Vf':float(np.sum((d<=thickness/2)*weights)/weights.sum()),
              'area_times_thickness_nominal_Vf':mesh['area_over_L2']*thickness,
              'mid_facet_cell_max_rho_min':float(wall_cell_max.min()),
              'mid_facet_cells_with_no_rho_ge_05':int(np.sum(wall_cell_max<.5)),
              'cell_mean_rho_ge_05_periodic_components':periodic_components(grid>=.5),
              'connectivity_definition':'6-neighbour thresholded cell-mean Gauss occupancy with XYZ wrap; geometric diagnostic, not mechanical proof',
              'reference_volume':float(volume),'minimum_hex_mapping_determinant':float(min_det)}
    # Plane-to-plane periodic fields; independently audited mesh seams above.
    q=rng.random((1024,3));periodic=[]
    for axis in range(3):
        shift=np.eye(3)[axis];error=np.max(np.abs(surface.query(q)-surface.query(q+shift)))
        periodic.append(float(error))
    # Two views show actual Gauss locations alongside a continuous distance slice.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    ticks=np.linspace(0,1,256);a,b=np.meshgrid(ticks,ticks,indexing='ij')
    flatX=X.reshape(-1,3);flatRho=rho.ravel()
    views=[]
    for k,z in enumerate([.25,.5,.75]):
        query=np.stack([a,b,np.full_like(a,z)],axis=-1);dist=surface.query(query)
        dens=np.asarray(thickness_occupancy(dist,thickness,ell))
        views.append({'z_over_L':z,'distance':dist,'rho':dens})
        axes[0,k].imshow(dens.T,origin='lower',extent=[0,1,0,1],cmap='viridis',vmin=0,vmax=1)
        axes[0,k].contour(ticks,ticks,dist.T,levels=[thickness/2],colors='white',linewidths=.5)
        axes[0,k].set_title(f'Distance-band slice z/L={z:g}')
        plane=np.argmin(np.abs(np.unique(flatX[:,2])-z));zq=np.unique(flatX[:,2])[plane]
        mask=np.abs(flatX[:,2]-zq)<1e-12
        axes[1,k].scatter(flatX[mask,0],flatX[mask,1],c=flatRho[mask],s=2,cmap='viridis',vmin=0,vmax=1,rasterized=True)
        axes[1,k].set_title(f'Actual Gauss plane z/L={zq:.6f}')
        for ax in axes[:,k]:ax.set(xlabel='x/L',ylabel='y/L',xlim=(0,1),ylim=(0,1),aspect='equal')
    fig.suptitle('diverse_28: constant distance-band t/L=0.05, N64; geometry only')
    fig.savefig(directory/'representation.png',dpi=160);plt.close(fig)
    np.savez_compressed(directory/'gauss_field.npz',physical_quad_points=X,JxW=weights,distance=d,rho=rho,
                        surface_vertices=v,surface_triangles=f,node_ids=node_ids,element_ids=element_ids)
    code_paths=['surface_distance.py','scripts/prepare_thin_target.py','density_fem.py','fem.py','pbc.py','pixi.toml','pixi.lock']
    versions={name:importlib.metadata.version(name) for name in ['libigl','numpy','scipy','jax','jax-fem','fenics-basix']}
    passed=bool(mesh['pass'] and sum(bracket_fail)==0 and coverage['mid_facet_cells_with_no_rho_ge_05']==0
                and coverage['cell_mean_rho_ge_05_periodic_components']==1 and min_det>0 and max(periodic)<1e-10)
    result={'case':cfg,'mesh':mesh,'offsets':offsets,'coverage':coverage,'periodic_distance_max_error_over_L':periodic,
            'step1_pass':passed,'interpretation':'Geometry/integration readiness only; bending, force response and equilibrium gradients are not validated',
            'no_displacement_solve':True,'no_Abaqus_job':True,'environment':versions,
            'timing':{'tree_and_mesh_seconds':t_tree-t0,'Gauss_query_seconds':collected['query_seconds'],'total_seconds':time.perf_counter()-t0},
            'maximum_process_RSS_GiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
            'input_sha256':{'input.json':sha(directory/'input.json'),'input/shell_mesh.inc':sha(directory/'input/shell_mesh.inc')},
            'source_sha256':{p:sha(ROOT/p) for p in code_paths},'gauss_field_sha256':sha(directory/'gauss_field.npz')}
    (directory/'step1.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    capture(parser.parse_args().directory.resolve())

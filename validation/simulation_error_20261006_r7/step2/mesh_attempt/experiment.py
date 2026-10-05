"""Bounded automatic quadratic tetrahedral reference-mesh attempt.

No Abaqus submission before geometric, periodic and element quality gates.
"""
from pathlib import Path
import hashlib,json,shutil,time,traceback
import numpy as np
import gmsh
E=Path('/home/xuehu/projects/tpms_jax/validation/simulation_error_20261006_r7/step2')
O=E/'mesh_attempt';O.mkdir(exist_ok=False)
started=time.perf_counter();stage='import'
gmsh.initialize();gmsh.option.setNumber('General.Terminal',1)
try:
    with np.load(E/'sharp_surface.npz') as f:points=f['points_mm'];tri=f['triangles']
    gmsh.model.add('sharp_periodic_thin_solid')
    tag=gmsh.model.addDiscreteEntity(2)
    gmsh.model.mesh.addNodes(2,tag,np.arange(1,len(points)+1),points.ravel())
    gmsh.model.mesh.addElementsByType(tag,2,np.arange(1,len(tri)+1),(tri+1).ravel())
    stage='classify_reparametrize'
    gmsh.model.mesh.classifySurfaces(np.deg2rad(45),True,True,np.pi)
    gmsh.model.mesh.createGeometry()
    surfaces=gmsh.model.getEntities(2)
    stage='periodic_surface_matching'
    plane={}
    for _,s in surfaces:
        _,xyz,_=gmsh.model.mesh.getNodes(2,s,True)
        xyz=np.asarray(xyz).reshape(-1,3)
        for a in range(3):
            for side in [0.,10.]:
                if np.max(abs(xyz[:,a]-side))<1e-7:
                    other=[i for i in range(3) if i!=a]
                    # Full original cap point set, not just a similar bounding box.
                    cloud=np.unique(np.round(xyz[:,other],7),axis=0)
                    key=(a,hashlib.sha256(cloud.tobytes()).hexdigest())
                    plane.setdefault(key,{})[side]=s
    pairs=[]
    for (a,_),caps in plane.items():
        if set(caps)!={0.,10.}:raise ValueError('Classified opposite planar patches do not match exactly')
        transform=np.eye(4);transform[a,3]=10
        gmsh.model.mesh.setPeriodic(2,[caps[10.]],[caps[0.]],transform.ravel())
        pairs.append((a,caps[0.],caps[10.]))
    if not pairs:raise ValueError('No periodic cap pairs recognized')
    stage='volume_mesh'
    loop=gmsh.model.geo.addSurfaceLoop([s for _,s in surfaces]);vol=gmsh.model.geo.addVolume([loop])
    gmsh.model.geo.synchronize()
    gmsh.option.setNumber('Mesh.MeshSizeMin',.16)
    gmsh.option.setNumber('Mesh.MeshSizeMax',.20)
    gmsh.option.setNumber('Mesh.MeshSizeFromCurvature',0)
    gmsh.option.setNumber('Mesh.MeshSizeFromPoints',0)
    gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary',0)
    gmsh.option.setNumber('Mesh.Algorithm3D',1)
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize('Netgen')
    gmsh.model.mesh.setOrder(2)
    stage='quality'
    types,tags,nodes=gmsh.model.mesh.getElements(3)
    if list(types)!=[11]:raise ValueError('Expected only quadratic 10-node tetrahedra')
    elementtags=tags[0]
    quality=np.array(gmsh.model.mesh.getElementQualities(elementtags,'minSJ'))
    gmsh.write(str(O/'reference.msh'))
    nt,xyz,_=gmsh.model.mesh.getNodes();xyz=np.array(xyz).reshape(-1,3)
    np.savez_compressed(O/'reference.npz',node_labels=nt,points_mm=xyz,element_labels=elementtags,tetra10=nodes[0].reshape(-1,10))
    mappings=[]
    for a,master,slave in pairs:
        m,sl,ma,_=gmsh.model.mesh.getPeriodicNodes(2,slave,True)
        mappings.append({'axis':a,'master_surface':master,'slave_surface':slave,'node_pairs':len(sl)})
        if m!=master or len(sl)==0:raise ValueError('Quadratic periodic mapping missing')
    result={'status':'quadratic_mesh_ready_pending_full_quality' if quality.min()>0 else 'rejected_nonpositive_scaled_jacobian',
            'nodes':len(nt),'elements':len(elementtags),'scaled_J_quantiles':np.quantile(quality,[0,.01,.5,1]).tolist(),
            'periodic_surface_pairs':mappings,'body_seconds':time.perf_counter()-started,
            'scope':'Mesh preparation only. Actual volume Jacobians, opposite-node coordinates, thickness, density and analysis quality remain to be checked before acceptance.'}
except Exception as exc:
    result={'status':'mesh_attempt_stopped','stage':stage,'error':str(exc),'body_seconds':time.perf_counter()-started,
            'scope':'No independent mechanical reference obtained; do not treat geometry or a failed mesh as truth.'}
    traceback.print_exc()
finally:
    gmsh.finalize()
    shutil.copy2(__file__,O/'experiment.py')
    result['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['geometry_sha256']=hashlib.sha256((E/'sharp_surface.npz').read_bytes()).hexdigest()
    (O/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)

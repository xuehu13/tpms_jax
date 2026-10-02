"""Conforming piecewise-linear binary Gyroid domain, with matched periodic caps.

Freudenthal background tetrahedra are CLIPPED at both signed levels. Void is
absent. Cut polyhedra are triangulated consistently and filled by tetrahedra.
This is a surface approximation to |G|<=c, not element-centroid voxel deletion.
"""
import itertools
import numpy as np

C = 0.541062
FACES = ((0,2,1),(0,1,3),(1,2,3),(2,0,3))
TET_EDGES = ((0,1),(1,2),(2,0),(0,3),(1,3),(2,3))


def gyroid_numpy(xyz):
    x,y,z = np.moveaxis(2*np.pi*np.asarray(xyz),-1,0)
    return np.sin(x)*np.cos(y)+np.sin(y)*np.cos(z)+np.sin(z)*np.cos(x)


def clip_polygon(polygon, level, greater):
    result = []
    for a,b in zip(polygon,polygon[1:]+polygon[:1]):
        inside_a = a[3] >= level if greater else a[3] <= level
        inside_b = b[3] >= level if greater else b[3] <= level
        if inside_a:
            result.append(a)
        if inside_a != inside_b:
            t = (level-a[3])/(b[3]-a[3])
            point = a+t*(b-a)
            point[3] = level
            result.append(point)
    return result


def ordered_polygon(vertices, normal):
    points = np.array(vertices)
    normal = normal/np.linalg.norm(normal)
    direction = np.eye(3)[np.argmin(np.abs(normal))]
    e1 = np.cross(normal,direction); e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal,e1)
    centered = points[:,:3]-points[:,:3].mean(axis=0)
    angles = np.arctan2(centered@e2,centered@e1)
    return list(points[np.argsort(angles)])


def linear_domain(n, c=C, origins=None):
    if n < 4 or not 0 < c < 1.5:
        raise ValueError("Require n>=4 and 0<c<1.5")
    integers = np.array(list(itertools.product(range(n+1),repeat=3)))
    lattice = integers/n
    # Opposite planes sample identical periodic coordinates, bit for bit.
    values = gyroid_numpy((integers % n)/n)
    coordinates, tags, tets, intersections = [], {}, [], {}
    def node(point):
        key = tuple(np.rint(np.asarray(point)*1e13).astype(np.int64))
        if key not in tags:
            tags[key] = len(coordinates)
            coordinates.append(np.array(key,dtype=float)/1e13)
        return tags[key]
    def vertex(i):
        return np.r_[lattice[i],values[i]]
    def index(p):
        return (p[0]*(n+1)+p[1])*(n+1)+p[2]
    if origins is None:
        origins = itertools.product(range(n),repeat=3)
    for origin in origins:
        origin = np.array(origin)
        for permutation in itertools.permutations(range(3)):
            offsets = [origin.copy()]
            for direction in permutation:
                next_point = offsets[-1].copy(); next_point[direction] += 1
                offsets.append(next_point)
            original_ids = [index(p) for p in offsets]
            tetra = np.array([vertex(i) for i in original_ids])
            if np.linalg.det((tetra[1:,:3]-tetra[0,:3]).T) < 0:
                tetra[[1,2]] = tetra[[2,1]]
                original_ids[1],original_ids[2] = original_ids[2],original_ids[1]
            g = tetra[:,3]
            if g.min() > c or g.max() < -c:
                continue
            if np.max(np.abs(g)) <= c:
                tets.append([node(p[:3]) for p in tetra])
                continue
            # Compute each isosurface/background-edge intersection ONCE, in
            # canonical endpoint order. Opposite polygon traversal directions
            # otherwise differ by floating roundoff at the node-hash boundary,
            # causing duplicate vertices and nonmanifold cells (G48 regression).
            candidates = {-c:[], c:[]}
            for a,b in TET_EDGES:
                lo,hi = sorted((original_ids[a],original_ids[b]))
                for level in (-c,c):
                    if min(values[lo],values[hi]) < level < max(values[lo],values[hi]):
                        key = (lo,hi,level)
                        if key not in intersections:
                            fraction = (level-values[lo])/(values[hi]-values[lo])
                            intersections[key] = np.r_[lattice[lo]+fraction*(lattice[hi]-lattice[lo]),level]
                        candidates[level].append(intersections[key])
            def canonical(point):
                if point[3] in candidates and candidates[point[3]]:
                    choices = np.array(candidates[point[3]])
                    distances = np.sum((choices[:,:3]-point[:3])**2,axis=1)
                    selected = np.argmin(distances)
                    if distances[selected] > 1e-24:
                        raise ValueError('Cut point is not on an original tetrahedral edge')
                    return choices[selected]
                return point
            polygons = []
            for face in FACES:
                polygon = clip_polygon(list(tetra[list(face)]),-c,True)
                polygon = clip_polygon(polygon,c,False)
                if len(polygon) >= 3:
                    polygons.append([canonical(point) for point in polygon])
            gradient = np.linalg.solve(tetra[1:,:3]-tetra[0,:3],g[1:]-g[0])
            for level,normal in ((-c,-gradient),(c,gradient)):
                vertices = {}
                for polygon in polygons:
                    for p in polygon:
                        if p[3] == level:
                            vertices[tuple(np.rint(p[:3]*1e13).astype(np.int64))] = p
                if len(vertices) >= 3:
                    polygons.append(ordered_polygon(list(vertices.values()),normal))
            unique = {node(p[:3]) for polygon in polygons for p in polygon}
            if len(unique) < 4:
                raise ValueError("Degenerate clipped tetrahedron")
            center = node(np.array([coordinates[i] for i in unique]).mean(axis=0))
            for polygon in polygons:
                ids = [node(p[:3]) for p in polygon]
                # Lexicographic physical coordinates are invariant under the
                # periodic translation; node insertion order is irrelevant.
                anchor = min(range(len(ids)),key=lambda i:tuple(coordinates[ids[i]]))
                ids = ids[anchor:]+ids[:anchor]
                for i in range(1,len(ids)-1):
                    tet = [center,ids[0],ids[i],ids[i+1]]
                    xyz = np.array([coordinates[j] for j in tet])
                    det = np.linalg.det((xyz[1:]-xyz[0]).T)
                    if abs(det) < 1e-20:
                        raise ValueError("Zero-volume clipped element")
                    if det < 0:
                        tet[2],tet[3] = tet[3],tet[2]
                    tets.append(tet)
    return np.array(coordinates),np.array(tets,dtype=int)


def boundary_faces(cells):
    faces = cells[:,FACES].reshape(-1,3)
    keys = np.sort(faces,axis=1)
    unique,first,counts = np.unique(keys,axis=0,return_index=True,return_counts=True)
    if counts.max() > 2:
        raise ValueError("Nonmanifold tetrahedral mesh")
    return faces[first[counts == 1]]


def audit_linear(points,cells,c=C):
    xyz = points[cells]
    determinants = np.linalg.det(np.transpose(xyz[:,1:]-xyz[:,0,None,:],(0,2,1)))
    if determinants.min() <= 0:
        raise ValueError("Nonpositive tetrahedral Jacobian")
    boundary = boundary_faces(cells)
    edges = np.sort(boundary[:,((0,1),(1,2),(2,0))].reshape(-1,2),axis=1)
    _,edge_counts = np.unique(edges,axis=0,return_counts=True)
    if not np.all(edge_counts == 2):
        raise ValueError("Binary boundary is not a closed manifold")
    periodic = {}
    for axis in (0,1):
        transverse = [a for a in range(3) if a != axis]
        sets = []
        for plane in (0,1):
            selected = boundary[np.all(points[boundary,axis] == plane,axis=1)]
            keys = {tuple(sorted(tuple(np.rint(points[i,transverse]*1e12).astype(np.int64)) for i in face)) for face in selected}
            sets.append(keys)
        if sets[0] != sets[1] or not sets[0]:
            raise ValueError("Opposite cap triangulations do not match")
        periodic[str(axis)] = len(sets[0])
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    edges = cells[:,TET_EDGES].reshape(-1,2)
    graph = coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(len(points),len(points)))
    components,_ = connected_components(graph,directed=False)
    # The cropped sheet has small corner pieces. They join the body through
    # x/y periodic pairs; requiring Euclidean connectivity would reject the
    # correct domain. Check connectivity of the constrained domain instead.
    masters = {}
    representatives = []
    for point in points:
        canonical = point.copy()
        canonical[:2][canonical[:2] == 1] = 0
        key = tuple(np.rint(canonical*1e12).astype(np.int64))
        if key not in masters:
            masters[key] = len(masters)
        representatives.append(masters[key])
    representatives = np.array(representatives)
    joined = representatives[edges]
    graph = coo_matrix((np.ones(len(joined)),(joined[:,0],joined[:,1])),shape=(len(masters),len(masters)))
    periodic_components,_ = connected_components(graph,directed=False)
    if periodic_components != 1:
        raise ValueError("Binary solid is disconnected after x/y periodic pairing")
    caps = np.zeros(len(boundary),dtype=bool)
    for axis in range(3):
        for plane in (0,1):
            caps |= np.all(points[boundary,axis] == plane,axis=1)
    free_surface = boundary[~caps]
    sample = np.r_[points[np.unique(free_surface)],points[free_surface].mean(axis=1)]
    residual = np.abs(np.abs(gyroid_numpy(sample))-c)
    lengths = np.sum((xyz[:,np.array(TET_EDGES)[:,0]]-xyz[:,np.array(TET_EDGES)[:,1]])**2,axis=(1,2))
    quality = 12*(3*determinants/6)**(2/3)/lengths
    return {"nodes":len(points),"elements":len(cells),"volume":float(determinants.sum()/6),
        "min_detJ":float(determinants.min()),"mean_ratio_min":float(quality.min()),
        "mean_ratio_p01":float(np.quantile(quality,0.01)),"mean_ratio_median":float(np.median(quality)),
        "connected_components":int(components),"periodic_connected_components":int(periodic_components),"boundary_triangles":len(boundary),
        "periodic_cap_triangles":periodic,"surface_G_residual_max":float(residual.max()) if len(residual) else None,
        "surface_G_residual_rms":float(np.sqrt(np.mean(residual**2))) if len(residual) else None}


def quadratic_nodes(points,cells):
    # Abaqus C3D10 ordering: 12,23,31,14,24,34 after four vertices.
    coordinates = list(points.copy())
    edge_nodes = {}
    result = []
    for cell in cells:
        mids = []
        for a,b in TET_EDGES:
            edge = tuple(sorted((int(cell[a]),int(cell[b]))))
            if edge not in edge_nodes:
                edge_nodes[edge] = len(coordinates)
                coordinates.append((points[edge[0]]+points[edge[1]])/2)
            mids.append(edge_nodes[edge])
        result.append(list(cell)+mids)
    return np.array(coordinates),np.array(result,dtype=int)


def remesh_domain(points, cells, refinement=0):
    """Remesh only the volume; preserve the prescribed closed triangle skin.

    Optional uniform subdivision refines the SAME planar domain. Geometry N
    and finite-element refinement are consequently independent parameters.
    """
    import gmsh
    boundary = boundary_faces(cells)
    used = np.unique(boundary)
    gmsh.initialize()
    gmsh.logger.start()
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.model.add('binary_gyroid')
        surface = gmsh.model.addDiscreteEntity(2)
        gmsh.model.mesh.addNodes(2, surface, (used+1).tolist(), points[used].ravel().tolist())
        gmsh.model.mesh.addElementsByType(surface, 2, list(range(1,len(boundary)+1)),
                                         (boundary+1).ravel().tolist())
        loop = gmsh.model.geo.addSurfaceLoop([surface])
        gmsh.model.geo.addVolume([loop])
        gmsh.model.geo.synchronize()
        gmsh.option.setNumber('Mesh.MeshOnlyEmpty', 1)
        gmsh.model.mesh.generate(3)
        # Netgen's bundled optimizer segfaulted on the N=16 cut skin in this
        # environment. Use Gmsh's own volume optimizer and retain its warnings.
        gmsh.model.mesh.optimize('', niter=5)
        for _ in range(refinement):
            gmsh.model.mesh.refine()
        tags, coordinates, _ = gmsh.model.mesh.getNodes()
        types, _, connectivity = gmsh.model.mesh.getElements(3)
        if len(types) != 1 or types[0] != 4:
            raise ValueError('Expected linear tetrahedra from volume mesher')
        lookup = {int(tag): i for i,tag in enumerate(tags)}
        result = np.array([lookup[int(t)] for t in connectivity[0]]).reshape(-1,4)
        # Gmsh 4.15.2 can return identical tetrahedra on this multi-component
        # closed skin. Keep one occurrence, then require a manifold boundary
        # and unchanged volume; never remove distinct or small tetrahedra.
        _, unique = np.unique(np.sort(result,axis=1),axis=0,return_index=True)
        duplicate_count = len(result)-len(unique)
        result = result[np.sort(unique)]
        xyz = np.array(coordinates).reshape(-1,3)
        det = np.linalg.det(np.transpose(xyz[result][:,1:]-xyz[result][:,0,None,:],(0,2,1)))
        negative = det < 0
        result[negative,1:3] = result[negative,2:0:-1]
        messages = gmsh.logger.get()
        if any(message.startswith('Error') for message in messages):
            raise ValueError('Gmsh reported an error: '+str(messages))
        warnings = [message for message in messages if message.startswith('Warning')]
        metadata = {'gmsh_version':gmsh.__version__, 'warnings':warnings,
                    'refinement':refinement, 'identical_tetrahedra_removed':duplicate_count}
        # This also detects asymmetric boundary recovery/Steiner insertion.
        audit_linear(xyz,result)
        before = audit_linear(points,cells)['volume']
        after = audit_linear(xyz,result)['volume']
        if abs(before-after) > 1e-10:
            raise ValueError('Volume mesher changed the prescribed binary domain')
        return xyz,result,metadata
    finally:
        gmsh.logger.stop()
        gmsh.finalize()

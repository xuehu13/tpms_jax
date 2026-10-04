"""Generate binary Gyroid/Primitive sheet-solid C3D10 inputs. No jobs are submitted here."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binary_gyroid import C, audit_linear, linear_domain, quadratic_nodes, remesh_domain


def coordinate_key(point):
    return tuple(np.rint(np.asarray(point)*1e12).astype(np.int64))


def constraints(points, lateral):
    if lateral not in ('fixed', 'relaxed_free'):
        raise ValueError('Unsupported lateral condition')
    lookup = {coordinate_key(point):i+1 for i,point in enumerate(points)}
    if len(lookup) != len(points):
        raise ValueError('Coincident nodes')
    controls = tuple(len(points)+i+1 for i in range(3))
    equations, pairs = [], []
    for label,point in enumerate(points,1):
        canonical = point.copy()
        jump = np.array([int(point[0] == 1), int(point[1] == 1), 0])
        canonical[:2] -= jump[:2]
        if not np.any(jump):
            continue
        key = coordinate_key(canonical)
        if key not in lookup:
            raise ValueError('Missing periodic master for '+str(point))
        master = lookup[key]
        pairs.append((label,master,jump.tolist()))
        for comp in (1,2,3):
            if comp == 3 and point[2] in (0,1):
                continue
            terms = [(label,comp,1.0),(master,comp,-1.0)]
            if comp < 3 and jump[comp-1]:
                terms.append((controls[comp-1],comp,-1.0))
            equations.append(terms)
    top = np.flatnonzero(points[:,2] == 1)+1
    bottom = np.flatnonzero(points[:,2] == 0)+1
    if len(top) == 0 or len(bottom) == 0:
        raise ValueError('Missing load faces')
    for label in top:
        equations.append([(int(label),3,1.0),(controls[2],3,-1.0)])
    anchor = lookup.get((0,0,0))
    if anchor is None:
        if lateral != "fixed":
            raise ValueError("Non-corner anchor currently supports fixed macro lateral strain only")
        # With zero macro lateral strain this fixes only the rigid translation.
        anchor = int(bottom[np.lexsort((points[bottom-1,1], points[bottom-1,0]))][0])
    bcs = [(int(label),3,0.0) for label in bottom]
    bcs += [(anchor,1,0.0),(anchor,2,0.0),(controls[2],3,-0.01)]
    if lateral == 'fixed':
        bcs += [(controls[0],1,0.0),(controls[1],2,0.0)]
    eliminated = [terms[0][:2] for terms in equations]
    fixed = [term[:2] for term in bcs]
    references = {term[:2] for terms in equations for term in terms[1:]}
    if len(set(eliminated)) != len(eliminated) or len(set(fixed)) != len(fixed):
        raise ValueError('Duplicate equation elimination or BC')
    if set(eliminated) & (set(fixed)|references):
        raise ValueError('Eliminated DOF reused in equation or BC')
    for eps in ([0,0,-0.01],[0.003,0.004,-0.01]):
        values = {(label,i+1):float(point[i]*eps[i]) for label,point in enumerate(points,1) for i in range(3)}
        values.update({(controls[i],i+1):eps[i] for i in range(3)})
        if max(abs(sum(coef*values[(label,comp)] for label,comp,coef in eq)) for eq in equations) > 1e-12:
            raise ValueError('Constraint fails independent affine-field check')
    return controls,equations,bcs,pairs,top,bottom


def uniform_domain(n=2):
    points = np.array(list(itertools.product(range(n+1),repeat=3)),float)/n
    def index(p): return (p[0]*(n+1)+p[1])*(n+1)+p[2]
    cells = []
    for origin in itertools.product(range(n),repeat=3):
        for perm in itertools.permutations(range(3)):
            path = [np.array(origin)]
            for axis in perm:
                new = path[-1].copy(); new[axis] += 1; path.append(new)
            cell = [index(p) for p in path]
            if np.linalg.det((points[cell[1:]]-points[cell[0]]).T) < 0:
                cell[1],cell[2] = cell[2],cell[1]
            cells.append(cell)
    return points,np.array(cells)


def prepare(output, n=8, refinement=0, lateral='fixed', uniform=False, cached_mesh=None, c=C, family="gyroid"):
    if n < 4 or refinement not in (0,1) or lateral not in ('fixed','relaxed_free'):
        raise ValueError('Require geometry N>=4, refinement 0/1, and supported lateral condition')
    if family not in ("gyroid", "primitive") or (family == "primitive" and (uniform or np.asarray(c).ndim != 0)):
        raise ValueError("Primitive requires scalar threshold and nonuniform geometry")
    output.mkdir(parents=True,exist_ok=True)
    if uniform:
        points,cells = uniform_domain()
        metadata = {'mesher':'Freudenthal cube patch'}
        prefix = 'binary_uniform_C3D10'
    else:
        if cached_mesh:
            saved = np.load(cached_mesh)
            if 'geometry_N' not in saved or 'c' not in saved or int(saved['geometry_N']) != n or not np.array_equal(np.asarray(saved['c']),np.asarray(c)):
                raise ValueError('Cached mesh must record matching geometry_N and c')
            cached_family = str(saved["family"]) if "family" in saved else "gyroid"
            if cached_family != family:
                raise ValueError("Cached mesh family mismatch")
            points,cells = saved['points'],saved['cells']
            metadata = {'mesher':'cached boundary-preserving mesh', 'cache_sha256':hashlib.sha256(Path(cached_mesh).read_bytes()).hexdigest()}
            # Refine an already meshed volume through Gmsh, if requested.
            # Cache is for development runs only, not the reproducible default.
            cached_refinement = int(saved['fe_refinement']) if 'fe_refinement' in saved else 0
            if cached_refinement > refinement:
                raise ValueError('Cached FE refinement exceeds requested level')
            if refinement > cached_refinement:
                points,cells,metadata = remesh_domain(points,cells,refinement,c=c,family=family)
        else:
            points,cells = linear_domain(n, c, family=family)
            points,cells,metadata = remesh_domain(points,cells,refinement,c=c,family=family)
        prefix = f'binary_{family}_G{n}_R{refinement}_C3D10'
    audit = audit_linear(points,cells,c,family)
    if uniform:
        # Surface G residuals have no geometric meaning for this patch test.
        audit.pop('surface_G_residual_max'); audit.pop('surface_G_residual_rms')
    points,cells = quadratic_nodes(points,cells)
    controls,equations,bcs,pairs,top,bottom = constraints(points,lateral)
    job = prefix+'_'+lateral
    lines = ['*HEADING','Binary solid, quadratic tetrahedra, XY periodic / flat axial faces',
             '** Small-strain E=10, nu=0.3; void is absent; gross-cell volume=1',
             '*NODE, NSET=PHYSICAL']
    lines += [f'{label}, '+', '.join(f'{v:.17g}' for v in point) for label,point in enumerate(points,1)]
    for name,label in zip(('QX','QY','QZ'),controls):
        lines += [f'*NODE, NSET={name}',f'{label}, 0., 0., 0.']
    lines += ['*ELEMENT, TYPE=C3D10, ELSET=SOLID']
    lines += [f'{label}, '+', '.join(str(int(v)+1) for v in cell) for label,cell in enumerate(cells,1)]
    for name,ids in (('TOP',top),('BOTTOM',bottom)):
        lines += [f'*NSET, NSET={name}']
        lines += [', '.join(map(str,ids[i:i+16])) for i in range(0,len(ids),16)]
    lines += ['*MATERIAL, NAME=SOLID_MATERIAL','*ELASTIC','10., 0.3',
              '*SOLID SECTION, ELSET=SOLID, MATERIAL=SOLID_MATERIAL',',']
    for eq in equations:
        lines += ['*EQUATION',str(len(eq)),', '.join(f'{label}, {comp}, {coef:.17g}' for label,comp,coef in eq)]
    lines += ['*STEP, NAME=COMPRESSION, NLGEOM=NO','*STATIC','1., 1.','*BOUNDARY']
    lines += [f'{label}, {comp}, {comp}, {value:.17g}' for label,comp,value in bcs]
    lines += ['*OUTPUT, FIELD, FREQUENCY=1','*NODE OUTPUT','U, RF',
              '*ELEMENT OUTPUT, ELSET=SOLID','S, E, IVOL',
              '*OUTPUT, HISTORY, FREQUENCY=1','*ENERGY OUTPUT','ALLSE','*END STEP']
    inp = output/(job+'.inp')
    inp.write_text('\n'.join(lines)+'\n',encoding='ascii')
    np.savez_compressed(output/(job+'.mesh.npz'),points=points,cells=cells)
    manifest = {'case':job,'model':'uniform' if uniform else 'binary_sheet_'+family,'family':family,
                'geometry_N':n if not uniform else None,'fe_refinement':refinement,
                'c':float(c) if np.asarray(c).ndim == 0 else np.asarray(c).tolist(),'E_s':10.,'nu':.3,'gross_volume':1.,'lateral':lateral,
                'geometry':audit,'meshing':metadata,'nodes':len(points),'elements':len(cells),
                'controls':list(controls),'equations':len(equations),'periodic_pairs':len(pairs),
                'input_sha256':hashlib.sha256(inp.read_bytes()).hexdigest(),
                'mesh_sha256':hashlib.sha256((output/(job+'.mesh.npz')).read_bytes()).hexdigest(),
                'relative_tolerance':1e-6,'displacement_tolerance':1e-8,
                'convergence_claim':False}
    if uniform:
        eps = np.array([0,0,-.01]) if lateral == 'fixed' else np.array([.003,.003,-.01])
        lam,mu = 10*.3/(1.3*.4),10/2.6
        sigma = lam*eps.sum()+2*mu*eps
        manifest['analytic'] = {'eps':eps.tolist(),'sigma_diagonal':sigma.tolist(),
                                'energy':float(.5*sigma@eps)}
    (output/(job+'.expected.json')).write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(json.dumps(manifest,indent=2),flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--n',type=int,default=8)
    parser.add_argument('--refinement',type=int,choices=(0,1),default=0)
    parser.add_argument('--lateral',choices=('fixed','relaxed_free'),default='fixed')
    parser.add_argument('--uniform',action='store_true')
    parser.add_argument('--cached-mesh',type=Path)
    parser.add_argument('--family',choices=('gyroid','primitive'),default='gyroid')
    geometry=parser.add_mutually_exclusive_group()
    geometry.add_argument('--c',type=float,default=C)
    geometry.add_argument('--width-parameters',nargs=4,type=float)
    args=parser.parse_args()
    prepare(args.output,args.n,args.refinement,args.lateral,args.uniform,args.cached_mesh,args.c if args.width_parameters is None else np.array(args.width_parameters),args.family)

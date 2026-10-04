"""Minimal family input extension for round4 step2; no FEM solver copied."""
from pathlib import Path
import hashlib
import json
import subprocess

P = Path('/home/xuehu/projects/tpms_jax')
O = P/'validation/mechanics_trust_20261003_r4/step2'
O.mkdir(exist_ok=False)


def sha(f):
    return hashlib.sha256(f.read_bytes()).hexdigest()


files = [f for f in P.glob('*.py')]
files += [f for f in (P/'scripts').iterdir() if f.is_file()]
files += list((P/'tests').glob('*.py')) + [P/'pixi.toml', P/'pixi.lock', P/'results/m4_numerical_study.csv']
for name in ('near_term_20261003', 'geometry_interface_20261003_r2', 'learning_bridge_20261003_r3', 'mechanics_trust_20261003_r4/step1'):
    files += [f for f in (P/'validation'/name).rglob('*') if f.is_file()]
(O/'preservation_before.json').write_text(json.dumps({'HEAD': subprocess.check_output(['git','-C',str(P),'rev-parse','HEAD'],text=True).strip(),
                                                    'sha256': {str(f):sha(f) for f in files}},indent=2)+'\n')


def edit(name, replacements, append=''):
    f = P/name
    old = f.read_text()
    backup = O/'source_before'/name
    backup.parent.mkdir(parents=True, exist_ok=True)
    backup.write_bytes(f.read_bytes())
    for before, after in replacements:
        assert before in old, (name, before)
        old = old.replace(before, after)
    f.write_text(old+append)


edit('geometry.py', [('__all__ = ["gyroid",', '__all__ = ["primitive", "gyroid",'),
                     ('fixed Gyroid amplitude convention', 'the stated implicit-field amplitude convention')],
     '\n\ndef primitive(xyz, L=1.0):\n    """Schwarz Primitive nodal approximation: cos(X)+cos(Y)+cos(Z)."""\n    return jnp.sum(jnp.cos(2*jnp.pi*jnp.asarray(xyz)/L), axis=-1)\n')
edit('binary_gyroid.py', [
    ('binary Gyroid domain', 'binary Gyroid/Primitive domain'),
    ('def width_numpy(', 'def implicit_numpy(xyz, family="gyroid"):\n    if family == "gyroid":\n        return gyroid_numpy(xyz)\n    if family == "primitive":\n        return np.cos(2*np.pi*np.asarray(xyz)).sum(axis=-1)\n    raise ValueError("Unsupported implicit family")\n\n\ndef width_numpy('),
    ('def linear_domain(n, c=C, origins=None):', 'def linear_domain(n, c=C, origins=None, family="gyroid"):'),
    ('values = gyroid_numpy(periodic_points)', 'values = implicit_numpy(periodic_points, family)'),
    ('def audit_linear(points,cells,c=C):', 'def audit_linear(points,cells,c=C,family="gyroid"):'),
    ('np.abs(gyroid_numpy(sample))', 'np.abs(implicit_numpy(sample,family))'),
    ('def remesh_domain(points, cells, refinement=0, c=C):', 'def remesh_domain(points, cells, refinement=0, c=C, family="gyroid"):'),
    ('audit_linear(xyz,result,c)', 'audit_linear(xyz,result,c,family)'),
    ('audit_linear(points,cells,c)', 'audit_linear(points,cells,c,family)')])
edit('scripts/prepare_abaqus_binary.py', [
    ('if anchor is None:\n        raise ValueError(\'Expected solid corner for displacement anchor\')',
     'if anchor is None:\n        if lateral != "fixed":\n            raise ValueError("Non-corner anchor currently supports fixed macro lateral strain only")\n        # With zero macro lateral strain this fixes only the rigid translation.\n        anchor = int(bottom[np.lexsort((points[bottom-1,1], points[bottom-1,0]))][0])'),
    ('cached_mesh=None, c=C):', 'cached_mesh=None, c=C, family="gyroid"):'),
    ('    output.mkdir(parents=True,exist_ok=True)', '    if family not in ("gyroid", "primitive") or (family == "primitive" and (uniform or np.asarray(c).ndim != 0)):\n        raise ValueError("Primitive requires scalar threshold and nonuniform geometry")\n    output.mkdir(parents=True,exist_ok=True)'),
    ('            points,cells = saved[\'points\'],saved[\'cells\']', '            cached_family = str(saved["family"]) if "family" in saved else "gyroid"\n            if cached_family != family:\n                raise ValueError("Cached mesh family mismatch")\n            points,cells = saved[\'points\'],saved[\'cells\']'),
    ('remesh_domain(points,cells,refinement,c=c)', 'remesh_domain(points,cells,refinement,c=c,family=family)'),
    ('linear_domain(n, c)', 'linear_domain(n, c, family=family)'),
    ("prefix = f'binary_gyroid_G{n}_R{refinement}_C3D10'", "prefix = f'binary_{family}_G{n}_R{refinement}_C3D10'"),
    ('audit_linear(points,cells,c)', 'audit_linear(points,cells,c,family)'),
    ("'model':'uniform' if uniform else 'binary_sheet_gyroid'", "'model':'uniform' if uniform else 'binary_sheet_'+family,'family':family"),
    ("parser.add_argument('--cached-mesh',type=Path)", "parser.add_argument('--cached-mesh',type=Path)\n    parser.add_argument('--family',choices=('gyroid','primitive'),default='gyroid')"),
    ('else np.array(args.width_parameters))', 'else np.array(args.width_parameters),args.family)')])
edit('scripts/m4_numerical_study.py', [
    ('(onp.abs(onp.asarray(gyroid(X_q))) <= problem.projection_c)',
     '(onp.abs(onp.asarray(getattr(problem, "geometry_field", gyroid)(X_q))) <= problem.projection_c)')])
edit('scripts/capture_binary_projection_reference.py', [
    ('parser.add_argument("--voxel-M",', 'parser.add_argument("--family", choices=("gyroid", "primitive"), default="gyroid")\n    parser.add_argument("--field-out", type=Path, help="Optional uniform-grid total displacement for mode comparison")\n    parser.add_argument("--voxel-M",'),
    ('    if args.out.exists():', '    if args.family == "primitive" and (args.voxel_M is not None or args.target_vf is not None):\n        parser.error("Primitive currently supports direct analytic input without volume calibration")\n    if args.field_out is not None and args.field_out.exists():\n        parser.error("Field output already exists")\n    if args.out.exists():'),
    ('    voxel_record = None', '    voxel_record = None\n    if args.family == "primitive":\n        from geometry import primitive, project_field\n        def rho_quad(problem):\n            problem.projection_c, problem.projection_calibration = args.c, None\n            problem.geometry_field = primitive\n            return project_field(primitive(problem.physical_quad_points), args.c, args.beta)'),
    ('        del problem, sol_list, points, w', '        if args.field_out is not None:\n            indices = np.rint(points*args.N).astype(int)\n            grid = np.empty((args.N+1, args.N+1, args.N+1, 3))\n            H = np.diag([row["eps_x"], row["eps_y"], EPS_Z])\n            grid[indices[:,0], indices[:,1], indices[:,2]] = w + points@H.T\n            args.field_out.parent.mkdir(parents=True, exist_ok=True)\n            np.savez_compressed(args.field_out, total_u=grid, N=args.N, H=H)\n            row["displacement_field"] = {"path": str(args.field_out), "sha256": hashlib.sha256(args.field_out.read_bytes()).hexdigest()}\n        del problem, sol_list, points, w'),
    ('"interpolation_power": 1}', '"interpolation_power": 1, "family": args.family}')])
edit('scripts/run_abaqus_binary.ps1', [
    ('gyroid_G[0-9]+', '(gyroid|primitive)_G[0-9]+')])
edit('scripts/abaqus_mesh_quality.py', [
    ('def diagnose(odb_path, expected_path, out):', 'def diagnose(odb_path, expected_path, out, nodal_out=None):'),
    ('    finally:\n        odb.close()',
     '        if nodal_out is not None:\n            physical = region(odb, "PHYSICAL", "nodeSets")\n            _, labels, _, u = read_field(frame.fieldOutputs["U"].getSubset(region=physical), nodal=True)\n            if not np.array_equal(labels, np.arange(1,len(mesh["points"])+1)):\n                raise ValueError("Incomplete nodal field")\n            take = np.unique(np.linspace(0,len(labels)-1,min(4096,len(labels))).astype(int))\n            nodal_out.parent.mkdir(parents=True,exist_ok=True)\n            np.savez_compressed(nodal_out, points=mesh["points"][take], u=u[take], labels=labels[take])\n            result["nodal_sample"] = {"points":len(take), "path":str(nodal_out), "sha256":hashlib.sha256(nodal_out.read_bytes()).hexdigest()}\n    finally:\n        odb.close()'),
    ('    a = p.parse_args()', '    p.add_argument("--nodal-out", type=Path)\n    a = p.parse_args()'),
    ('diagnose(a.odb, a.expected, a.out)', 'diagnose(a.odb, a.expected, a.out, a.nodal_out)')])
edit('tests/test_abaqus_binary.py', [], '''


def test_primitive_physical_definition_and_independent_jax_field():
    from geometry import primitive
    points = np.array([[0,0,0],[.25,.25,.25],[.5,.5,.5],[.125,.25,.375]])
    expected = np.array([3.,0.,-3.,0.])
    np.testing.assert_allclose(binary.implicit_numpy(points,"primitive"),expected,atol=1e-14)
    np.testing.assert_allclose(primitive(points),expected,atol=1e-14)
    np.testing.assert_allclose(binary.implicit_numpy(points+1,"primitive"),expected,atol=1e-14)


def test_primitive_fixed_translation_anchor_is_on_solid_bottom():
    points,cells = binary.linear_domain(8,c=.6,family="primitive")
    audit = binary.audit_linear(points,cells,.6,"primitive")
    assert audit["periodic_connected_components"] == 1
    assert .25 < audit["volume"] < .4
    points,cells = binary.quadratic_nodes(points,cells)
    _,_,bcs,_,_,bottom = prepare.constraints(points,"fixed")
    anchors = [label for label,comp,value in bcs if comp in (1,2) and label <= len(points)]
    assert len(set(anchors)) == 1 and anchors[0] in bottom
    assert not np.any(np.all(points==0,axis=1))
    with pytest.raises(ValueError,match="fixed macro lateral"):
        prepare.constraints(points,"relaxed_free")


def test_primitive_cache_cannot_be_reused_as_gyroid(tmp_path):
    points,cells = binary.linear_domain(4,c=.6,family="primitive")
    cache=tmp_path/"primitive.npz"
    np.savez(cache,points=points,cells=cells,geometry_N=4,c=.6,fe_refinement=0,family="primitive")
    report=prepare.prepare(tmp_path/"p",n=4,c=.6,cached_mesh=cache,family="primitive")
    assert report["model"] == "binary_sheet_primitive"
    assert report["case"] == "binary_primitive_G4_R0_C3D10_fixed"
    with pytest.raises(ValueError,match="family mismatch"):
        prepare.prepare(tmp_path/"g",n=4,c=.6,cached_mesh=cache)
''')
(O/'setup.py').write_bytes(Path(__file__).read_bytes())
print('Minimal family/anchor/input extensions written; original solvers untouched',flush=True)

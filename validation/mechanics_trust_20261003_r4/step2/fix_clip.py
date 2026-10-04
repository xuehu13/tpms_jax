from pathlib import Path
import hashlib,json,subprocess

P=Path('/home/xuehu/projects/tpms_jax');O=P/'validation/mechanics_trust_20261003_r4/step2'
f=P/'binary_gyroid.py';old=f.read_text()
backup=O/'source_before_clipfix/binary_gyroid.py';backup.parent.mkdir(exist_ok=False);backup.write_bytes(f.read_bytes())
needle='                ids = [node(p[:3]) for p in polygon]\n'
assert old.count(needle)==1
new='''                # An isolevel passing through a lattice vertex can produce
                # the same clipped polygon vertex twice after node canonicalization.
                # Keep its first cyclic occurrence; zero-area faces have no volume.
                ids = list(dict.fromkeys(node(p[:3]) for p in polygon))
                if len(ids) < 3:
                    continue
'''
f.write_text(old.replace(needle,new))
t=P/'tests/test_abaqus_binary.py'
t.write_text(t.read_text()+'''


def test_isolevel_at_lattice_vertex_preserves_closed_positive_cut_domain():
    # G=-.25000000000000006 at a vertex. The cut previously repeated one node
    # in a polygon and generated a zero-volume tetrahedron.
    points,cells=binary.linear_domain(48,c=.25,origins=[(0,7,19)])
    xyz=points[cells]
    det=np.linalg.det(np.transpose(xyz[:,1:]-xyz[:,:1],(0,2,1)))
    assert np.all(det>0)
    surface=binary.boundary_faces(cells)
    edges=np.sort(surface[:,((0,1),(1,2),(2,0))].reshape(-1,2),axis=1)
    _,counts=np.unique(edges,axis=0,return_counts=True)
    assert np.all(counts==2)
    # All eight cube vertices lie in/on the threshold band: exact volume is one cube.
    corners=np.array([(x,y,z) for x in (0,1) for y in (7,8) for z in (19,20)])/48
    assert np.max(np.abs(binary.gyroid_numpy(corners)))<=.25+1e-14
    np.testing.assert_allclose(det.sum()/6,1/48**3,rtol=1e-10)
''')
plan_path=O/'plan.json';initial=plan_path.read_bytes();(O/'plan_initial.json').write_bytes(initial)
plan=json.loads(initial)
plan['execution_revision']={'reason':'A repeated polygon vertex at an exact signed isolevel creates a zero-volume cut tet; deduplicate only coincident node IDs, retaining the physical domain',
                            'new_reference_directory':'thin_gyroid/G48_clipfix','old_failure_retained':True,
                            'same_c_material_loading_beta_and_FE_budget':True,
                            'binary_source_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),
                            'Windows_runner':'Use the current desktop PowerShell7 executable; PowerShell5 inherited incompatible module paths and failed before datacheck'}
plan_path.write_text(json.dumps(plan,indent=2)+'\n')
(O/'fix_clip.py').write_bytes(Path(__file__).read_bytes())
print('Coincident-vertex fix applied; target domain and mechanics budget unchanged')

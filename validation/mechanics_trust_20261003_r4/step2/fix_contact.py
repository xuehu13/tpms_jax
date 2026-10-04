"""Canonicalize arithmetic equality at global isolevels, not finite cut volumes."""
from pathlib import Path
import json,hashlib
P=Path('/home/xuehu/projects/tpms_jax');O=P/'validation/mechanics_trust_20261003_r4/step2'
f=P/'binary_gyroid.py';old=f.read_text()
backup=O/'source_before_contactfix/binary_gyroid.py';backup.parent.mkdir(exist_ok=False);backup.write_bytes(f.read_bytes())
needle='    widths = width_numpy(periodic_points,c)\n'
new='''    widths = width_numpy(periodic_points,c)
    # Analytic isolevels can differ by a few ulps at lattice vertices. Snap
    # only arithmetic equality globally so every adjacent tetrahedron agrees.
    equality_tol = 16*np.finfo(float).eps*np.maximum(1, np.maximum(np.abs(values), widths))
    for sign in (-1,1):
        on_level = np.abs(values-sign*widths) <= equality_tol
        values[on_level] = sign*widths[on_level]
'''
assert old.count(needle)==1
old=old.replace(needle,new)
needle='            if np.min(g-local_width) > 0 or np.max(g+local_width) < 0:\n'
new='''            # A point/edge/face-only contact with the closed band has zero
            # 3D measure; it does not generate a material volume element.
            if np.min(g-local_width) >= 0 or np.max(g+local_width) <= 0:
'''
assert old.count(needle)==1
f.write_text(old.replace(needle,new))
t=P/'tests/test_abaqus_binary.py'
t.write_text(t.read_text()+'''


def test_isolevel_vertex_contact_has_no_volume_and_nearby_cut_survives(monkeypatch):
    # The known G48 vertex differs from the exact .25 isolevel by only ulps.
    # Its cell is exterior except for this vertex; no volumetric solid exists.
    corners=np.array([(x,y,z) for x in (0,1) for y in (8,9) for z in (44,45)])/48
    assert np.min(binary.gyroid_numpy(corners))>=.25-4e-15
    _,cells=binary.linear_domain(48,c=.25,origins=[(0,8,44)])
    assert cells.size==0
    # Exact isolated contact is empty; a finite cut just below it is retained.
    def affine(xyz,family):
        return .25+np.asarray(xyz).sum(axis=-1)
    monkeypatch.setattr(binary,'implicit_numpy',affine)
    _,cells=binary.linear_domain(8,c=.25,origins=[(0,0,0)])
    assert cells.size==0
    def finite_cut(xyz,family):
        return .25-1e-3+np.asarray(xyz).sum(axis=-1)
    monkeypatch.setattr(binary,'implicit_numpy',finite_cut)
    points,cells=binary.linear_domain(8,c=.25,origins=[(0,0,0)])
    xyz=points[cells]
    det=np.linalg.det(np.transpose(xyz[:,1:]-xyz[:,:1],(0,2,1)))
    assert np.all(det>0)
    np.testing.assert_allclose(det.sum()/6,1e-9/6,rtol=1e-8)
''')
plan_path=O/'plan.json';plan=json.loads(plan_path.read_text())
plan['execution_revision']['contact_fix']={'reason':'Pure vertex contact at G=.25 differs by 7.8e-16; global field equality snapped within 16 machine eps scaled by field magnitude, zero-dimensional contacts excluded',
    'new_reference_directory':'thin_gyroid/G48_contactfix','first_two_failures_retained':True,'no_finite_volume_threshold_or_mechanics_change':True,
    'source_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'further_geometry_attempts_if_rejected':0}
plan_path.write_text(json.dumps(plan,indent=2)+'\n')
(O/'fix_contact.py').write_bytes(Path(__file__).read_bytes())
print('Machine-precision isolevel contact fix applied; no further geometry attempt if rejected')

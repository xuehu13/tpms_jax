import importlib.util
from pathlib import Path
import numpy as np
import pytest
import binary_gyroid as binary
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('prepare_binary',ROOT/'scripts'/'prepare_abaqus_binary.py')
prepare=importlib.util.module_from_spec(spec);spec.loader.exec_module(prepare)


def test_binary_boundary_topology_and_periodic_connectivity():
    points,cells=binary.linear_domain(8)
    audit=binary.audit_linear(points,cells)
    assert audit['periodic_connected_components'] == 1
    assert audit['connected_components'] == 3
    assert 0 < audit['volume'] < 1
    assert audit['min_detJ'] > 0
    # The clipped volume is not a count of retained background tetrahedra.
    assert not np.isclose(audit['volume']*6*8**3,round(audit['volume']*6*8**3))


def test_clipping_obeys_both_signed_levels():
    triangle=[np.array([0.,0.,0.,-2.]),np.array([1.,0.,0.,0.]),np.array([0.,1.,0.,2.])]
    clipped=binary.clip_polygon(binary.clip_polygon(triangle,-.5,True),.5,False)
    values=np.array(clipped)
    assert np.all(values[:,3] >= -.5)
    assert np.all(values[:,3] <= .5)
    # For this triangle the exact affine field is -2+2*x+4*y.
    np.testing.assert_allclose(values[:,3],-2+2*values[:,0]+4*values[:,1],atol=1e-15)


def test_c3d10_edge_order_and_affine_mapping():
    points,cells=prepare.uniform_domain(2)
    audit=binary.audit_linear(points,cells)
    assert audit['volume'] == pytest.approx(1.)
    quadratic,tets=binary.quadratic_nodes(points,cells)
    for i,(a,b) in enumerate(binary.TET_EDGES):
        np.testing.assert_allclose(quadratic[tets[:,4+i]],(points[cells[:,a]]+points[cells[:,b]])/2)
    # Independent quadratic shape functions must reproduce an affine map.
    barycentric=np.array([.1,.2,.3,.4])
    shape=np.r_[barycentric*(2*barycentric-1),[4*barycentric[a]*barycentric[b] for a,b in binary.TET_EDGES]]
    np.testing.assert_allclose(np.einsum('a,ead->ed',shape,quadratic[tets]),np.einsum('a,ead->ed',barycentric,points[cells]),atol=1e-15)


@pytest.mark.parametrize('lateral',['fixed','relaxed_free'])
def test_unstructured_constraints_cover_quadratic_boundary_nodes(lateral):
    points,cells=binary.linear_domain(4)
    points,cells=binary.quadratic_nodes(points,cells)
    controls,eq,bcs,pairs,top,bottom=prepare.constraints(points,lateral)
    assert len(top) > 0 and len(bottom) > 0
    eliminated={terms[0][:2] for terms in eq}
    for label in top:
        assert (label,3) in eliminated
    for slave,master,jump in pairs:
        np.testing.assert_allclose(points[slave-1]-points[master-1],jump,atol=1e-12)
    # Missing one periodic counterpart must be rejected before writing INP.
    missing=np.flatnonzero((points[:,0] == 0)&(points[:,1] > 0)&(points[:,1] < 1)&(points[:,2] > 0)&(points[:,2] < 1))[0]
    with pytest.raises(ValueError,match='Missing periodic master'):
        prepare.constraints(np.delete(points,missing,axis=0),lateral)


def test_prepare_uniform_has_no_void_material_or_subroutine(tmp_path):
    report=prepare.prepare(tmp_path,uniform=True)
    text=(tmp_path/(report['case']+'.inp')).read_text()
    assert '*ELEMENT, TYPE=C3D10' in text
    assert '*USER ELEMENT' not in text
    assert '*DEPVAR' not in text
    assert 'emin_ratio' not in report
    assert report['analytic']['energy'] == pytest.approx(.0006730769230769231)


def test_incompatible_cached_geometry_is_rejected(tmp_path):
    path=tmp_path/'cache.npz'
    np.savez(path,points=np.zeros((1,3)),cells=np.zeros((1,4),int),geometry_N=16,c=binary.C)
    with pytest.raises(ValueError,match='matching geometry_N'):
        prepare.prepare(tmp_path,n=8,cached_mesh=path)


def test_volume_mesher_preserves_domain_and_periodic_caps():
    pytest.importorskip('gmsh')
    points,cells=binary.linear_domain(4)
    before=binary.audit_linear(points,cells)
    result,tets,metadata=binary.remesh_domain(points,cells)
    after=binary.audit_linear(result,tets)
    assert after['volume'] == pytest.approx(before['volume'],abs=1e-10)
    assert after['periodic_cap_triangles'] == before['periodic_cap_triangles']
    assert after['periodic_connected_components'] == 1


def test_volume_audit_rejects_inverted_element():
    points,cells=prepare.uniform_domain()
    cells=cells.copy();cells[0,[1,2]]=cells[0,[2,1]]
    with pytest.raises(ValueError,match='Nonpositive'):
        binary.audit_linear(points,cells)


def test_g48_shared_intersections_do_not_create_nonmanifold_cells():
    # This cell previously produced three faces with incidence >2, because
    # opposite edge traversal generated nearly coincident contour vertices.
    points,cells=binary.linear_domain(48,origins=[(39,0,13)])
    surface=binary.boundary_faces(cells)
    edges=np.sort(surface[:,((0,1),(1,2),(2,0))].reshape(-1,2),axis=1)
    _,count=np.unique(edges,axis=0,return_counts=True)
    assert np.all(count == 2)


def test_bulk_reader_preserves_label_ip_pairs_under_permutation():
    sys.path.insert(0,str(ROOT/'scripts'))
    from extract_abaqus_binary import read_field
    labels=np.repeat([1,2],4);ips=np.tile([1,2,3,4],2)
    expected=np.column_stack((100*labels+ips,10*labels-ips))
    permutation=[7,0,5,2,1,6,3,4]
    block=SimpleNamespace(data=expected[permutation],elementLabels=labels[permutation],
                          integrationPoints=ips[permutation],instance=SimpleNamespace(name='SOLID'))
    instance,observed_labels,observed_ips,observed=read_field(SimpleNamespace(bulkDataBlocks=[block]))
    assert instance == 'SOLID'
    np.testing.assert_array_equal(observed_labels,labels)
    np.testing.assert_array_equal(observed_ips,ips)
    np.testing.assert_array_equal(observed,expected)


@pytest.mark.parametrize('c',[.48,.60])
def test_threshold_specific_binary_cache_and_manifest(tmp_path,c):
    points,cells=binary.linear_domain(4,c=c)
    cache=tmp_path/'mesh.npz'
    np.savez(cache,points=points,cells=cells,geometry_N=4,c=c,fe_refinement=0)
    manifest=prepare.prepare(tmp_path/'case',n=4,cached_mesh=cache,c=c)
    assert manifest['c'] == c
    assert manifest['geometry']['volume'] == pytest.approx(binary.audit_linear(points,cells,c)['volume'])
    with pytest.raises(ValueError,match='matching geometry_N and c'):
        prepare.prepare(tmp_path/'wrong',n=4,cached_mesh=cache,c=binary.C)



def test_primitive_physical_definition_and_independent_jax_field():
    from geometry import primitive
    points = np.array([[0,0,0],[.25,.25,.25],[.5,.5,.5],[.125,.25,.375]])
    expected = np.array([3.,0.,-3.,0.])
    np.testing.assert_allclose(binary.implicit_numpy(points,"primitive"),expected,atol=1e-14)
    import jax
    jax.config.update("jax_enable_x64", True)
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

"""Freeze references and evaluate each HEX8 field at its own actual Gauss points."""
from pathlib import Path
import json,sys,time,hashlib,shutil
import numpy as np
from scipy.special import expit
from hex8_model import B,R,PlatenXY8,interpolate,preflight,sha,jax,make_density_hyperelastic_problem
from surface_distance import PeriodicSurfaceDistance

def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def initialize():
    assert not (B/'protocol.json').exists()
    refs={k:{'case_dir':str(R/('validation/platen_xy_diverse04_20261012_r54' if k=='diverse_04' else 'validation/platen_xy_diverse_batch_20261012_r55')/k)} for k in ['diverse_28','diverse_02','diverse_04','diverse_05']}
    base=json.loads((R/'validation/platen_xy_diverse_batch_20261012_r55/protocol.json').read_text())
    protocol={**base,'stage':'HEX8_JAX_only_authorized','cases':list(refs),'references':refs,'element':'HEX8','Gauss_points_per_element':8,
      'quadrature_order':2,'mass':'positive row-sum lumping with exact projected affine inertia','future_dynamic_jobs':4,
      'execution_order':list(refs),'no_new_abaqus':True,'request':'Test the current common-end-rim platen XY model using standard full-integration HEX8; reuse frozen shell and HEX27 results',
      'contact_control':'Original frozen rim points/area_weight/side/phi/labels unchanged; only Q1 interpolation node_ids/shape rebuilt',
      'per_geometry_budget_seconds':{'preparation':300,'jax_build_compile_checks_solve':2400,'analysis':120},
      'maximum_batch_seconds':11280,'budget_rule':'Each geometry one physical path, no parameter retries. Separate300s preparation +2400s JAX including build/compile/stability checks; preserve partial paths on failure/budget.',
      'interpretation':'Practical HEX8/8QP/row-sum versus HEX27/27QP/HRZ package comparison, not isolated interpolation-order causality',
      'authorization':'User requested HEX8 testing of diverse28/02/04/05 under latest loading, JAX only',
      'protected_source_sha256':{str(p):sha(p) for p in [R/'hyperelastic_fem.py',R/'pbc.py',R/'fem.py',R/'scripts/thin_target_explicit.py',R/'validation/platen_xy_G_20261010_r51/tools/platen_model.py']}}
    write(B/'protocol.json',protocol)
    write(B/'status.json',{'stage':'prepared_for_HEX8','active_jobs':[],'queue':list(refs),'cases':{},'gradients':'paused','heartbeat':'paused','new_Abaqus_jobs':0})

def prepare(kind):
    start=time.time();proto=json.loads((B/'protocol.json').read_text());old=Path(proto['references'][kind]['case_dir']);E=B/kind
    assert not (E/'input/protocol.json').exists()
    cfg=json.loads((old/'input/protocol.json').read_text())
    surface=Path(cfg['source_shell_surface'])
    with np.load(surface) as z:v=z['surface_vertices'];faces=z['surface_triangles']
    distance=PeriodicSurfaceDistance(v,faces);collected={}
    def field(p):
        X=np.asarray(p.physical_quad_points);w=np.asarray(p.fe.JxW);d=distance.query(X)
        phi=expit((cfg['thickness_mm']/cfg['L_mm']/2-d)/(cfg['interface_10_90_mm']/cfg['L_mm']/(2*np.log(9))))
        collected.update(physical_quad_points=X,JxW=w,distance=d,rho=phi);return phi
    with jax.default_device(jax.devices('cpu')[0]):
        p=make_density_hyperelastic_problem(cfg['N'],rho_quad=field,eta=cfg['eta'],periodic_axes=(0,1,2),element_degree=1,quadrature_order=2,material_model='objective_void')
    assert p.fe.num_quads==8;np.testing.assert_array_equal(p.rho,collected['rho'])
    (E/'input').mkdir(parents=True);(E/'checks').mkdir()
    gauss=E/'input/gauss_field.npz';np.savez_compressed(gauss,**collected)
    original_contact=Path(cfg['contact_cache_path'])
    with np.load(original_contact) as z:contact=dict(z)
    ids,shape=interpolate(contact['points'],p);newcontact={**contact,'node_ids':ids,'shape':shape}
    for k in ['points','side','area_weight','phi','shell_node_labels']:np.testing.assert_array_equal(contact[k],newcontact[k])
    dest=E/'input/rimline_contact.npz';np.savez_compressed(dest,**newcontact)
    cfg.update(case_id=kind+'_HEX8_platen_xy_r56',element='HEX8',element_degree=1,Gauss_points_per_cell=8,quadrature_order=2,
      stage='HEX8 common midsurface-end platen20pct; compare to frozen r54/r55 references, no new design AD',
      original_Gauss_field=str(gauss),original_Gauss_sha256=sha(gauss),contact_cache_path=str(dest),contact_cache_sha256=sha(dest),
      jax_contact='Same frozen rim points and weights; own Q1 force interpolation; same kn320',
      mass='positive row_sum',reference_case_dir=str(old),reference_contact_sha256=sha(original_contact),
      budget_seconds=proto['per_geometry_budget_seconds'],no_new_abaqus=True)
    ex=PlatenXY8(p,cfg,newcontact);check=preflight(p,ex,newcontact)
    vf=float(np.sum(collected['rho']*collected['JxW'])/np.sum(collected['JxW']))
    with np.load(json.loads((old/'input/protocol.json').read_text())['original_Gauss_field']) as z:oldvf=float(np.sum(z['rho']*z['JxW'])/np.sum(z['JxW']))
    check.update(kind=kind,preparation_wall_seconds=time.time()-start,occupancy_volume_fraction=vf,HEX27_occupancy_volume_fraction=oldvf,
      occupancy_relative_change=vf/oldvf-1,total_mass_tonne=float(np.sum(ex.nodem)*cfg['L_mm']),
      contact_area_mm2=[float(contact['area_weight'][contact['side']==s].sum()*cfg['L_mm']**2) for s in [0,1]],
      same_contact_points_weights_verified=True,actual_own_Gauss_equality=True,
      reference_case_dir=str(old),reference_source_sha256={str(x):sha(x) for x in [old/'input/protocol.json',old/'jax/result.json',surface,original_contact]})
    assert time.time()-start<proto['per_geometry_budget_seconds']['preparation']
    write(E/'checks/preparation.json',check);write(E/'input/protocol.json',cfg)
    print(json.dumps({k:v for k,v in check.items() if k not in ['zero_state','reference_source_sha256','periodic_source_substitution']}),flush=True)

if __name__=='__main__':initialize() if sys.argv[1]=='init' else prepare(sys.argv[1])

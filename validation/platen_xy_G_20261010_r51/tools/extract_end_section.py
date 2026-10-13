"""Read local static end-section ODB; do not infer G accuracy from this test."""
from pathlib import Path
import json,sys
import numpy as np
from odbAccess import openOdb
odbfile,cfgfile,out=map(Path,sys.argv[1:]);cfg=json.loads(cfgfile.read_text())
if out.exists():raise FileExistsError(out)
def data(v):
    try:return np.array(v.dataDouble,dtype=float)
    except Exception:return np.array(v.data,dtype=float)
odb=openOdb(str(odbfile.resolve()),readOnly=True)
try:
    step=odb.steps['PRESS'];nv=cfg['physical_nodes'];rp=cfg['labels']['top']
    xyz={n.label:np.array(n.coordinates,dtype=float) for inst in odb.rootAssembly.instances.values() for n in inst.nodes if n.label<=nv}
    rows=[];field_keys=[];saved_u=[]
    base=np.array(cfg['base_nodes']);contact=np.array(cfg['contact_nodes']);all_top=np.array(cfg['all_top_nodes'])
    for frame in step.frames:
        du={v.nodeLabel:data(v) for v in frame.fieldOutputs['U'].values}
        rf={v.nodeLabel:data(v) for v in frame.fieldOutputs['RF'].values}
        if rp not in rf:continue
        force=float(-rf[rp][2]);bottom=float(sum(rf[int(n)][2] for n in base))
        plate_z=10+du[rp][2];current={int(n):xyz[int(n)]+du[int(n)] for n in all_top}
        ll,rr,mm=[cfg['labels'][s] for s in ('tip_left','tip_right','tip_mid')]
        cross=current[rr]-current[ll]
        normal_fields=[(name,f) for name,f in frame.fieldOutputs.items() if name.startswith('CNORMF')]
        fn={int(n):np.zeros(3) for n in contact}
        for name,field in normal_fields:
            for v in field.values:
                if v.nodeLabel in fn:fn[v.nodeLabel]+=data(v)
        total_cf=float(-sum(v[2] for v in fn.values()))
        direct_x=float(sum(current[n][0]*(-v[2]) for n,v in fn.items())/total_cf) if total_cf>1e-20 else None
        base_x=float(sum(xyz[int(n)][0]*rf[int(n)][2] for n in base)/bottom) if abs(bottom)>1e-20 else None
        rows.append({'time':float(frame.frameValue),'compression':float(-du[rp][2]/10),'force_N':force,'bottom_force_N':bottom,
            'tip_mid_U1_mm':float(du[mm][0]),'tip_mid_U3_mm':float(du[mm][2]),
            'top_cross_section_rotation_rad':float(np.arctan2(cross[2],cross[0])),
            'max_contact_node_penetration_mm':float(max(0,max(current[int(n)][2]-plate_z for n in contact))),
            'max_physical_cap_penetration_mm':float(max(0,max(p[2]-plate_z for p in current.values()))),
            'contact_normal_force_N':total_cf,'contact_force_x_centroid_mm':direct_x,
            'contact_force_lever_from_current_mid_mm':None if direct_x is None else direct_x-current[mm][0],
            'base_reaction_inferred_force_x_mm':base_x,
            'base_inferred_lever_from_current_mid_mm':None if base_x is None else base_x-current[mm][0],
            'base_RF1_sum_N':float(sum(rf[int(n)][0] for n in base))})
        field_keys=list(frame.fieldOutputs.keys());saved_u.append(np.array([du[n] for n in sorted(xyz)]))
    regions=list(step.historyRegions.values());energy={}
    for name in ('ALLIE','ALLSE','ALLWK','ALLAE'):
        found=[r for r in regions if name in r.historyOutputs]
        if found:energy[name]=np.array(found[0].historyOutputs[name].data).tolist()
    result={'case':cfg['case'],'complete':bool(abs(rows[-1]['compression']-.02)<1e-8),'status':'static_local_benchmark',
        'scope':cfg['scope'],'field_output_keys':field_keys,'path':rows,'energy':energy,
        'maximum_reaction_imbalance_N':max(abs(r['bottom_force_N']-r['force_N']) for r in rows),
        'maximum_contact_node_penetration_mm':max(r['max_contact_node_penetration_mm'] for r in rows),
        'maximum_physical_cap_penetration_mm':max(r['max_physical_cap_penetration_mm'] for r in rows),
        'no_G_change':True,'no20pct':True,'new_design_AD':False}
    np.savez_compressed(out.with_name('field.npz'),labels=sorted(xyz),xyz_mm=np.array([xyz[n] for n in sorted(xyz)]),u_mm=saved_u[-1],path_u_mm=np.array(saved_u))
finally:odb.close()
out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('path','energy')},indent=2))

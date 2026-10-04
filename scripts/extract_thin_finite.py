"""Read-only matched finite-strain XYZ shell extraction with Abaqus Python."""
from pathlib import Path
import argparse,hashlib,json,math
import numpy as np


def data(v):
    try:return v.dataDouble
    except Exception:return v.data


def extract(odb_path,input_path,out):
    from odbAccess import openOdb
    cfg=json.loads(input_path.read_text());assert not out.exists()
    odb=openOdb(path=str(odb_path.resolve()),readOnly=True)
    try:
        step=odb.steps[cfg['step_name']];frame=step.frames[-1]
        assert abs(frame.frameValue-1)<1e-8
        u={v.nodeLabel:np.array(data(v),dtype=float) for v in frame.fieldOutputs['U'].values}
        rf={v.nodeLabel:np.array(data(v),dtype=float) for v in frame.fieldOutputs['RF'].values}
        ur={v.nodeLabel:np.array(data(v),dtype=float) for v in frame.fieldOutputs['UR'].values}
        xyz={n.label:np.array(n.coordinates,dtype=float) for inst in odb.rootAssembly.instances.values() for n in inst.nodes}
        qz=cfg['macro_control_label'];labels=sorted(n for n in xyz if n!=qz);delta=float(u[qz][2]);F=float(rf[qz][2])
        energies={k:[float(r.historyOutputs[k].data[-1][1]) for r in step.historyRegions.values() if k in r.historyOutputs]
                  for k in ('ALLSE','ALLAE','ALLWK')}
        assert all(len(v)==1 for v in energies.values());energies={k:v[0] for k,v in energies.items()}
        groups={};L=cfg['cell_size_mm']
        for n in labels:
            x=xyz[n]/L;key=tuple(np.round(np.where(np.isclose(x,1,atol=1e-9),0,x),9));groups.setdefault(key,[]).append(n)
        pbc=rot=0.;a=cfg['compression']
        for members in groups.values():
            root=min(members)
            for n in members:
                jump=np.array([0.,0.,-a*(xyz[n][2]-xyz[root][2])])
                pbc=max(pbc,float(np.max(np.abs(u[n]-u[root]-jump))))
                rot=max(rot,float(np.max(np.abs(ur[n]-ur[root]))))
        sth=[]
        if 'STH' in frame.fieldOutputs:sth=[float(data(v)) for v in frame.fieldOutputs['STH'].values]
        rows=[]
        for name,st in odb.steps.items():
            for i in range(1,len(st.frames)):
                fr=st.frames[i]
                du={v.nodeLabel:np.array(data(v),dtype=float) for v in fr.fieldOutputs['U'].values}
                dr={v.nodeLabel:np.array(data(v),dtype=float) for v in fr.fieldOutputs['RF'].values}
                rows.append({'step':name,'time':float(fr.frameValue),'compression':-float(du[qz][2])/L,'Fz_N':float(dr[qz][2])})
        U=energies['ALLSE'];total=U+energies['ALLAE'];work=energies['ALLWK']
        comp=np.r_[0.,[r['compression'] for r in rows]];force=np.r_[0.,[r['Fz_N'] for r in rows]]
        macro_work=float(np.sum(-L*.5*(force[1:]+force[:-1])*np.diff(comp)))
        work_gap=abs(total-work)/max(abs(work),1e-30)
        stored_U_tolerance=max(1e-8,1e-7*abs(a*L))
        checks={'complete_step':True,'node_count':len(labels)==cfg['physical_node_count'],
                'finite':bool(all(np.isfinite(u[n]).all() for n in labels) and all(math.isfinite(x) for x in (F,U,work))),
                'load_delta_stored_float_precision':abs(delta+a*L)<=max(1e-8,1e-7*abs(a*L)),
                'XYZ_periodic_stored_float_precision':pbc<=stored_U_tolerance,
                'XYZ_rotation_periodic_rad_le_1e-8':rot<=1e-8,
                'relative_total_energy_work_le_1pct':work_gap<=.01,
                'macro_work_vs_ALLWK_le_0p1pct':abs(macro_work-work)<=.001*max(abs(work),1e-30),
                'artificial_energy_fraction_le_1pct':abs(energies['ALLAE'])<=.01*abs(work),
                'positive_current_shell_thickness':bool(sth) and min(sth)>0}
        field=out.with_name('shell_field.npz');np.savez_compressed(field,labels=labels,xyz_mm=np.array([xyz[n] for n in labels]),
                                                                u_mm=np.array([u[n] for n in labels]),ur=np.array([ur[n] for n in labels]))
        result={'status':'ok' if all(checks.values()) else 'check_failed','compression':a,'checks':checks,'Fz_N':F,'delta_mm':delta,
                'energy_N_mm':U,'total_energy_N_mm':total,'energies_N_mm':energies,
                'artificial_energy_fraction':energies['ALLAE']/work,'periodic_U_error_mm':pbc,'periodic_UR_error_rad':rot,
                'stored_U_check_tolerance_mm':stored_U_tolerance,
                'relative_total_energy_work_gap':work_gap,'diagnostic_work_within_0p1pct':work_gap<=.001,
                'macro_control_trapezoid_work_N_mm':macro_work,
                'quality_scope':'feasibility reference; <=1% energy balance screen, not strict nonlinear shell energy certification',
                'current_shell_thickness_mm':{'min':min(sth) if sth else None,'max':max(sth) if sth else None},
                'force_path':rows,'field_sha256':hashlib.sha256(field.read_bytes()).hexdigest(),
                'extractor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'odb_path':str(odb_path.resolve()),'input_sha256':hashlib.sha256(input_path.read_bytes()).hexdigest()}
    finally:odb.close()
    with out.open('x') as f:f.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='force_path'},indent=2))
    if not all(checks.values()):raise RuntimeError('Shell checks failed; output retained')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('odb',type=Path);p.add_argument('input',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();extract(a.odb,a.input,a.out)

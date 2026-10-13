"""Read only one completed, matched linear shell ODB using Abaqus Python."""
from pathlib import Path
import argparse, hashlib, json, math


def data(value):
    try:return value.dataDouble
    except Exception:return value.data


def extract(odb_path, input_path, output):
    from odbAccess import openOdb
    cfg=json.loads(input_path.read_text());assert not output.exists()
    odb=openOdb(path=str(odb_path.resolve()),readOnly=True)
    try:
        step=odb.steps['COMPRESSION'];frame=step.frames[-1]
        assert abs(frame.frameValue-1)<1e-8
        u={v.nodeLabel:[float(x) for x in data(v)] for v in frame.fieldOutputs['U'].values}
        rf={v.nodeLabel:[float(x) for x in data(v)] for v in frame.fieldOutputs['RF'].values}
        xyz={n.label:[float(x) for x in n.coordinates] for inst in odb.rootAssembly.instances.values() for n in inst.nodes}
        qz=cfg['macro_control_label'];F=rf[qz][2];delta=u[qz][2]
        physical=[{'label':n,'xyz_mm':p,'U_mm':u[n]} for n,p in xyz.items() if n!=qz]
        Fb=sum(rf.get(row['label'],[0,0,0])[2] for row in physical if abs(row['xyz_mm'][2])<1e-8)
        energies={name:[float(r.historyOutputs[name].data[-1][1]) for r in step.historyRegions.values() if name in r.historyOutputs]
                  for name in ('ALLSE','ALLAE','ALLWK')}
        assert all(len(vals)==1 for vals in energies.values()),energies
        energies={name:vals[0] for name,vals in energies.items()};U=energies['ALLSE']
        expected_delta=cfg['epsilon_z']*cfg['cell_size_mm']
        checks={'complete_step':True,'node_count':len(physical)==cfg['physical_node_count'],
            'finite':all(math.isfinite(x) for row in physical for x in row['U_mm']) and all(math.isfinite(x) for x in [F,Fb,U]),
            'load_delta':abs(delta-expected_delta)<1e-9,
            'relative_total_energy_work_le_1e-5':abs(U+energies['ALLAE']-.5*F*delta)<=abs(U)*1e-5,
            'artificial_energy_fraction_le_1pct':abs(energies['ALLAE'])<=.01*abs(energies['ALLWK'])}
        if not cfg.get('diagnostic_xyz',False):checks['relative_force_balance_le_1e-5']=abs(F+Fb)<=abs(F)*1e-5
        ur={v.nodeLabel:[float(x) for x in data(v)] for v in frame.fieldOutputs['UR'].values}
        lines=(input_path.parent/'shell.inp').read_text().splitlines(); gaps=[]; rgaps=[]
        for i,line in enumerate(lines):
            if line.strip().lower()!='*equation':continue
            nt=int(lines[i+1].strip()); fields=[]; j=i+2
            while len(fields)<3*nt:
                fields.extend([x.strip() for x in lines[j].split(',') if x.strip()]); j+=1
            terms=[(int(fields[k]),int(fields[k+1]),float(fields[k+2])) for k in range(0,3*nt,3)]
            value=sum(c*(u[n][v-1] if v<=3 else ur[n][v-4]) for n,v,c in terms)
            (gaps if terms[0][1]<=3 else rgaps).append(abs(value))
        maxu=max(gaps,default=0.); maxr=max(rgaps,default=0.)
        checks['periodic_translation_gap_le_1e-8_mm']=maxu<=1e-8
        checks['periodic_rotation_gap_le_1e-8_rad']=maxr<=1e-8
        result_periodic={'translation_max_mm':maxu,'rotation_max_rad':maxr,'equation_count':len(gaps)+len(rgaps)}
        result={'periodic':result_periodic,'status':'ok' if all(checks.values()) else 'check_failed','checks':checks,
            'Fz_top_N':F,'Fz_bottom_N':Fb,'delta_mm':delta,'energy_N_mm':U,'energies_N_mm':energies,
            'stiffness_N_per_mm':F/delta,'effective_axial_MPa':F/(cfg['cell_size_mm']**2*cfg['epsilon_z']),
            'artificial_energy_fraction':energies['ALLAE']/energies['ALLWK'],
            'total_energy_N_mm':U+energies['ALLAE'],
            'extractor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'nodes':physical,'odb_path':str(odb_path.resolve()),'input_sha256':hashlib.sha256(input_path.read_bytes()).hexdigest()}
    finally:odb.close()
    with output.open('x') as stream:stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='nodes'},indent=2))
    if not all(checks.values()):raise RuntimeError('Shell consistency failed; result retained')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('odb',type=Path);p.add_argument('input',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();extract(a.odb,a.input,a.out)

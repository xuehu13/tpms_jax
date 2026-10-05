"""Read-only Explicit curve/energy extraction for the matched periodic shell."""
from pathlib import Path
import argparse,hashlib,json
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
        regions=list(step.historyRegions.values())
        control=[r for r in regions if 'U3' in r.historyOutputs and 'RF3' in r.historyOutputs]
        assert len(control)==1
        displacement=np.asarray(control[0].historyOutputs['U3'].data,dtype=float)
        force=np.asarray(control[0].historyOutputs['RF3'].data,dtype=float)
        np.testing.assert_allclose(displacement[:,0],force[:,0],rtol=0,atol=1e-10)
        time=displacement[:,0];a=-displacement[:,1]/cfg['cell_size_mm']
        histories={}
        for name in ['ALLIE','ALLSE','ALLKE','ALLAE','ALLVD','ALLWK','ETOTAL']:
            values=[np.asarray(r.historyOutputs[name].data,dtype=float) for r in regions if name in r.historyOutputs]
            assert len(values)==1,name
            histories[name]=np.interp(time,values[0][:,0],values[0][:,1])
        loading=(time<=cfg['load_time_seconds'])&(a>=.01)
        ie=histories['ALLIE'];wk=histories['ALLWK']
        ratio=histories['ALLKE']/np.maximum(np.abs(ie),1e-30)
        weights=np.r_[0.,np.diff(time)]
        fraction=float(np.sum(weights[loading]*(ratio[loading]<=.05))/np.sum(weights[loading]))
        hold=time>=cfg['load_time_seconds']+cfg['hold_time_seconds']*.5
        macro_work=float(np.sum(.5*(force[1:,1]+force[:-1,1])*np.diff(displacement[:,1])))
        # Boundary work is integral RF du: RF and compression displacement are both negative.
        et=histories['ETOTAL'];scale=max(float(np.max(np.abs(wk))),1e-30)
        drift=float(np.max(np.abs(et-et[0]))/scale)
        ae=float(np.max(np.abs(histories['ALLAE']))/scale)
        u={v.nodeLabel:np.asarray(data(v),dtype=float) for v in frame.fieldOutputs['U'].values}
        ur={v.nodeLabel:np.asarray(data(v),dtype=float) for v in frame.fieldOutputs['UR'].values}
        xyz={n.label:np.asarray(n.coordinates,dtype=float) for inst in odb.rootAssembly.instances.values() for n in inst.nodes}
        q=cfg['macro_control_label'];labels=sorted(n for n in xyz if n!=q);groups={};L=cfg['cell_size_mm']
        for n in labels:
            x=xyz[n]/L;key=tuple(np.round(np.where(np.isclose(x,1,atol=1e-9),0,x),9));groups.setdefault(key,[]).append(n)
        periodic=rotation=0.
        for members in groups.values():
            r=min(members)
            for n in members:
                jump=np.array([0.,0.,-.2*(xyz[n][2]-xyz[r][2])])
                periodic=max(periodic,float(np.max(np.abs(u[n]-u[r]-jump))))
                rotation=max(rotation,float(np.max(np.abs(ur[n]-ur[r]))))
        checks={'complete':abs(frame.frameValue-cfg['total_time_seconds'])<=1e-7*cfg['total_time_seconds'],
                'finite':bool(np.isfinite(force).all() and all(np.isfinite(u[n]).all() for n in labels)),
                'target_compression':abs(a[-1]-.2)<=1e-7,
                'physical_node_count':len(labels)==cfg['physical_node_count'],
                'XYZ_periodic_mm_le_1e-6':periodic<=1e-6,
                'XYZ_rotation_rad_le_1e-6':rotation<=1e-6,
                'global_energy_drift_le_1pct':drift<=.01,
                'most_loading_time_KE_below_5pct':fraction>=.95,
                'terminal_KE_below_5pct':ratio[-1]<=.05,
                'artificial_energy_le_5pct_work':ae<=.05}
        checks={name:bool(value) for name,value in checks.items()}
        field=out.with_name('shell_field.npz')
        np.savez_compressed(field,labels=labels,xyz_mm=np.array([xyz[n] for n in labels]),
                            u_mm=np.array([u[n] for n in labels]),ur=np.array([ur[n] for n in labels]))
        rows=[{'time':float(t),'compression':float(x),'Fz_N':float(f),**{k:float(histories[k][i]) for k in histories}}
              for i,(t,x,f) in enumerate(zip(time,a,force[:,1]))]
        result={'status':'ok' if all(checks.values()) else 'complete_diagnostic','checks':checks,'compression':.2,
                'Fz_N':float(force[-1,1]),'hold_mean_Fz_N':float(np.mean(force[hold,1])),
                'hold_min_max_Fz_N':[float(np.min(force[hold,1])),float(np.max(force[hold,1]))],
                'energy_N_mm':float(histories['ALLSE'][-1]),'energies_N_mm':{k:float(v[-1]) for k,v in histories.items()},
                'KE_fraction_of_IE_terminal':float(ratio[-1]),'KE_fraction_of_IE_loading_p95':float(np.quantile(ratio[loading],.95)),
                'loading_time_fraction_KE_below_5pct':fraction,'global_energy_drift_relative_work':drift,
                'artificial_energy_relative_max_work':ae,'macro_work_from_RF_U_N_mm':macro_work,
                'macro_work_relative_ALLWK_difference':abs(macro_work-wk[-1])/max(abs(wk[-1]),1e-30),
                'periodic_U_error_mm':periodic,'periodic_UR_error_rad':rotation,
                'force_path':rows,'no_contact_diagnostic':cfg['no_contact'],
                'quality_scope':'energy and inertia screen only; loading-rate/geometry-contact validity remain separate',
                'odb_path':str(odb_path.resolve()),'input_sha256':hashlib.sha256(input_path.read_bytes()).hexdigest(),
                'field_sha256':hashlib.sha256(field.read_bytes()).hexdigest(),
                'extractor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    finally:odb.close()
    out.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in result.items() if k!='force_path'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('odb',type=Path);p.add_argument('input',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();extract(a.odb,a.input,a.out)

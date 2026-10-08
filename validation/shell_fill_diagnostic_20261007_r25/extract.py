"""Read-only Abaqus r25 histories and sampled constraint/volume evidence.

Keeps incomplete/aborted paths. Does not call a partial path a 20% result,
or a dynamic snap a quasi-static validation.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
def data(v):
    try:return v.dataDouble
    except Exception:return v.data
def extract(odb_path,cfg_path,out):
    from odbAccess import openOdb
    cfg=json.loads(cfg_path.read_text());assert not out.exists()
    odb=openOdb(path=str(odb_path.resolve()),readOnly=True)
    try:
        step=odb.steps[cfg['step_name']]
        regs=step.historyRegions
        ctrl=[r for r in regs.values() if 'U3' in r.historyOutputs and 'RF3' in r.historyOutputs]
        assert len(ctrl)==1
        u=np.asarray(ctrl[0].historyOutputs['U3'].data,float)
        f=np.asarray(ctrl[0].historyOutputs['RF3'].data,float)
        np.testing.assert_allclose(u[:,0],f[:,0],atol=1e-12,rtol=0)
        t=u[:,0];a=-u[:,1]/10.;energies={};groups={}
        globalreg=[r for name,r in regs.items() if name.lower().startswith('assembly')]
        assert len(globalreg)==1,list(regs.keys())
        for name in ['ALLIE','ALLSE','ALLKE','ALLAE','ALLVD','ALLWK','ETOTAL']:
            v=np.asarray(globalreg[0].historyOutputs[name].data,float)
            energies[name]=np.interp(t,v[:,0],v[:,1])
        for region,r in regs.items():
            if region.lower().startswith('assembly') or 'U3' in r.historyOutputs:continue
            if 'ALLIE' in r.historyOutputs:
                groups[region]={name:[[float(x),float(y)] for x,y in o.data] for name,o in r.historyOutputs.items()}
        insts=odb.rootAssembly.instances.values()
        xyz={n.label:np.array(n.coordinates,float) for inst in insts for n in inst.nodes}
        L=10.;labels=range(1,cfg['physical_node_count']+1);periodic_groups={}
        for n in labels:
            key=tuple(np.round(np.where(np.isclose(xyz[n]/L,1.,atol=1e-9),0.,xyz[n]/L),9))
            periodic_groups.setdefault(key,[]).append(n)
        frame_rows=[]; minvol=None;max_periodic=0.;max_rotation=0.
        for frame in step.frames:
            d={v.nodeLabel:np.asarray(data(v),float) for v in frame.fieldOutputs['U'].values}
            rot={v.nodeLabel:np.asarray(data(v),float) for v in frame.fieldOutputs['UR'].values} if 'UR' in frame.fieldOutputs else {}
            s=float(np.clip(frame.frameValue/.004,0.,1.))
            has_actual_control=cfg['macro_control_label'] in d
            compression=-float(d[cfg['macro_control_label']][2])/10. if has_actual_control else .2*(10*s**3-15*s**4+6*s**5)
            pe=0.;re=0.
            for n,r,dz in cfg['host_periodic_relations']:
                pe=max(pe,float(np.max(np.abs(d[n]-d[r]-np.array([0.,0.,-compression*dz])))))
            shell_pe=0.
            for members in periodic_groups.values():
                r=min(members)
                for n in members:
                    shell_pe=max(shell_pe,float(np.max(np.abs(d[n]-d[r]-np.array([0.,0.,-compression*(xyz[n][2]-xyz[r][2])])))))
                    if n in rot and r in rot:re=max(re,float(np.max(np.abs(rot[n]-rot[r]))))
            vol=[float(data(v)) for v in frame.fieldOutputs['EVOL'].values] if 'EVOL' in frame.fieldOutputs else []
            vmin=min(vol) if vol else None
            if vmin is not None:minvol=vmin if minvol is None else min(minvol,vmin)
            max_periodic=max(max_periodic,pe,shell_pe);max_rotation=max(max_rotation,re)
            frame_rows.append({'time':float(frame.frameValue),'compression':compression,
                'macro_compression_source':'actual_control_U3' if has_actual_control else 'unchanged_analytic_amplitude',
                'host_PBC_error_mm':pe,'shell_PBC_error_mm':shell_pe,'shell_UR_PBC_error_rad':re,
                'shell_pin_displacement_mm':float(np.max(np.abs(d[cfg['pin_node_label']]))),
                'min_host_EVOL_mm3':vmin,'nonpositive_host_EVOL_count':sum(v<=0 for v in vol)})
        work=float(np.sum(.5*(f[1:,1]+f[:-1,1])*np.diff(u[:,1])))
        wk=energies['ALLWK'];scale=max(float(np.max(np.abs(wk))),1e-30)
        et=energies['ETOTAL'];drift=float(np.max(np.abs(et-et[0]))/scale)
        ae=float(np.max(np.abs(energies['ALLAE']))/scale)
        rows=[{'time':float(x),'compression':float(y),'Fz_N':float(z),
               **{k:float(v[i]) for k,v in energies.items()}} for i,(x,y,z) in enumerate(zip(t,a,f[:,1]))]
        result={'status':'extracted_dynamic_diagnostic','last_compression':float(a[-1]),
                'last_history_time':float(t[-1]),'last_field_time':float(step.frames[-1].frameValue),
                'target_reached':bool(abs(a[-1]-.17)<1e-7 and abs(t[-1]-cfg['total_time_seconds'])<1e-8),
                'finite_curve':bool(np.isfinite(f).all() and all(np.isfinite(v).all() for v in energies.values())),
                'force_path':rows,'sampled_fields':frame_rows,'element_group_histories':groups,
                'global_energy_drift_relative_work':drift,'artificial_energy_relative_work':ae,
                'macro_work_RF_U_N_mm':work,'RF_work_relative_ALLWK_difference':abs(work-wk[-1])/max(abs(wk[-1]),1e-30),
                'max_sampled_PBC_error_mm':max_periodic,'max_sampled_UR_PBC_error_rad':max_rotation,
                'min_sampled_host_EVOL_mm3':minvol,
                'volume_scope':'C3D8R sampled element volume, not whole-time or whole-space Jacobian certification',
                'quasistatic_certification':False,'independent_20pct_JAX_validation':False,
                'ODB_retained':str(odb_path.resolve()),'INP_sha256':hashlib.sha256(odb_path.parent.joinpath(cfg.get('inp_basename','shell_fill.inp')).read_bytes()).hexdigest()}
    finally:odb.close()
    out.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in result.items() if k not in ['force_path','sampled_fields','element_group_histories']},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('odb',type=Path);p.add_argument('input',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();extract(a.odb,a.input,a.out)

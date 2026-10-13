"""Read the failed D ODB verbatim; preserve the catastrophic tail, no solver call."""
from pathlib import Path
import json,sys
import numpy as np
from odbAccess import openOdb
n=Path(sys.argv[1]);cfg=json.loads((n/'protocol.json').read_text());assert not (n/'result_raw.json').exists()
odb=openOdb(str(n/'platen.odb'),readOnly=True)
def data(v):
    try:return np.asarray(v.dataDouble,dtype=float)
    except Exception:return np.asarray(v.data,dtype=float)
try:
    step=odb.steps['PLATEN'];regions=list(step.historyRegions.values())
    def series(label,key):
        found=[r for r in regions if r.name.endswith('.'+str(label))];assert len(found)==1
        return np.array(found[0].historyOutputs[key].data,dtype=float)
    top=cfg['labels']['top'];bottom=cfg['labels']['bottom'];u=series(top,'U3');t=u[:,0]
    interp=lambda x:np.interp(t,x[:,0],x[:,1])
    values={'compression':-u[:,1]/10,'force_N':-interp(series(top,'RF3')),'bottom_force_N':interp(series(bottom,'RF3')),
        'ax_mm':interp(series(cfg['labels']['ax'],'U1')),'ay_mm':interp(series(cfg['labels']['ay'],'U2'))}
    for key in ['ALLIE','ALLSE','ALLKE','ALLAE','ALLVD','ALLWK','ETOTAL']:
        found=[r for r in regions if key in r.historyOutputs];assert len(found)==1
        values[key]=interp(np.array(found[0].historyOutputs[key].data,dtype=float))
    np.savez_compressed(n/'raw_history.npz',time=t,**values)
    rows=[{'time':float(ti),**{k:(float(v[i]) if np.isfinite(v[i]) else None) for k,v in values.items()}} for i,ti in enumerate(t)]
    nv=cfg['physical_nodes'];xyz={p.label:np.asarray(p.coordinates) for inst in odb.rootAssembly.instances.values() for p in inst.nodes if p.label<=nv}
    labels=np.array(sorted(xyz));index={int(x):i for i,x in enumerate(labels)};times=[];disp=[];finite=[]
    for frame in step.frames:
        d=np.full((nv,3),np.nan);seen=set()
        for value in frame.fieldOutputs['U'].values:
            if value.nodeLabel in index:d[index[value.nodeLabel]]=data(value);seen.add(value.nodeLabel)
        assert len(seen)==nv
        times.append(frame.frameValue);disp.append(d);finite.append(bool(np.isfinite(d).all()))
    np.savez_compressed(n/'frames.npz',times=np.asarray(times),labels=labels,xyz_mm=np.asarray([xyz[int(i)] for i in labels]),u_mm=np.asarray(disp),finite=np.asarray(finite))
    result={'status':'Abaqus_ExcessiveDistortion','complete':False,'full20':False,'path':rows,'scope':cfg['stage'],
        'failure_element':16329,'raw_history_includes_fatal_increment':False,'fatal_increment_source':'platen.sta; ODB history stops before fatal field frame','frame_count':len(times),'frame_times_s':times,
        'terminal_time_s':float(t[-1]),'terminal_compression':float(values['compression'][-1]),
        'interpretation':'Failed reference path, not a valid full20 response. Preserve raw tail; diagnostic windows fixed before D/JAX submission.'}
finally:odb.close()
(n/'result_raw.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['path','frame_times_s']},indent=2))

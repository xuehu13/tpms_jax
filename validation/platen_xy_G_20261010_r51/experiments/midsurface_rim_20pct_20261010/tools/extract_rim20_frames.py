"""Read-only extra field snapshots for the single authorized rim20 ODB."""
from pathlib import Path
import json,sys
import numpy as np
from odbAccess import openOdb
native=Path(sys.argv[1]);cfg=json.loads((native/'protocol.json').read_text())
out=native/'frames.npz'
if out.exists():raise FileExistsError(out)
odb=openOdb(str(native/'platen.odb'),readOnly=True)
def data(v):
    try:return np.array(v.dataDouble,dtype=float)
    except Exception:return np.array(v.data,dtype=float)
try:
    frames=odb.steps['PLATEN'].frames;nv=cfg['physical_nodes']
    xyz={n.label:np.array(n.coordinates) for inst in odb.rootAssembly.instances.values() for n in inst.nodes if n.label<=nv}
    labels=np.array(sorted(xyz));lookup={int(n):i for i,n in enumerate(labels)}
    times=[];disp=[]
    for f in frames:
        u=np.zeros((nv,3));seen=set()
        for v in f.fieldOutputs['U'].values:
            if v.nodeLabel in lookup:u[lookup[v.nodeLabel]]=data(v);seen.add(v.nodeLabel)
        if len(seen)!=nv:raise ValueError('Missing wall displacement values')
        times.append(f.frameValue);disp.append(u)
    np.savez_compressed(out,times=np.array(times),labels=labels,xyz_mm=np.array([xyz[int(n)] for n in labels]),u_mm=np.array(disp))
    (native/'frames_receipt.json').write_text(json.dumps({'frames':len(times),'wall_nodes':nv,'time_end':times[-1],'read_only':True},indent=2)+'\n')
finally:odb.close()
print('Extracted '+str(len(times))+' displacement frames, no solver call.')

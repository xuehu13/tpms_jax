"""Read the unchanged original G shell ODB; no new analysis is launched."""
from pathlib import Path
import sys
import numpy as np
from odbAccess import openOdb
def data(v):
    try:return v.dataDouble
    except Exception:return v.data
odbp,out=map(Path,sys.argv[1:]);assert not out.exists()
odb=openOdb(str(odbp.resolve()),readOnly=True)
try:
    fr=odb.steps['PLATEN'].frames[-1];nv=18789
    xyz={n.label:np.asarray(n.coordinates,dtype=float) for inst in odb.rootAssembly.instances.values() for n in inst.nodes if n.label<=nv}
    u={v.nodeLabel:np.asarray(data(v),dtype=float) for v in fr.fieldOutputs['U'].values if v.nodeLabel<=nv}
    ur={v.nodeLabel:np.asarray(data(v),dtype=float) for v in fr.fieldOutputs['UR'].values if v.nodeLabel<=nv}
    labels=np.asarray(sorted(xyz));assert len(labels)==nv and len(u)==nv and len(ur)==nv;t=float(fr.frameValue)
    np.savez_compressed(out,labels=labels,xyz_mm=np.array([xyz[n] for n in labels]),u_mm=np.array([u[n] for n in labels]),ur_rad=np.array([ur[n] for n in labels]),time_s=fr.frameValue)
finally:odb.close()
print('Original shell pose saved, 18789 nodes, terminal time',t)

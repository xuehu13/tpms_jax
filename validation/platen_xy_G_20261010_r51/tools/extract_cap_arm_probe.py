"""Abaqus read-only verification of offset motion and force-overclosure units."""
from pathlib import Path
import json,sys
import numpy as np
from odbAccess import openOdb
def data(v):
    try:return v.dataDouble
    except Exception:return v.data
odbp,cfgp,out=map(Path,sys.argv[1:]); assert not out.exists()
odb=openOdb(str(odbp.resolve()),readOnly=True)
try:
    step=odb.steps['PLATEN'];fr=step.frames[-1]
    def field(name):return {v.nodeLabel:np.asarray(data(v)) for v in fr.fieldOutputs[name].values}
    u,ur,rf=field('U'),field('UR'),field('RF')
    expected=np.array([.25*np.cos(.1),0,-.01-.25*np.sin(.1)])
    actual=np.array([.25,0,0])+u[5]
    expectedforce=2*(-expected[2]); force=rf[10][2]
    result={'complete':bool(abs(fr.frameValue-.0012)<1e-8),'actual_point_mm':actual.tolist(),'expected_point_mm':expected.tolist(),
            'motion_error_mm':float(np.max(abs(actual-expected))),'ground_force_N':float(force),'expected_force_N':float(expectedforce),
            'force_relative_error':float(force/expectedforce-1),'field_keys':list(fr.fieldOutputs.keys()),
            'passed':bool(np.max(abs(actual-expected))<1e-7 and abs(force/expectedforce-1)<1e-5)}
    np.savez_compressed(out.with_name('field.npz'),point=actual,expected=expected)
finally:odb.close()
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

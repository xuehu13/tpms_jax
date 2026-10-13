"""Read the failed ordinary host cell and actual control in its fatal frame."""
from pathlib import Path
import json,numpy as np
from odbAccess import openOdb
P=Path.cwd();odb=openOdb(path=str(P/'nearzero_gauge.odb'),readOnly=True)
try:
    step=odb.steps['compress20'];fr=step.frames[-1];cfg=json.loads((P/'input_gauge.json').read_text())
    def val(v):
        try:return v.dataDouble
        except:return v.data
    d={v.nodeLabel:np.asarray(val(v),float) for v in fr.fieldOutputs['U'].values}
    inst=next(iter(odb.rootAssembly.instances.values()))
    elem=next(e for e in inst.elements if e.label==121978)
    xyz={n.label:np.asarray(n.coordinates,float) for n in inst.nodes}
    x=np.array([xyz[n] for n in elem.connectivity]);u=np.array([d[n] for n in elem.connectivity])
    signs=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],float)
    dX=signs.T@x/8.;dx=signs.T@(x+u)/8.;F=dx.T@np.linalg.inv(dX.T)
    q=cfg['macro_control_label'];s=fr.frameValue/.004;a=.2*(10*s**3-15*s**4+6*s**5)
    z={'failing_element':121978,'element_type':elem.type,'belongs_to':'ordinary FILLER, not shell',
        'reference_center_mm':x.mean(axis=0).tolist(),'node_labels':list(map(int,elem.connectivity)),
        'fatal_field_time':float(fr.frameValue),'analytic_compression':float(a),
        'control_present_in_fatal_field':q in d,'actual_control_compression':-float(d[q][2])/10. if q in d else None,
        'center_F':F.tolist(),'center_J_from_failed_field':float(np.linalg.det(F)),
        'initial_volume_mm3':float(abs(np.linalg.det(dX))*8.),
        'scope':'fatal/failed ordinary diagnostic field, not accepted JAX or full path certificate'}
finally:odb.close()
(P/'fatal_field_diagnostic.json').write_text(json.dumps(z,indent=2))
print(json.dumps(z,indent=2))

"""Read the existing r30 ODB only: section stress/strain, no new Abaqus analysis."""
from pathlib import Path
import json,time,hashlib
from odbAccess import openOdb
P=Path(r'E:\ABAQUS\2026temp\Abaqus_Work\tpms_jax_abaqus\initial_tangent_20261008_r30_diverse04\shell.odb')
D=Path(r'\\wsl.localhost\Ubuntu-24.04\home\xuehu\projects\tpms_jax\validation\initial_bias_mechanism_20261008_r32')
def data(v):
    try:return v.dataDouble
    except Exception:return v.data
start=time.perf_counter();before=hashlib.sha256(P.read_bytes()).hexdigest()
odb=openOdb(path=str(P),readOnly=True)
try:
    frame=odb.steps['COMPRESSION'].frames[-1]; result={'frame_time':float(frame.frameValue),'fields':{},'odb_path':str(P),'odb_sha256':before}
    for name in ['S','E']:
        f=frame.fieldOutputs[name];rows=[];sections={}
        for v in f.values:
            sp=v.sectionPoint; sections[str(sp.number)]=sp.description
            rows.append([int(v.elementLabel),int(v.integrationPoint),int(sp.number),*[float(x) for x in data(v)]])
        result['fields'][name]={'labels':list(f.componentLabels),'engineering_tensor':bool(f.isEngineeringTensor),'sections':sections,'rows':rows}
finally:odb.close()
assert hashlib.sha256(P.read_bytes()).hexdigest()==before
result['seconds']=time.perf_counter()-start
out=D/'shell_sections.json';assert not out.exists();out.write_text(json.dumps(result,allow_nan=False)+'\n')
print(json.dumps({'seconds':result['seconds'],'fields':{n:{k:v for k,v in f.items() if k!='rows'} for n,f in result['fields'].items()},'rows':{n:len(f['rows']) for n,f in result['fields'].items()}},indent=2))

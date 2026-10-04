"""Completed uniform finite-strain ODB path; SENER integrates with CURRENT IVOL."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
from extract_uniform_baseline import data,region,point_key


def extract(odb_path,expected_path,out):
    from odbAccess import openOdb
    from abaqusConstants import INTEGRATION_POINT
    expected=json.loads(expected_path.read_text())
    if odb_path.stem!=expected['case']:raise ValueError('ODB and prepared case names differ')
    odb=openOdb(path=str(odb_path.resolve()),readOnly=True)
    rows=[];fields=[]
    try:
        if list(odb.steps.keys())!=[s['name'] for s in expected['steps']]:
            raise ValueError('Missing/unexpected load steps')
        solid=region(odb,'SOLID','elementSets');physical=region(odb,'PHYSICAL','nodeSets')
        nodes={int(k):np.asarray(v,float) for k,v in expected['nodes'].items()}
        labels=sorted(nodes);count=8*expected['N']**3
        rel=expected['relative_physics_tolerance'];utol=expected['displacement_tolerance']
        # Verify the actual initial state; do not assume missing output means zero.
        frame0=odb.steps[expected['steps'][0]['name']].frames[0]
        initial_u=frame0.fieldOutputs['U'].getSubset(region=physical).values
        initial_rf=frame0.fieldOutputs['RF'].values
        if len(initial_u)!=len(nodes) or max(abs(float(x)) for v in initial_u for x in data(v))>utol:
            raise ValueError('Initial physical displacement is not zero')
        if max(abs(float(x)) for v in initial_rf for x in data(v))>1e-7:
            raise ValueError('Initial reaction is not zero')
        rows.append({'compression':0.,'Fz_top':0.,'Fz_bottom':0.,'ALLSE':0.,'IP_energy':0.,
                     'current_volume':1.,'J_min':1.,'J_max':1.,'initial_state':'Zero U/RF verified in first ODB frame; initial volume from prepared mesh',
                     'checks':{'initial_zero_U_RF':True}})
        fields.append(np.zeros((len(nodes),3)))
        for definition in expected['steps']:
            step=odb.steps[definition['name']];frame=step.frames[-1]
            if abs(frame.frameValue-1.)>1e-8:raise ValueError('Unfinished step: '+definition['name'])
            a=definition['compression'];truth=definition['analytic'];s=1-a
            controls=[];forces=[]
            for axis,name in enumerate(('QX','QY','QZ')):
                node_set=region(odb,name,'nodeSets')
                uv=frame.fieldOutputs['U'].getSubset(region=node_set).values
                rf=frame.fieldOutputs['RF'].getSubset(region=node_set).values
                if len(uv)!=1 or len(rf)!=1:raise ValueError('Macro control set count mismatch')
                controls.append(float(data(uv[0])[axis]));forces.append(float(data(rf[0])[axis]))
            outputs={}
            for name in ('S','LE','IVOL','SENER'):
                values=frame.fieldOutputs[name].getSubset(region=solid,position=INTEGRATION_POINT).values
                if len(values)!=count:raise ValueError('Incomplete '+name+' quadrature output')
                outputs[name]={point_key(v):data(v) for v in values}
            keys=set(outputs['IVOL'])
            if any(set(v)!=keys for v in outputs.values()):raise ValueError('Quadrature keys do not match')
            volumes={k:float(outputs['IVOL'][k]) for k in keys}
            if any(v<=0 or not math.isfinite(v) for v in volumes.values()):raise ValueError('Invalid current quadrature volume')
            volume=sum(volumes.values());J=np.array([v*count for v in volumes.values()])
            sigma=[sum(float(outputs['S'][k][i])*volumes[k] for k in keys)/volume for i in range(6)]
            IP_energy=sum(float(outputs['SENER'][k])*volumes[k] for k in keys)
            history={}
            for name in ('ALLSE','ALLWK','ALLAE'):
                values=[r.historyOutputs[name].data[-1][1] for r in step.historyRegions.values() if name in r.historyOutputs]
                if len(values)!=1:raise ValueError('Unique energy history missing: '+name)
                history[name]=float(values[0])
            physical_u=frame.fieldOutputs['U'].getSubset(region=physical).values
            if len(physical_u)!=len(nodes):raise ValueError('Incomplete physical nodal output')
            observed={v.nodeLabel:np.asarray(data(v),float) for v in physical_u}
            if set(observed)!=set(nodes):raise ValueError('Node labels differ from prepared input')
            affine_error=max(float(np.max(np.abs(observed[k]-np.array([0,0,-a*nodes[k][2]])))) for k in nodes)
            coordinate_error=max(float(np.max(np.abs(np.asarray(v.instance.getNodeFromLabel(v.nodeLabel).coordinates)-nodes[v.nodeLabel]))) for v in physical_u)
            bottom=frame.fieldOutputs['RF'].getSubset(region=region(odb,'BOTTOM','nodeSets')).values
            bottom_force=sum(float(data(v)[2]) for v in bottom)
            by_xyz={tuple(x):k for k,x in nodes.items()};periodic_error=0.
            for k,x in nodes.items():
                for axis in (0,1):
                    if x[axis]==1:
                        low=x.copy();low[axis]=0.;master=by_xyz[tuple(low)]
                        periodic_error=max(periodic_error,float(np.abs(observed[k]-observed[master]).max()))
            loaded_error=max(abs(observed[k][2]+a*nodes[k][2]) for k in nodes if nodes[k][2] in (0,1))
            piola_diagonal=[sigma[0]*s,sigma[1]*s,sigma[2]]
            force_scale=max(abs(truth['Fz']),1e-6);energy_scale=max(abs(truth['energy']),1e-12)
            logarithmic_J_error=max(abs(math.exp(sum(float(x) for x in outputs['LE'][k][:3]))-volumes[k]*count) for k in keys)
            checks={'finite_outputs':all(math.isfinite(x) for x in sigma+controls+forces+[volume,IP_energy]+list(history.values())),
                'positive_detF':float(J.min())>0,'current_volume':abs(volume-s)<=rel,
                'analytic_detF':max(abs(float(J.min())-s),abs(float(J.max())-s))<=rel,
                'LE_vs_current_volume_J':logarithmic_J_error<=rel,
                'analytic_force':abs(forces[2]-truth['Fz'])<=rel*force_scale,
                'analytic_energy':abs(history['ALLSE']-truth['energy'])<=rel*energy_scale,
                'energy_IP_vs_ALLSE':abs(IP_energy-history['ALLSE'])<=rel*energy_scale,
                'no_artificial_energy':abs(history['ALLAE'])<=rel*energy_scale,
                'macro_RF_vs_first_piola':max(abs(forces[i]-piola_diagonal[i]) for i in range(3))<=rel*force_scale,
                'analytic_cauchy':max(abs(sigma[i]-truth['cauchy_diagonal'][i]) for i in range(3))<=rel*force_scale,
                'mean_shear':max(abs(x) for x in sigma[3:])<=rel*force_scale,
                'top_bottom_balance':abs(forces[2]+bottom_force)<=rel*force_scale,
                'macro_displacement':max(abs(controls[0]),abs(controls[1]),abs(controls[2]+a))<=utol,
                'affine_displacement':affine_error<=utol,'ODB_coordinates':coordinate_error<=1e-10,
                'periodic_displacement':periodic_error<=utol,'loaded_faces':loaded_error<=utol}
            row={'compression':a,'step':definition['name'],'completed_step_time':float(frame.frameValue),
                 'Fz_top':forces[2],'Fz_bottom':bottom_force,'macro_RF':forces,'macro_displacements':controls,
                 'mean_cauchy':sigma,'mean_first_piola_diagonal':piola_diagonal,
                 'current_volume':volume,'J_min':float(J.min()),'J_max':float(J.max()),'IP_energy':IP_energy,**history,
                 'max_affine_displacement_error':affine_error,'periodic_error':periodic_error,'loaded_face_error':loaded_error,
                 'logarithmic_J_error':logarithmic_J_error,'checks':{k:bool(v) for k,v in checks.items()}}
            rows.append(row);fields.append(np.array([observed[k] for k in labels]))
        a=np.array([r['compression'] for r in rows]);force=np.array([r['Fz_top'] for r in rows])
        work=float(np.sum(-.5*(force[1:]+force[:-1])*np.diff(a)));stored=rows[-1]['ALLSE'];gap=abs(work-stored)/stored
        npz=out.with_suffix('.displacement.npz')
        np.savez_compressed(npz,compression=a,points=np.array([nodes[k] for k in labels]),u=np.array(fields),node_labels=labels)
        passed=all(all(r['checks'].values()) for r in rows) and gap<=.001
        report={'case':odb_path.stem,'status':'ok' if passed else 'check_failed','rows':rows,
                'work':{'path_trapezoid':work,'stored_energy':stored,'relative_gap':gap,'uses_half_Fu':False},
                'energy_definition':'SENER is per current volume; IP_energy=sum(SENER*IVOL), ALLSE is total stored energy',
                'piola_reconstruction':'Uniform affine F=diag(1,1,1-a) only; sigma*diag(1-a,1-a,1)',
                'input_sha256':expected['input_sha256'],'displacement_sha256':hashlib.sha256(npz.read_bytes()).hexdigest(),
                'odb':str(odb_path.resolve()),'expected':str(expected_path.resolve())}
    finally:odb.close()
    with out.open('x') as stream:stream.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
    if not passed:
        print(json.dumps([(r['compression'],[k for k,v in r['checks'].items() if not v]) for r in rows if not all(r['checks'].values())]))
    return passed


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('odb',type=Path)
    parser.add_argument('--expected',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    raise SystemExit(0 if extract(args.odb,args.expected,args.out) else 1)

"""Extract native C3D10 U/RF/S/E/IVOL/ALLSE and test physical identities.

Run with `abaqus python`; this script never submits an analysis. Passing these
checks establishes consistency, not mesh or geometry convergence.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from extract_uniform_baseline import data, region


def key(point):
    return tuple(np.rint(np.asarray(point)*1e12).astype(np.int64))


def read_field(field, nodal=False):
    """Bulk read with explicit label/IP sorting; retain double-value fallback."""
    labels,ips,arrays,instances=[],[],[],set()
    try:
        for block in field.bulkDataBlocks:
            # Copy while the block is alive; never retain a view into ODB data.
            values=np.array(block.data,dtype=float,copy=True)
            current=block.nodeLabels if nodal else block.elementLabels
            labels.append(np.array(current,dtype=int,copy=True))
            if not nodal:
                ips.append(np.array(block.integrationPoints,dtype=int,copy=True))
            arrays.append(values)
            instances.add(block.instance.name)
    except Exception:
        # Python bulk data rejects double-precision backing storage. Reading
        # FieldValue.dataDouble preserves correctness for that output mode.
        values=field.values
        arrays=[np.array([np.atleast_1d(data(v)) for v in values],dtype=float)]
        labels=[np.array([v.nodeLabel if nodal else v.elementLabel for v in values])]
        ips=[] if nodal else [np.array([v.integrationPoint for v in values])]
        instances={v.instance.name for v in values}
    if len(instances) != 1 or not arrays:
        raise ValueError('Expected one nonempty mesh instance in field')
    label=np.concatenate(labels);array=np.concatenate(arrays)
    ip=None if nodal else np.concatenate(ips)
    if len(label) != len(array) or (ip is not None and len(ip) != len(label)):
        raise ValueError('Bulk labels/IPs do not match data rows')
    order=np.argsort(label) if nodal else np.lexsort((ip,label))
    return next(iter(instances)),label[order],None if nodal else ip[order],array[order]


def extract(odb_path, expected_path, out_path):
    from odbAccess import openOdb
    from abaqusConstants import INTEGRATION_POINT
    manifest=json.loads(expected_path.read_text())
    if odb_path.stem != manifest['case']:
        raise ValueError('ODB name differs from prepared case')
    inp_path=expected_path.with_name(odb_path.stem+'.inp')
    if hashlib.sha256(inp_path.read_bytes()).hexdigest() != manifest['input_sha256']:
        raise ValueError('Prepared input hash mismatch')
    mesh_path=expected_path.with_name(odb_path.stem+'.mesh.npz')
    if hashlib.sha256(mesh_path.read_bytes()).hexdigest() != manifest['mesh_sha256']:
        raise ValueError('Mesh file hash mismatch')
    saved=np.load(mesh_path); points=saved['points']; cells=saved['cells']
    odb=openOdb(path=str(odb_path.resolve()),readOnly=True)
    try:
        step=odb.steps['COMPRESSION']; frame=step.frames[-1]
        if abs(frame.frameValue-1) > 1e-8:
            raise ValueError('Load step incomplete')
        solid=region(odb,'SOLID','elementSets')
        fields={};expected_labels=np.repeat(np.arange(1,manifest['elements']+1),4)
        expected_ips=np.tile(np.arange(1,5),manifest['elements'])
        field_instance=None
        for name in ('S','E','IVOL'):
            instance,label,ip,values=read_field(frame.fieldOutputs[name].getSubset(region=solid,position=INTEGRATION_POINT))
            if not np.array_equal(label,expected_labels) or not np.array_equal(ip,expected_ips):
                raise ValueError('Expected matching four-point C3D10 S/E/IVOL fields')
            if field_instance is not None and instance != field_instance:
                raise ValueError('Integration fields belong to different instances')
            field_instance=instance;fields[name]=values
        weights=fields['IVOL'].reshape(-1)
        stress=fields['S'];strain=fields['E']
        if not np.all(np.isfinite(stress)) or not np.all(np.isfinite(strain)) or not np.all(np.isfinite(weights)) or weights.min() <= 0:
            raise ValueError('Invalid integration-point field')
        if stress.shape[1] != 6 or strain.shape[1] != 6:
            raise ValueError('Expected six tensor components in global axes')
        volume=float(weights.sum())
        # Macroscopic stress uses the GROSS cell, including void. Abaqus E's
        # off-diagonal components are engineering shear; dot(S,E) is sigma:eps.
        sigma=(stress*weights[:,None]).sum(axis=0)/manifest['gross_volume']
        ip_energy=float(.5*np.einsum('qi,qi,q->',stress,strain,weights))
        controls=[]; forces=[]
        for axis,name in enumerate(('QX','QY','QZ')):
            control_set=region(odb,name,'nodeSets')
            us=frame.fieldOutputs['U'].getSubset(region=control_set).values
            rs=frame.fieldOutputs['RF'].getSubset(region=control_set).values
            if len(us) != 1 or len(rs) != 1:
                raise ValueError('Expected one control node')
            controls.append(float(data(us[0])[axis])); forces.append(float(data(rs[0])[axis]))
        energies=[r.historyOutputs['ALLSE'].data[-1][1] for r in step.historyRegions.values() if 'ALLSE' in r.historyOutputs]
        if len(energies) != 1:
            raise ValueError('Expected unique model ALLSE')
        energy=float(energies[0])
        macro_energy=float(.5*manifest['gross_volume']*np.dot(sigma[:3],controls))
        work=float(.5*np.dot(controls,forces))
        physical=region(odb,'PHYSICAL','nodeSets')
        instance_name,labels,_,u=read_field(frame.fieldOutputs['U'].getSubset(region=physical),nodal=True)
        if not np.array_equal(labels,np.arange(1,len(points)+1)) or instance_name != field_instance:
            raise ValueError('Incomplete physical displacement field')
        instance=odb.rootAssembly.instances[instance_name]
        coordinate_labels=np.array([node.label for node in instance.nodes])
        coordinates=np.array([node.coordinates for node in instance.nodes])
        coordinate_order=np.argsort(coordinate_labels)
        if not np.array_equal(coordinate_labels[coordinate_order][:len(points)],labels):
            raise ValueError('ODB coordinates cannot be mapped to physical labels')
        coordinate_error=float(np.max(np.abs(points-coordinates[coordinate_order][:len(points)])))
        if not np.all(np.isfinite(u)):
            raise ValueError('Nonfinite displacement')
        master_ids=np.flatnonzero((points[:,0]==0)|(points[:,1]==0))
        lookup={key(points[i]):i for i in master_ids}
        jump_error=0.
        for i in np.flatnonzero((points[:,0]==1)|(points[:,1]==1)):
            p=points[i]
            jump=np.array([int(p[0] == 1),int(p[1] == 1),0])
            if np.any(jump):
                target=p-jump
                master=lookup[key(target)]
                jump_error=max(jump_error,float(np.max(np.abs(u[i]-u[master]-jump*np.array(controls)))))
        top_error=float(np.max(np.abs(u[points[:,2] == 1,2]-controls[2])))
        bottom_error=float(np.max(np.abs(u[points[:,2] == 0,2])))
        bottom=region(odb,'BOTTOM','nodeSets')
        bottom_rf=sum(float(data(v)[2]) for v in frame.fieldOutputs['RF'].getSubset(region=bottom).values)
        rel=manifest['relative_tolerance']; utol=manifest['displacement_tolerance']
        stress_scale=max(abs(forces[2]),1e-12); energy_scale=max(abs(energy),1e-16)
        checks={'positive_volume':volume>0,
                'mesh_volume':abs(volume-manifest['geometry']['volume']) <= rel*volume,
                'ODB_coordinates':coordinate_error <= 1e-7,
                'energy_IP_vs_ALLSE':abs(ip_energy-energy) <= rel*energy_scale,
                'energy_work_vs_ALLSE':abs(work-energy) <= rel*energy_scale,
                'energy_macro_vs_ALLSE':abs(macro_energy-energy) <= rel*energy_scale,
                'control_RF_vs_gross_stress':max(abs(forces[i]-sigma[i]*manifest['gross_volume']) for i in range(3)) <= rel*stress_scale,
                'top_bottom_balance':abs(forces[2]+bottom_rf) <= rel*stress_scale,
                'periodic_displacement':jump_error <= utol,
                'loaded_face_displacement':max(top_error,bottom_error,abs(controls[2]+.01)) <= utol}
        if manifest['lateral'] == 'relaxed_free':
            checks['zero_lateral_stress']=max(abs(sigma[0]),abs(sigma[1]),abs(forces[0]),abs(forces[1])) <= rel*stress_scale
        else:
            checks['fixed_lateral_strain']=max(abs(controls[0]),abs(controls[1])) <= utol
        if 'analytic' in manifest:
            expected=manifest['analytic']
            checks['analytic_stress']=max(abs(sigma[i]-expected['sigma_diagonal'][i]) for i in range(3)) <= rel*stress_scale
            checks['analytic_energy']=abs(energy-expected['energy']) <= rel*energy_scale
            affine_error=float(np.max(np.abs(u-points*np.array(expected['eps']))))
            checks['analytic_affine_displacement']=affine_error <= utol
        else:
            affine_error=None
        finite=[energy,ip_energy,macro_energy,work,volume,bottom_rf]+sigma.tolist()+controls+forces
        checks['finite_outputs']=all(math.isfinite(v) for v in finite)
        measured={'volume_solid':volume,'gross_volume':manifest['gross_volume'],
                  'sigma_gross':sigma.tolist(),'sigma_solid_average':(sigma*manifest['gross_volume']/volume).tolist(),
                  'macro_displacements':controls,'macro_RF':forces,'bottom_RFz':bottom_rf,
                  'ALLSE':energy,'IP_energy':ip_energy,'macro_energy':macro_energy,'control_work':work,
                  'integration_points':len(weights),'max_periodic_error':jump_error,
                  'max_loaded_face_error':max(top_error,bottom_error),'max_coordinate_error':coordinate_error,
                  'max_analytic_affine_error':affine_error}
        report={'case':odb_path.stem,'odb':str(odb_path.resolve()),'measured':measured,
                'checks':{k:bool(v) for k,v in checks.items()},
                'status':'ok' if all(checks.values()) else 'check_failed',
                'convergence_claim':False,'input_sha256':manifest['input_sha256']}
    finally:
        odb.close()
    out_path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    return report['status'] == 'ok'


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('odb',type=Path)
    parser.add_argument('--expected',required=True,type=Path)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    raise SystemExit(0 if extract(args.odb,args.expected,args.out or args.odb.with_suffix('.acceptance.json')) else 1)

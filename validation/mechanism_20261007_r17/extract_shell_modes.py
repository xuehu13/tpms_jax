"""Read existing ODB frames only; no Abaqus analysis is submitted."""
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np

def data(v):
    try:return v.dataDouble
    except Exception:return v.data

def extract(odb_path,input_path,out):
    from odbAccess import openOdb
    out.mkdir(exist_ok=False);cfg=json.loads(input_path.read_text());start=time.perf_counter()
    odb=openOdb(path=str(odb_path.resolve()),readOnly=True)
    try:
        step=odb.steps[cfg['step_name']]
        controls=[r for r in step.historyRegions.values() if 'U3' in r.historyOutputs and 'RF3' in r.historyOutputs]
        assert len(controls)==1
        history=np.asarray(controls[0].historyOutputs['U3'].data)
        force=np.asarray(controls[0].historyOutputs['RF3'].data)
        instances=list(odb.rootAssembly.instances.values())
        nodes={n.label:np.asarray(n.coordinates,dtype=float) for inst in instances for n in inst.nodes}
        labels=sorted(n for n in nodes if n!=cfg['macro_control_label'])
        xyz=np.array([nodes[n] for n in labels]);index={n:i for i,n in enumerate(labels)}
        triangles=np.array([[index[n] for n in e.connectivity] for inst in instances for e in inst.elements if e.type=='S3R'])
        assert len(labels)==cfg['physical_node_count']
        cross=np.cross(xyz[triangles[:,1]]-xyz[triangles[:,0]],xyz[triangles[:,2]]-xyz[triangles[:,0]])
        area=np.linalg.norm(cross,axis=1)/2
        weights=np.zeros(len(labels));normals=np.zeros_like(xyz)
        for j in range(3):
            np.add.at(weights,triangles[:,j],area/3)
            np.add.at(normals,triangles[:,j],cross)
        normals/=np.linalg.norm(normals,axis=1)[:,None]
        np.savez_compressed(out/'reference.npz',labels=labels,xyz_mm=xyz,triangles=triangles,
                            reference_area_weights_mm2=weights,reference_vertex_normals=normals)
        targets=[.10,.114,.12,.14,.16,.17,.20];rows=[];fields=[]
        frame_compression=np.array([-np.interp(f.frameValue,history[:,0],history[:,1])/cfg['cell_size_mm'] for f in step.frames])
        all_energies={}
        for name in ('ALLSE','ALLIE','ALLKE','ALLAE','ALLVD','ALLWK','ETOTAL'):
            values=[np.asarray(r.historyOutputs[name].data) for r in step.historyRegions.values() if name in r.historyOutputs]
            assert len(values)==1;all_energies[name]=values[0]
        used=set()
        for target in targets:
            candidates=np.array([i for i,f in enumerate(step.frames) if f.frameValue<=cfg['load_time_seconds']+1e-12])
            i=int(candidates[np.argmin(np.abs(frame_compression[candidates]-target))])
            if i in used:continue
            used.add(i);frame=step.frames[i];a=float(frame_compression[i])
            udata={v.nodeLabel:np.asarray(data(v),dtype=float) for v in frame.fieldOutputs['U'].values}
            u=np.array([udata[n] for n in labels]);affine=np.zeros_like(u);affine[:,2]=-a*xyz[:,2]
            w=u-affine;translation=np.sum(weights[:,None]*w,axis=0)/weights.sum();w-=translation
            normal=np.sum(w*normals,axis=1)
            fields.append(w)
            name='frame_a{:.4f}.npz'.format(target)
            np.savez_compressed(out/name,u_mm=u,fluctuation_mm=w,removed_translation_mm=translation,
                normal_fluctuation_mm=normal,time_s=frame.frameValue,compression=a,frame_index=i)
            row={'requested_compression':target,'compression':a,'frame_index':i,'time_s':frame.frameValue,
                 'Fz_N':float(np.interp(frame.frameValue,force[:,0],force[:,1])),
                 'fluctuation_RMS_mm':float(np.sqrt(np.sum(weights*np.sum(w*w,axis=1))/weights.sum())),
                 'normal_fluctuation_RMS_mm':float(np.sqrt(np.sum(weights*normal**2)/weights.sum())),
                 'maximum_fluctuation_mm':float(np.linalg.norm(w,axis=1).max()),
                 'field_outputs_available':list(frame.fieldOutputs.keys()),'file':name}
            row.update({k:float(np.interp(frame.frameValue,v[:,0],v[:,1])) for k,v in all_energies.items()})
            rows.append(row)
        gram=np.array([[np.sum(weights[:,None]*a*b) for b in fields] for a in fields])
        correlation=gram/np.sqrt(np.diag(gram)[:,None]*np.diag(gram)[None,:])
        result={'status':'read_only_existing_shell_frames','frames_total':len(step.frames),'rows':rows,
                'area_weighted_fluctuation_correlations':correlation.tolist(),
                'alignment':'subtract macro affine displacement, then reference-area-weighted translation; no spatial fitting',
                'scope':'Displacement patterns, not eigenmodes or independent quasistatic certification',
                'new_abaqus_jobs':0,'odb_path':str(odb_path),'input_sha256':hashlib.sha256(input_path.read_bytes()).hexdigest(),
                'extractor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'seconds':time.perf_counter()-start}
        (out/'modes.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        print(json.dumps(result,indent=2))
    finally:odb.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('odb',type=Path);p.add_argument('input',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();extract(a.odb,a.input,a.out)

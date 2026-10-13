"""Read-only force, free lateral displacement, energy and XY check."""
from pathlib import Path
import json,sys
import numpy as np
from odbAccess import openOdb
def data(v):
    try:return v.dataDouble
    except Exception:return v.data
odbfile,cfgfile,out=map(Path,sys.argv[1:]);cfg=json.loads(cfgfile.read_text())
if out.exists():raise FileExistsError(out)
odb=openOdb(str(odbfile.resolve()),readOnly=True)
try:
    step=odb.steps['PLATEN'];regions=list(step.historyRegions.values())
    def region(label):
        found=[r for r in regions if r.name.endswith('.'+str(label))]
        assert len(found)==1,(label,[r.name for r in regions]);return found[0]
    def series(label,name):return np.array(region(label).historyOutputs[name].data,dtype=float)
    top=cfg['labels']['top'];bottom=cfg['labels']['bottom'];u=series(top,'U3');t=u[:,0]
    interp=lambda s:np.interp(t,s[:,0],s[:,1])
    ft=interp(series(top,'RF3'));fb=interp(series(bottom,'RF3'))
    ax=interp(series(cfg['labels']['ax'],'U1'));ay=interp(series(cfg['labels']['ay'],'U2'))
    en={}
    for name in ['ALLIE','ALLSE','ALLKE','ALLAE','ALLVD','ALLWK','ETOTAL']:
        found=[r for r in regions if name in r.historyOutputs];assert len(found)==1
        en[name]=interp(np.array(found[0].historyOutputs[name].data))
    frame=step.frames[-1];nv=cfg['physical_nodes'];L=cfg['L_mm']
    xyz={n.label:np.array(n.coordinates) for inst in odb.rootAssembly.instances.values() for n in inst.nodes if n.label<=nv}
    du={v.nodeLabel:np.array(data(v)) for v in frame.fieldOutputs['U'].values if v.nodeLabel<=nv}
    ur={v.nodeLabel:np.array(data(v)) for v in frame.fieldOutputs['UR'].values if v.nodeLabel<=nv}
    groups={}
    for n,p in xyz.items():groups.setdefault(tuple(np.round(np.r_[np.where(np.isclose(p[:2],L),0,p[:2]),p[2]],8)),[]).append(n)
    error=rotation=0.
    for g in groups.values():
        root=min(g)
        for n in g:
            expected=np.r_[(xyz[n][:2]-xyz[root][:2])/L*np.array([ax[-1],ay[-1]]),0.]
            error=max(error,float(np.max(abs(du[n]-du[root]-expected))));rotation=max(rotation,float(np.max(abs(ur[n]-ur[root]))))
    ztop=L+u[-1,1];penetration=[]
    for side in [0,1]:
        labels=[n for n,p in xyz.items() if abs(p[2]-side*L)<1e-8]
        z=np.array([xyz[n][2]+du[n][2] for n in labels]);penetration.append(float(np.max(np.maximum(-z if side==0 else z-ztop,0))))
    work=float(np.sum(.5*(ft[1:]+ft[:-1])*np.diff(u[:,1])));den=max(float(np.max(abs(en['ALLWK']))),1e-30)
    rows=[dict(time=float(ti),compression=float(-ui/L),force_N=float(-fi),bottom_force_N=float(bi),
               ax_mm=float(xi),ay_mm=float(yi),**{k:float(v[i]) for k,v in en.items()})
          for i,(ti,ui,fi,bi,xi,yi) in enumerate(zip(t,u[:,1],ft,fb,ax,ay))]
    result={'status':'complete_diagnostic','complete':bool(abs(t[-1]-(cfg['load_time_s']+cfg['hold_time_s']))<1e-8),
            'XY_U_error_mm':error,'XY_UR_error_rad':rotation,'terminal_rim_penetration_mm':penetration,
            'macro_work_RF_U_N_mm':work,'work_relative_ALLWK_error':abs(work-en['ALLWK'][-1])/den,
            'energy_drift_relative_work':float(np.max(abs(en['ETOTAL']-en['ETOTAL'][0]))/den),
            'terminal_KE_over_IE':float(en['ALLKE'][-1]/max(abs(en['ALLIE'][-1]),1e-30)),'path':rows,
            'scope':cfg['stage'],'no_self_contact':True,'contact_type':cfg['shell_contact']}
    np.savez_compressed(out.with_name('field.npz'),labels=sorted(xyz),xyz_mm=np.array([xyz[n] for n in sorted(xyz)]),u_mm=np.array([du[n] for n in sorted(xyz)]))
finally:odb.close()
out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='path'},indent=2))

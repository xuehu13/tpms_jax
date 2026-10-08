"""Frozen-state through-thickness diagnostic. No equilibrium, Abaqus job, or AD."""
from pathlib import Path
import sys,json,time,hashlib,argparse
R=Path('/home/xuehu/projects/tpms_jax');D=Path(__file__).resolve().parent
sys.path.insert(0,str(R))
import numpy as np,basix
from scipy.special import expit
from jax_fem.basis import get_elements
from surface_distance import PeriodicSurfaceDistance
L=10.;N=32;T=.5;E=10.;NU=.3;ETA=1e-4;ELL=.05/(2*np.log(9));A=1e-4
MU=E/(2*(1+NU));LAM=E*NU/((1+NU)*(1-2*NU));LPS=2*MU*LAM/(LAM+2*MU)
BUDGET=900.;BATCH=4096
R30=R/'validation/initial_tangent_20261008_r30'
R32=R/'validation/initial_bias_mechanism_20261008_r32'
R33=R/'validation/local_quadrature_20261008_r33'
R34=R/'validation/local_reequilibrium_20261008_r34'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,v):(D/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def freeze():
    for n,h in json.loads((D/'frozen_before.json').read_text()).items():assert sha(R/n)==h,n
def prepare():
    started=time.perf_counter();freeze()
    mask=np.load(R32/'local_energy.npz')['topmask'];faces=np.flatnonzero(mask)
    selected=np.load(R33/'selected_cells.npy')
    assert len(selected)==1941 and len(faces)>0
    np.save(D/'profile_faces.npy',faces)
    cfg={'case':'diverse_04 r30/r34 saved initial static states','N':N,'L_mm':L,'t_mm':T,
      'E_MPa':E,'nu':NU,'eta':ETA,'interface_10_90_mm':.05,
      'selected_cells':len(selected),'selected_cells_sha256':sha(R33/'selected_cells.npy'),
      'profile_faces':len(faces),'profile_faces_sha256':sha(D/'profile_faces.npy'),
      'region':'Same r33 1941 cells for original-Gauss partitions; all r32 frozen top20pct-area face centroids for normal-line probes',
      'occupancy_partitions':[0.,.001,.01,.9,1.],
      'distance_bins_mm':[0.,.1,.2,.225,.25,.275,.30,.35],
      'distance_tail':'distance>=0.35mm separately retained',
      'line_z_range_mm':[-.35,.35],'line_points':81,'point_batch':BATCH,
      'Gauss_frames':'original nearest periodic triangle facet normal; no smoothed-normal replacement',
      'line_frames':'fixed source triangle facet normal; closest-face switches recorded, not filtered',
      'nearest_edge_tangential_distance_ratio_flag':.05,
      'denominators':{'energy':'each saved state evaluated at the SAME original 27-point rule in SAME 1941 cells',
         'strain':'macro compression 1e-4; weighted least-squares/RMS plane-stress prediction reported separately',
         'profile':'fixed source-face area weights at each signed z; not volume integration',
         'stress':'E_MPa, actual occupancy scaling and unit-material stress both retained'},
      'budget_seconds':BUDGET,'gates':{'r30_original_full_energy_relative':1e-10,
         'r30_selected_original_energy_relative':1e-10,'pointwise_energy_identity_relative':1e-10,
         'r32_normal_and_shear_reproduction_relative':1e-10,'finite_output':True},
      'causal_stiffness_claim_allowed':False,'new_FEM_or_Abaqus_or_AD':False,
      'new_integration_levels':False,'production_changed':False,
      'prepare_seconds':time.perf_counter()-started}
    write('protocol.json',cfg);print(json.dumps(cfg,indent=2),flush=True)
def run():
    start=time.perf_counter();freeze();cfg=json.loads((D/'protocol.json').read_text())
    sel=np.load(R33/'selected_cells.npy');pf=np.load(D/'profile_faces.npy')
    assert sha(R33/'selected_cells.npy')==cfg['selected_cells_sha256']
    assert sha(D/'profile_faces.npy')==cfg['profile_faces_sha256']
    cache=np.load(R/'validation/geometry_transfer_20261006_r15/gauss_field.npz')
    v=cache['surface_vertices'];tri=cache['surface_triangles'];qp=cache['physical_quad_points']
    phi=cache['rho'];weight=cache['JxW']*L**3
    cr=np.cross(v[tri[:,1]]-v[tri[:,0]],v[tri[:,2]]-v[tri[:,0]])
    nn=np.linalg.norm(cr,axis=1);normals=cr/nn[:,None];area=.5*nn*L**2
    centers=v[tri].mean(axis=1);surface=PeriodicSurfaceDistance(v,tri)
    distance,face,hit=surface.query(qp,details=True)
    n=normals[face];delta=qp%1-hit
    z=np.sum(delta*n,axis=-1)*L
    tangent=delta-(z/L)[:,:,None]*n
    edge_ratio=np.linalg.norm(tangent,axis=-1)/np.maximum(distance,1e-14)
    family,cell,_,_,degree,order=get_elements('HEX27');el=basix.create_element(family,cell,degree)
    local=np.rint(2*el.points[order]).astype(int)
    origins=2*np.indices((N,)*3).reshape(3,-1).T;idx=origins[:,None,:]+local[None,:,:]
    cells=(idx[:,:,0]*65+idx[:,:,1])*65+idx[:,:,2]
    def grads(ref):return el.tabulate(1,ref)[1:,:,:,:][:,:,:,0].transpose(1,2,0)[:,order,:]*N
    g=grads(qp[0]*N);checks={};summaries={};states={}
    def components(ep,n,ph):
        en=np.einsum('...ij,...j->...i',ep,n);enn=np.sum(en*n,axis=-1)
        ent=en-enn[...,None]*n
        eT=ep-en[..., :,None]*n[...,None,:]-n[..., :,None]*en[...,None,:]+enn[...,None,None]*n[..., :,None]*n[...,None,:]
        trT=np.trace(eT,axis1=-2,axis2=-1);pred=-LAM/(LAM+2*MU)*trT
        scale=ETA+(1-ETA)*ph
        ps=scale*(.5*LPS*trT**2+MU*np.sum(eT*eT,axis=(-1,-2)))
        un=scale*.5*(LAM+2*MU)*(enn-pred)**2
        us=scale*2*MU*np.sum(ent*ent,axis=-1)
        U=scale*(.5*LAM*np.trace(ep,axis1=-2,axis2=-1)**2+MU*np.sum(ep*ep,axis=(-1,-2)))
        return {'total':U,'tangential_ps':ps,'normal':un,'shear':us,'enn':enn,'pred':pred,
          'normal_mismatch':enn-pred,'ent_norm':np.linalg.norm(ent,axis=-1),
          'unit_normal_stress':(LAM+2*MU)*(enn-pred),'actual_normal_stress':scale*(LAM+2*MU)*(enn-pred),
          'actual_shear_stress_norm':2*scale*MU*np.linalg.norm(ent,axis=-1)}
    def strain_stats(d,w):
        denom=np.sum(w*d['pred']**2);ws=w.sum()
        return {'normal_prediction_rms':float(np.sqrt(denom/ws)),
          'normal_strain_rms':float(np.sqrt(np.sum(w*d['enn']**2)/ws)),
          'normal_mismatch_rms':float(np.sqrt(np.sum(w*d['normal_mismatch']**2)/ws)),
          'shear_strain_vector_rms':float(np.sqrt(np.sum(w*d['ent_norm']**2)/ws)),
          'normal_vs_ps_prediction_least_squares_factor':float(np.sum(w*d['enn']*d['pred'])/denom) if denom>0 else None,
          'normal_mismatch_over_prediction_rms':float(np.sqrt(np.sum(w*d['normal_mismatch']**2)/denom)) if denom>0 else None,
          'actual_normal_stress_rms_MPa':float(np.sqrt(np.sum(w*d['actual_normal_stress']**2)/ws))}
    for label,path,resultfile in [('r30',R30,R30/'jax_result.json'),('r34',R34,R34/'result.json')]:
        st=np.load(path/'linear_state.npz');states[label]={key:st[key] for key in ['q','class_ids','H']}
        q=st['q'];ci=st['class_ids'];H=st['H'];assert H[2,2]==-A
        grad=H+np.einsum('cni,qnj->cqij',q[ci[cells]],g,optimize=True)
        ep=.5*(grad+grad.swapaxes(-1,-2));d=components(ep,n,phi)
        identity=float(np.linalg.norm(d['total']-d['tangential_ps']-d['normal']-d['shear'])/np.linalg.norm(d['total']))
        checks[label+'_pointwise_energy_identity']=identity<=1e-10
        expected=json.loads(resultfile.read_text());full=float(np.sum(d['total']*weight))
        selected_total=float(np.sum(d['total'][sel]*weight[sel]));part=[]
        for name,lo,hi in [('deep_void',0.,.001),('void_tail',.001,.01),('transition',.01,.9),('solid_core',.9,1.0000000001)]:
            mask=(phi[sel]>=lo)&(phi[sel]<hi);w=weight[sel][mask]
            dd={k:a[sel][mask] for k,a in d.items()}
            en={k:float(np.sum(dd[k]*w)) for k in ['total','tangential_ps','normal','shear']}
            en.update({'region':name,'points':int(mask.sum()),'volume_mm3':float(w.sum()),
               'fraction_selected_total':en['total']/selected_total,
               'normal_fraction_selected_total':en['normal']/selected_total,
               'shear_fraction_selected_total':en['shear']/selected_total,
               'normal_plus_shear_fraction_group_energy':(en['normal']+en['shear'])/en['total'] if en['total']>0 else None})
            if len(w):en.update(strain_stats(dd,w))
            part.append(en)
        bins=[];edges=cfg['distance_bins_mm']
        for lo,hi in zip(edges,edges[1:]+[None]):
            mask=(distance[sel]*L>=lo)
            if hi is not None:mask&=distance[sel]*L<hi
            ww=weight[sel][mask];dd={k:a[sel][mask] for k,a in d.items()}
            row={'distance_from_mm':lo,'distance_to_mm':hi,'points':int(mask.sum())}
            for key in ['total','tangential_ps','normal','shear']:row[key+'_N_mm']=float(np.sum(dd[key]*ww))
            if len(ww):row.update(strain_stats(dd,ww))
            bins.append(row)
        if label=='r30':
            checks['r30_original_full_energy']=abs(full/expected['energy_N_mm']-1)<=1e-10
            original=json.loads((R33/'result.json').read_text())['selected_original']['total_energy_N_mm']
            checks['r30_selected_original_energy']=abs(selected_total/original-1)<=1e-10
            prior=json.loads((R32/'analysis.json').read_text())['background']
            normal_full=float(np.sum(d['normal']*weight));shear_full=float(np.sum(d['shear']*weight))
            checks['r32_normal_and_shear_reproduced']=max(abs(normal_full/prior['normal_relaxation_energy_N_mm']-1),
                  abs(shear_full/prior['transverse_shear_energy_N_mm']-1))<=1e-10
        summaries[label]={'own_equilibrium_energy_N_mm':expected['energy_N_mm'],
          'common_original27_full_energy_N_mm':full,'common_original27_selected_energy_N_mm':selected_total,
          'pointwise_identity_relative':identity,'partitions':part,'distance_bins':bins,
          'nearest_edge_frame_flag_energy_fraction':float(np.sum(d['total'][sel]*weight[sel]*(edge_ratio[sel]>.05))/selected_total),
          'common_rule_is_not_r34_own_mixed_rule':label=='r34'}
        print(json.dumps({'stage':'Gauss_partitions','state':label,'seconds':time.perf_counter()-start}),flush=True)
    # Fixed normal lines are observational probes, not an integration rule.
    zz=np.linspace(*cfg['line_z_range_mm'],cfg['line_points'])
    x=centers[pf,None,:]+normals[pf,None,:]*zz[None,:,None]/L
    dl,fl,_=surface.query(x,details=True);phl=expit((.25-dl*L)/ELL)
    line_n=np.broadcast_to(normals[pf,None,:],x.shape);xf=x.reshape(-1,3)%1
    cellid=np.floor(xf*N).astype(int);ref=xf*N-cellid
    cind=(cellid[:,0]*N+cellid[:,1])*N+cellid[:,2]
    lines={};profile_checks={}
    for label,st in states.items():
        flat={};q=st['q'];ci=st['class_ids'];H=st['H']
        for pos in range(0,len(xf),BATCH):
            if time.perf_counter()-start>BUDGET:raise TimeoutError('frozen total postprocessing budget exceeded')
            stop=min(pos+BATCH,len(xf));gg=grads(ref[pos:stop])
            gu=H+np.einsum('pni,pnj->pij',q[ci[cells[cind[pos:stop]]]],gg,optimize=True)
            ep=.5*(gu+gu.swapaxes(-1,-2))
            dd=components(ep,line_n.reshape(-1,3)[pos:stop],phl.ravel()[pos:stop])
            for key,val in dd.items():
                if key not in flat:flat[key]=np.empty(len(xf))
                flat[key][pos:stop]=val
        values={key:val.reshape(len(pf),len(zz)) for key,val in flat.items()}
        wa=area[pf,None];ws=wa.sum(axis=0)
        rms=lambda a:np.sqrt(np.sum(wa*a*a,axis=0)/ws)
        denom=np.sum(wa*values['pred']**2,axis=0)
        profile={'normal_strain_rms':rms(values['enn']),
         'plane_stress_normal_prediction_rms':rms(values['pred']),
         'normal_mismatch_rms':rms(values['normal_mismatch']),
         'shear_strain_vector_rms':rms(values['ent_norm']),
         'normal_vs_ps_prediction_fit':np.sum(wa*values['enn']*values['pred'],axis=0)/denom,
         'actual_normal_stress_rms_MPa':rms(values['actual_normal_stress']),
         'unit_normal_stress_rms_MPa':rms(values['unit_normal_stress']),
         'actual_shear_stress_rms_MPa':rms(values['actual_shear_stress_norm']),
         'normal_to_total_area_weighted_energy_ratio':np.sum(wa*values['normal'],axis=0)/np.sum(wa*values['total'],axis=0),
         'shear_to_total_area_weighted_energy_ratio':np.sum(wa*values['shear'],axis=0)/np.sum(wa*values['total'],axis=0)}
        checks[label+'_finite_profiles']=all(np.isfinite(a).all() for a in profile.values())
        lines[label]={key:val.tolist() for key,val in profile.items()}
        # Summary in pre-frozen normal-line windows; no small-point ratios or adaptive filtering.
        windows=[]
        for name,lo,hi in [('midwall',0.,.1),('inner_wall',.1,.2),('near_interface',.2,.275),('outside_band',.275,.35+1e-12)]:
            mask=(abs(zz)>=lo)&(abs(zz)<hi);wz=np.broadcast_to(wa,(len(pf),mask.sum()))
            dd={k:a[:,mask] for k,a in values.items()}
            row={'window':name,'abs_z_from_mm':lo,'abs_z_to_mm':hi,'point_count':int(mask.sum()*len(pf))}
            row.update(strain_stats(dd,wz));windows.append(row)
        summaries[label]['normal_line_windows']=windows
        print(json.dumps({'stage':'normal_lines','state':label,'points':len(xf),'seconds':time.perf_counter()-start}),flush=True)
    alignment=np.sum(normals[fl]*line_n,axis=-1)
    occupancy_reference=expit((.25-abs(zz))/ELL)
    occupancy_mean=np.sum(area[pf,None]*phl,axis=0)/area[pf].sum()
    geom={'signed_z_mm':zz.tolist(),'area_weighted_occupancy':occupancy_mean.tolist(),
      'planar_reference_occupancy':occupancy_reference.tolist(),
      'closest_face_changed_area_fraction':(np.sum(area[pf,None]*(fl!=pf[:,None]),axis=0)/area[pf].sum()).tolist(),
      'normal_alignment_area_weighted_mean':(np.sum(area[pf,None]*alignment,axis=0)/area[pf].sum()).tolist(),
      'normal_frame_is_a_facet_proxy':True,'changed_faces_not_filtered':True}
    checks['budget']=time.perf_counter()-start<=BUDGET
    out={'status':'ok' if all(checks.values()) else 'not_accepted','checks':checks,
      'states':summaries,'line_profiles':lines,'line_geometry':geom,
      'selected_cells':len(sel),'profile_faces':len(pf),'profile_face_area_mm2':float(area[pf].sum()),
      'definition':'Common original Gauss partitions plus fixed normal-line probes; pointwise plane-stress residual and shear split are diagnostic, not compatible relaxation or causal stiffness decomposition.',
      'no_new_FEM_Abaqus_AD_or_production_change':True,'cause_of_5pct_certified':False,
      'wall_seconds':time.perf_counter()-start}
    write('result.json',out);freeze();print(json.dumps({'status':out['status'],'checks':checks,'wall_seconds':out['wall_seconds']},indent=2),flush=True)
    if out['status']!='ok':sys.exit(2)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);a=p.parse_args()
    (prepare if a.action=='prepare' else run)()

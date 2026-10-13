"""Postprocess this one declared simplified-boundary pair; no mechanics changes."""
from pathlib import Path
import csv,hashlib,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
E=Path(__file__).resolve().parents[1];D=E.parents[1];R=D.parents[1]
sys.path.insert(0,str(D/'tools'))
sys.path.insert(0,str(R))
O=E/'analysis';O.mkdir(exist_ok=True)  # Regenerate this experiment's derived tables/figures only.
cfg=json.loads((E/'input/protocol.json').read_text());T=cfg['load_time_s'];L=cfg['L_mm']
def load(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def integrate(y,x):return float(np.sum((y[1:]+y[:-1])*.5*np.diff(x)))
def arr(result,key):return np.array([x[key] for x in result['path']],dtype=float)
def time_average(res,key,a,b):
    t=arr(res,'time');y=arr(res,key);inside=t[(t>a)&(t<b)]
    xx=np.r_[a,inside,b];return integrate(np.interp(xx,t,y),xx)/(b-a)
def curve(res,end):
    t=arr(res,'time');mask=t<min(T,end)-1e-13
    eps=np.r_[arr(res,'compression')[mask],np.interp(min(T,end),t,arr(res,'compression'))]
    force=np.r_[arr(res,'force_N')[mask],np.interp(min(T,end),t,arr(res,'force_N'))]
    # Snap only the completed target coordinate within ODB float precision.
    if min(T,end)>=T-1e-13 and abs(eps[-1]-cfg['target_compression'])<1e-8:
        eps[-1]=cfg['target_compression']
    ue,rev=np.unique(eps[::-1],return_index=True);return ue,force[len(eps)-1-rev]
j=load(E/'jax/result.json') if (E/'jax/result.json').exists() else None
a=load(E/'abaqus/result.json')
if j is None and (E/'jax/path_latest.json').exists():j={'complete':False,'path':load(E/'jax/path_latest.json'),'status':'interrupted_saved_path'}
if j is None:raise ValueError('No JAX forward path available')
jt=arr(j,'time');at=arr(a,'time');end=min(T,jt[-1],at[-1]);ej,fj=curve(j,end);ea,fa=curve(a,end)
emax=min(ej[-1],ea[-1]);grid=np.linspace(0,emax,1001);J=np.interp(grid,ej,fj);A=np.interp(grid,ea,fa)
ae,af=curve(a,T);den=float(max(np.max(abs(arr(a,'force_N')[at<=T])),abs(np.interp(T,at,arr(a,'force_N')))));fixed=.7241182386842194
primary=float(np.sqrt(np.mean((J-A)**2)));workj=integrate(fj,ej*L);worka=integrate(fa,ea*L)
metrics={'scope':cfg['stage'],'JAX_status':j['status'],'JAX_complete':j['complete'],'Abaqus_complete':a['complete'],
         'paired_full20_and_hold':bool(j['complete'] and a['complete'] and jt[-1]>=.0042-1e-10),
         'compared_compression_max':float(emax),'primary_shell_full20_peak_N':den,'contextual_original2pct_peak_N':fixed,
         'curve_RMS_N':primary,'curve_RMS_over_new_shell_peak':primary/den,'curve_RMS_over_old_fixed_peak':primary/fixed,
         'common_loading_work_N_mm':{'JAX':workj,'Abaqus':worka,'relative_bias':workj/worka-1},
         'segments':[],'sample_force':[],'new_design_AD':False,'physical_finite_width_contact_validated':False}
for lo,hi in [(0,.02),(.02,.10),(.10,.20)]:
    if hi>emax+1e-10:continue
    mask=(grid>=lo)&(grid<=hi);v=float(np.sqrt(np.mean((J[mask]-A[mask])**2)))
    metrics['segments'].append({'compression':[lo,hi],'RMS_N':v,'RMS_over_same_full20_shell_peak':v/den})
for val in [.02,.05,.10,.15,.20]:
    if val>emax+1e-10:continue
    vj=float(np.interp(val,ej,fj));va=float(np.interp(val,ea,fa))
    metrics['sample_force'].append({'compression':val,'JAX_N':vj,'Abaqus_N':va,'relative_bias':vj/va-1 if abs(va)>1e-12 else None})
metrics['peak_in_compared_window']={}
for label,res in [('JAX',j),('Abaqus',a)]:
    times=arr(res,'time');mask=times<end
    force=np.r_[arr(res,'force_N')[mask],np.interp(end,times,arr(res,'force_N'))]
    eps=np.r_[arr(res,'compression')[mask],np.interp(end,times,arr(res,'compression'))]
    peak=int(np.argmax(force));metrics['peak_in_compared_window'][label]={'force_N':float(force[peak]),'compression':float(eps[peak])}
pj=metrics['peak_in_compared_window']['JAX'];pa=metrics['peak_in_compared_window']['Abaqus']
metrics['peak_force_relative_bias']=pj['force_N']/pa['force_N']-1
metrics['peak_compression_difference_percentage_points']=100*(pj['compression']-pa['compression'])
if metrics['paired_full20_and_hold']:
    metrics['hold_last0p1ms']={}
    for key in ['force_N','bottom_force_N']:
        mj=time_average(j,key,.0041,.0042);ma=time_average(a,key,.0041,.0042)
        metrics['hold_last0p1ms'][key]={'JAX_N':mj,'Abaqus_N':ma,'relative_bias':mj/ma-1}
    metrics['three_scalar_targets_within_10pct']=bool(primary/den<=.10 and abs(workj/worka-1)<=.10 and abs(metrics['hold_last0p1ms']['force_N']['relative_bias'])<=.10)
else:metrics['three_scalar_targets_within_10pct']=False
energy=arr(j,'energy_N_mm')+arr(j,'contact_energy_N_mm')+arr(j,'KE_N_mm')
travel=arr(j,'compression')*L;force=arr(j,'force_N')
work=np.r_[0,np.cumsum((force[1:]+force[:-1])*.5*np.diff(travel))]
momentum=arr(j,'momentum_z_N_s');net=arr(j,'bottom_force_N')-force
impulse=np.r_[0,np.cumsum((net[1:]+net[:-1])*.5*np.diff(jt))]
metrics['JAX_sampled_health']={'minimum_required_actual_J':float(np.min(arr(j,'required_positive_J_min'))),
    'minimum_all_actual_J':float(np.min(arr(j,'all_J_min'))),'max_negative_void_J_samples':int(np.max(arr(j,'negative_void_J_points'))),
    'energy_work_defect_over_max_sampled_work':float(np.max(abs(energy-energy[0]-work))/max(np.max(abs(work)),1e-30)),
    'impulse_defect_over_max_momentum':float(np.max(abs(momentum-momentum[0]-impulse))/max(np.max(abs(momentum)),1e-30)),
    'terminal_KE_over_IE':float(arr(j,'KE_over_IE')[-1]),'steps':j.get('steps'),'wall_seconds':j.get('wall_seconds'),
    'rejected_blocks':len(j.get('rejections',[])),'initial_dt_s':j.get('initial_dt_s'),'terminal_dt_s':j.get('terminal_dt_s'),
    'validation_scope':'Accepted block samples and selected tangent checks; not every-step or global contact certification.'}
metrics['Abaqus_health']={key:a[key] for key in ['XY_U_error_mm','XY_UR_error_rad','terminal_rim_penetration_mm','energy_drift_relative_work','terminal_KE_over_IE']}
with (O/'force_comparison.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['engineering_compression','JAX_upper_force_N','Abaqus_upper_force_N','difference_over_shell_full20_peak']);w.writerows(zip(grid,J,A,(J-A)/den))
for label,res in [('JAX',j),('Abaqus',a)]:
    keys=list(res['path'][0])
    with (O/(label+'_path.csv')).open('w',newline='') as f:
        w=csv.writer(f);w.writerow(keys);w.writerows([[json.dumps(x.get(k)) if isinstance(x.get(k),(list,dict)) else x.get(k) for k in keys] for x in res['path']])
plt.rcParams.update({'font.size':10,'figure.dpi':160,'savefig.dpi':180,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
for ax in axs[0]:
    ax.plot(ea*100,fa,label='Abaqus S3R / Explicit',color='#285c8e');ax.plot(ej*100,fj,label='JAX HEX27 / Verlet',color='#c45b25',ls='--')
    ax.set(xlabel='Engineering compression (%)',ylabel='Upper-platen compressive force (N)');ax.legend(fontsize=9);ax.grid(alpha=.2)
axs[0,0].set_title('Full loading: midsurface-rim contact, free XY');axs[0,1].set(xlim=(0,5),title='Initial segment (same new 20% ramp)')
axs[1,0].plot(jt*1000,force,label='JAX upper',color='#c45b25',ls='--');axs[1,0].plot(at*1000,arr(a,'force_N'),label='Abaqus upper',color='#285c8e')
axs[1,0].axvline(4,color='gray',ls=':');axs[1,0].set(xlabel='Physical time (ms)',ylabel='Compressive force (N)',title='Loading and 0.2 ms hold');axs[1,0].legend()
for key,style in [('ax_mm','-'),('ay_mm','--')]:
    axs[1,1].plot(jt*1000,arr(j,key),style,color='#c45b25',label='JAX '+key[:2]);axs[1,1].plot(at*1000,arr(a,key),style,color='#285c8e',label='Abaqus '+key[:2])
axs[1,1].set(xlabel='Physical time (ms)',ylabel='Macro displacement jump (mm)',title='Free transverse response');axs[1,1].legend(ncol=2)
for ax in axs[1]:ax.grid(alpha=.2)
fig.suptitle('G | L10 mm, t0.5 mm, NH E10 MPa | 4 ms + 0.2 ms | simplified rim loading')
fig.savefig(O/'force_and_lateral.png');plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
for res,t,label,ie,ke in [(j,jt,'JAX','energy_N_mm','KE_N_mm'),(a,at,'Abaqus','ALLIE','ALLKE')]:
    axs[0,0].plot(t*1000,arr(res,ie),label=label+' internal');axs[0,0].plot(t*1000,arr(res,ke),'--',label=label+' kinetic')
    mask=t>1e-5;axs[0,1].plot(t[mask]*1000,arr(res,ke)[mask]/np.maximum(abs(arr(res,ie)[mask]),1e-30),label=label)
axs[0,0].set(ylabel='Energy (N mm)',title='Internal and kinetic energy');axs[0,1].set(ylabel='KE / |IE|',title='Dynamic indicator; not a quasi-static declaration')
axs[1,0].plot(jt*1000,energy-energy[0],label='JAX internal + contact + kinetic');axs[1,0].plot(jt*1000,work,'--',label='JAX sampled external work')
axs[1,0].set(ylabel='Energy (N mm)',title='JAX sampled work balance')
axs[1,1].plot(jt*1000,arr(j,'required_positive_J_min'),label='Required actual J min');axs[1,1].plot(jt*1000,arr(j,'all_J_min'),label='All-domain actual J min')
axs[1,1].set(ylabel='Jacobian determinant',title='Actual J retained, including virtual-domain flips')
for ax in axs.flat:ax.set_xlabel('Physical time (ms)');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.savefig(O/'energy_and_J.png');plt.close(fig)
# Field comparison at the same time, with Abaqus output-time interpolation disclosed.
with np.load(R/'validation/canonical_tpms_20261009_r50/G/surface_shell_quality.npz') as z:vertices=z['surface_vertices'];tri=z['surface_triangles']
n=cfg['N'];nx=2*n+1
cell=np.clip(np.floor(vertices*n).astype(int),0,n-1);loc=vertices*n-cell
shape=lambda x:np.stack([2*(x-.5)*(x-1),4*x*(1-x),2*x*(x-.5)],axis=-1)
sh=np.einsum('ni,nj,nk->nijk',shape(loc[:,0]),shape(loc[:,1]),shape(loc[:,2])).reshape(-1,27)
offsets=np.indices((3,3,3)).reshape(3,-1).T;nodes=2*cell[:,None,:]+offsets
ids=(nodes[:,:,0]*nx+nodes[:,:,1])*nx+nodes[:,:,2]
np.testing.assert_allclose(sh.sum(axis=1),1.,atol=1e-13)
np.testing.assert_allclose(np.einsum('cn,cnj->cj',sh,nodes/(2*n)),vertices,atol=1e-13)
with np.load(E/'abaqus/frames.npz') as z:ft=z['times'];xyz=z['xyz_mm'];fu=z['u_mm'];labels=z['labels']
np.testing.assert_allclose(xyz,vertices*L,atol=2e-6)
b=vertices[:,:2]-.5
from pbc import periodic_node_classes
gridnodes=np.indices((nx,nx,nx)).reshape(3,-1).T/(nx-1)
classes,_=periodic_node_classes(gridnodes,nx-1,nx-1,nx-1,periodic_axes=(0,1))
with np.load(D/'input/cut_face_contact.npz') as z:cap=dict(z)
fieldmetrics=[];states=[]
for target in [5,10,15,20]:
    p=E/'jax'/f'state_{target:02d}pct.npz'
    if not p.exists():continue
    with np.load(p) as z:q=z['q'];macro=z['macro'];ts=float(z['time'])
    # Reuse the exact periodic equivalence classes; no model rebuild.
    nodal=q[np.asarray(classes)]
    uj=np.einsum('cn,cnj->cj',sh,nodal[ids])*L;uj[:,:2]+=b*macro*L
    k=int(np.clip(np.searchsorted(ft,ts),1,len(ft)-1));f=(ts-ft[k-1])/(ft[k]-ft[k-1]);ua=fu[k-1]*(1-f)+fu[k]*f
    delta=uj-ua;item={'target_pct':target,'time_s':ts,'Abaqus_interpolation_bracket_s':[float(ft[k-1]),float(ft[k])],
        'displacement_RMS_mm':float(np.sqrt(np.mean(np.sum(delta**2,axis=1)))),
        'relative_displacement_RMS':float(np.sqrt(np.mean(np.sum(delta**2,axis=1)))/max(np.sqrt(np.mean(np.sum(ua**2,axis=1))),1e-30))}
    for label,u in [('JAX',uj),('Abaqus',ua)]:
        zpos=vertices[:,2]*L+u[:,2];plate=L*(1-cfg['target_compression']*(10*min(ts/T,1)**3-15*min(ts/T,1)**4+6*min(ts/T,1)**5))
        item[label+'_all_midsurface_outside_platens_max_mm']=float(max(np.max(-zpos),np.max(zpos-plate),0))
        xy=vertices[:,:2]*L+u[:,:2]
        item[label+'_midsurface_xy_bounds_mm']=[np.min(xy,axis=0).tolist(),np.max(xy,axis=0).tolist()]
        item[label+'_within_Abaqus_platen_footprint']=bool(np.min(xy)>=-5 and np.max(xy)<=15)
    cz=cap['points'][:,2]*L+np.einsum('cn,cn->c',cap['shape'],nodal[cap['node_ids'],2])*L
    pen=np.maximum(np.where(cap['side']==0,-cz,cz-plate),0)
    item['JAX_finite_width_cut_face_outside_platen_max_mm']={}
    for ph in [.01,.5]:item['JAX_finite_width_cut_face_outside_platen_max_mm'][str(ph)]=float(np.max(pen[cap['phi']>=ph]))
    fieldmetrics.append(item);states.append((target,uj,ua))
write(O/'field_comparison.json',{'scope':'JAX Q2 interpolated midsurface vs Abaqus time-interpolated output fields; not a new equilibrium solution or absence-of-self-contact proof.','states':fieldmetrics})
selected=[x for x in states if x[0] in [10,20]] or states[-2:]
if selected:
    fig=plt.figure(figsize=(12,10));vmax=max(float(np.max(np.linalg.norm(u,axis=1))) for _,uj,ua in selected for u in [uj,ua])
    for col,(target,uj,ua) in enumerate(selected):
        for row,(label,u) in enumerate([('Abaqus',ua),('JAX',uj)]):
            ax=fig.add_subplot(2,len(selected),row*len(selected)+col+1,projection='3d');pos=vertices*L+u
            colors=plt.cm.viridis(np.mean(np.linalg.norm(u,axis=1)[tri],axis=1)/max(vmax,1e-12))
            ax.add_collection3d(Poly3DCollection(pos[tri],facecolors=colors,edgecolors='none'))
            ax.set(xlim=(-1,12),ylim=(-1,12),zlim=(-1,11),xlabel='X (mm)',ylabel='Y (mm)',zlabel='Z (mm)',title=f'{label} | near {target}% | displacement scale 1')
            ax.set_box_aspect((1,1,1));ax.view_init(20,-60);ax.set_title(f'{label} | near {target}% | displacement scale 1',fontsize=10,pad=3);ax.tick_params(labelsize=8)
    fig.suptitle('Same reference midsurface; actual displacements (Abaqus fields interpolated in output time)')
    fig.subplots_adjust(left=.02,right=.87,bottom=.07,top=.92,wspace=.08,hspace=.24)
    cax=fig.add_axes([.93,.15,.015,.65])
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0,vmax),cmap='viridis'),cax=cax,label='|U| (mm)')
    fig.savefig(O/'deformation_10_20pct.png');plt.close(fig)
metrics['field_comparison']=fieldmetrics
write(O/'metrics.json',metrics)
print(json.dumps({k:v for k,v in metrics.items() if k not in ['field_comparison']},indent=2))

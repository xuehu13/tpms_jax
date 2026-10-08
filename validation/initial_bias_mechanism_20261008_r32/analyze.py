"""Locate existing r30 energy; no new equilibrium, forward path, or AD."""
from pathlib import Path
import json,sys,time,hashlib
R=Path('/home/xuehu/projects/tpms_jax');D=Path(__file__).resolve().parent;sys.path.insert(0,str(R))
import numpy as np,basix
from jax_fem.basis import get_elements
from surface_distance import PeriodicSurfaceDistance
from hyperelastic_fem import MU,KAPPA
start=time.perf_counter();t=.5;L=10.
def write(n,v):(D/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
cache=np.load(R/'validation/geometry_transfer_20261006_r15/gauss_field.npz')
v=cache['surface_vertices'];tri=cache['surface_triangles'];labels=cache['element_ids'];phi=cache['rho'];w=cache['JxW'];qp=cache['physical_quad_points']
cross=np.cross(v[tri[:,1]]-v[tri[:,0]],v[tri[:,2]]-v[tri[:,0]]);norm=np.linalg.norm(cross,axis=1)
normal=cross/norm[:,None];area=.5*norm*L**2;centers=np.mean(v[tri],axis=1)
raw=json.loads((D/'shell_sections.json').read_text());S=raw['fields']['S'];E=raw['fields']['E']
assert S['labels']==['S11','S22','S33','S12'] and E['labels']==['E11','E22','E33','E12']
assert E['engineering_tensor'] and not S['engineering_tensor']
def section(f,sp):
    rows=[row for row in f['rows'] if row[2]==sp];bylabel={row[0]:row for row in rows}
    assert len(rows)==len(tri)==len(bylabel) and all(row[1]==1 for row in rows)
    return np.array([bylabel[int(label)][3:] for label in labels])
slo,shi=section(S,1),section(S,5);elo,ehi=section(E,1),section(E,5)
s0=(slo+shi)/2;e0=(elo+ehi)/2;sk=(shi-slo)/t;ek=(ehi-elo)/t
# E12 is engineering shear: its product with S12 appears once, not twice.
mem=.5*np.sum(s0*e0,axis=1)*area*t
bend=.5*np.sum(sk*ek,axis=1)*area*t**3/12
sh=json.loads((R/'validation/initial_tangent_20261008_r30/abaqus/shell_result.json').read_text())
ALLSE=sh['energy_N_mm'];recovered=float((mem+bend).sum());residual=(ALLSE-recovered)/ALLSE
shell={'membrane_N_mm':float(mem.sum()),'bending_N_mm':float(bend.sum()),'recovered_N_mm':recovered,
 'membrane_fraction_of_ALLSE':float(mem.sum()/ALLSE),'bending_fraction_of_ALLSE':float(bend.sum()/ALLSE),
 'unreconstructed_fraction_of_ALLSE':residual,'ALLSE_N_mm':ALLSE,
 'energy_recovery_within_1pct':abs(residual)<=.01,'interpretation':'Initial linear section S/E; transverse shear not requested in the original ODB. Recovered membrane/bending only, residual is not automatically assigned to any unique mechanism.'}
assert np.min(mem)>-1e-18 and np.min(bend)>-1e-18
state=np.load(R/'validation/initial_tangent_20261008_r30/linear_state.npz');q=state['q'];classids=state['class_ids'];H=state['H']
family,cell,_,_,degree,order=get_elements('HEX27');el=basix.create_element(family,cell,degree)
local=np.rint(2*el.points[order]).astype(int);origins=2*np.indices((32,)*3).reshape(3,-1).T
idx=origins[:,None,:]+local[None,:,:];cells=(idx[:,:,0]*65+idx[:,:,1])*65+idx[:,:,2]
g=el.tabulate(1,qp[0]*32)[1:,:,:,:][:,:,:,0].transpose(1,2,0)[:,order,:]*32
grad=H+np.einsum('cni,qnj->cqij',q[classids[cells]],g,optimize=True)
eps=.5*(grad+grad.swapaxes(-1,-2));s=1e-4+(1-1e-4)*phi;lam=KAPPA-2*MU/3
stress=s[:,:,None,None]*(lam*np.trace(eps,axis1=-2,axis2=-1)[:,:,None,None]*np.eye(3)+2*MU*eps)
bgenergy=.5*np.sum(stress*eps,axis=(-1,-2))*w*L**3
j=json.loads((R/'validation/initial_tangent_20261008_r30/jax_result.json').read_text())
recompute_error=abs(bgenergy.sum()/j['energy_N_mm']-1);assert recompute_error<=1e-10
surface=PeriodicSurfaceDistance(v,tri);_,faces,_=surface.query(qp.reshape(-1,3),details=True)
n=normal[faces];st=stress.reshape(-1,3,3);ep=eps.reshape(-1,3,3);sf=s.ravel();wf=w.ravel()*L**3
en=np.einsum('nij,nj->ni',ep,n);enn=np.sum(en*n,axis=1)
eT=ep-en[:,:,None]*n[:,None,:]-n[:,:,None]*en[:,None,:]+enn[:,None,None]*n[:,:,None]*n[:,None,:]
lambda_ps=2*MU*lam/(lam+2*MU)
uT=sf*(.5*lambda_ps*np.trace(eT,axis1=-2,axis2=-1)**2+MU*np.sum(eT*eT,axis=(1,2)))*wf
sn=np.einsum('nij,nj->ni',st,n);snn=np.sum(sn*n,axis=1);snt=sn-snn[:,None]*n
un=snn**2/(2*sf*(lam+2*MU))*wf;us=np.sum(snt*snt,axis=1)/(2*sf*MU)*wf
identity=float(np.linalg.norm(uT+un+us-bgenergy.ravel())/np.linalg.norm(bgenergy.ravel()));assert identity<1e-10
face_bg=np.bincount(faces,weights=bgenergy.ravel(),minlength=len(tri))
face_bg_T=np.bincount(faces,weights=uT,minlength=len(tri))
# Initial shell strain is available locally: recover the same homogeneous plane-stress law.
E0=10.;nu=.3;pred=np.empty_like(s0)
pred[:,0]=E0/(1-nu**2)*(e0[:,0]+nu*e0[:,1]);pred[:,1]=E0/(1-nu**2)*(e0[:,1]+nu*e0[:,0])
pred[:,2]=0.;pred[:,3]=MU*e0[:,3]
constit=float(np.linalg.norm(pred-s0)/np.linalg.norm(s0));assert constit<1e-6
# Fixed area ranking; no individual empty-triangle ratio is used.
density=(mem+bend)/area;rank=np.argsort(-density);cum=np.cumsum(area[rank])/area.sum();top=rank[cum<=.2]
if not len(top):top=rank[:1]
topmask=np.zeros(len(tri),dtype=bool);topmask[top]=True
groups=[]
for name,mask in [('top20pct_area',topmask),('remaining_area',~topmask)]:
    groups.append({'region':name,'area_fraction':float(area[mask].sum()/area.sum()),
      'shell_membrane_N_mm':float(mem[mask].sum()),'shell_bending_N_mm':float(bend[mask].sum()),
      'shell_recovered_energy_fraction':float((mem+bend)[mask].sum()/recovered),
      'background_nearest_face_energy_fraction':float(face_bg[mask].sum()/bgenergy.sum()),
      'background_nearest_face_energy_N_mm':float(face_bg[mask].sum()),
      'background_tangential_plane_stress_energy_N_mm':float(face_bg_T[mask].sum())})
# Normal variation proxy across periodic adjacent triangles, not exact principal curvature.
_,vid=np.unique(np.round(v%1,12),axis=0,return_inverse=True)
edges={};sumk=np.zeros(len(tri));count=np.zeros(len(tri),int);bad=0
for face,verts in enumerate(vid[tri]):
 for a,b in [(verts[0],verts[1]),(verts[1],verts[2]),(verts[2],verts[0])]:edges.setdefault(tuple(sorted((a,b))),[]).append(face)
for adjacent in edges.values():
 if len(adjacent)!=2:bad+=1;continue
 a,b=adjacent;delta=centers[a]-centers[b];delta-=np.rint(delta);dist=np.linalg.norm(delta)*L
 angle=np.arccos(np.clip(normal[a]@normal[b],-1,1));val=(angle/dist)**2
 sumk[[a,b]]+=val;count[[a,b]]+=1
proxy=np.sqrt(sumk/np.maximum(count,1));assert bad==0
weighted=lambda data,weight:float(np.sum(data*weight)/np.sum(weight))
summary={'status':'ok','scope':'Read existing r30 zero-state equilibrium only; no new FEM/Abaqus job or AD',
 'shell':shell,'shell_local_initial_constitutive_relative_error':constit,
 'background':{'recomputed_energy_relative_error':float(recompute_error),'pointwise_energy_identity_relative_error':identity,
  'tangential_plane_stress_energy_N_mm':float(uT.sum()),'normal_relaxation_energy_N_mm':float(un.sum()),
  'transverse_shear_energy_N_mm':float(us.sum()),'sum_is_original_total':True,
  'interpretation':'Pointwise algebraic split in nearest-face frames; no global strain compatibility, no causal contribution, no membrane/bending split of the background.'},
 'groups':groups,'normal_variation_proxy':{'units':'1/mm','area_weighted_mean':weighted(proxy,area),
 'shell_energy_weighted_mean':weighted(proxy,mem+bend),'top20_area_weighted_mean':weighted(proxy[topmask],area[topmask]),
 'remaining_area_weighted_mean':weighted(proxy[~topmask],area[~topmask]),'nonmanifold_periodic_edges':bad,
 'interpretation':'Facet normal angle divided by periodic centroid distance, averaged over adjacent faces; not principal curvature, thickness/radius, or proof of shell validity.'},
 'causal_factor_identified':False,'full_20pct_or_design_AD_certified':False,'postprocess_seconds':time.perf_counter()-start}
np.savez_compressed(D/'local_energy.npz',labels=labels,membrane=mem,bending=bend,background=face_bg,
 background_tangential=face_bg_T,area_mm2=area,normal_variation_proxy=proxy,topmask=topmask)
write('analysis.json',summary);print(json.dumps(summary,indent=2))
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained')
vals=[shell['membrane_fraction_of_ALLSE']*100,shell['bending_fraction_of_ALLSE']*100,residual*100]
axes[0].bar(['Membrane','Bending','Unreconstructed'],vals,color=['#235c91','#81a4bb','#bbbfc4'])
for i,val in enumerate(vals):axes[0].text(i,val+1,f'{val:.2f}%',ha='center')
axes[0].set_ylim(0,105);axes[0].set_ylabel('Fraction of shell ALLSE (%)');axes[0].set_title('Existing S3R initial response')
axes[1].bar([0,1],[groups[0]['shell_recovered_energy_fraction']*100,groups[0]['background_nearest_face_energy_fraction']*100],color=['#81a4bb','#235c91'])
axes[1].set_xticks([0,1],['Shell recovered','Background nearest face']);axes[1].set_ylabel('Energy in fixed top 20% shell area (%)');axes[1].set_ylim(0,105)
for i,gv in enumerate([groups[0]['shell_recovered_energy_fraction'],groups[0]['background_nearest_face_energy_fraction']]):axes[1].text(i,gv*100+1,f'{gv:.2%}',ha='center')
axes[1].set_title('Fixed top 20% area (not a causal test)')
fig.savefig(D/'mechanism_energy.png',dpi=170);plt.close(fig)
for name,h in json.loads((D/'frozen_before.json').read_text()).items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h,name

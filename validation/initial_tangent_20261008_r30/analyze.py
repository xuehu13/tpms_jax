"""Read r30 equilibrium once; regional/normal diagnostics are not causal solves."""
from pathlib import Path
import sys,json,hashlib
R=Path('/home/xuehu/projects/tpms_jax');D=Path(__file__).resolve().parent;sys.path.insert(0,str(R))
import numpy as np
import basix
from jax_fem.basis import get_elements
from surface_distance import PeriodicSurfaceDistance
from hyperelastic_fem import MU,KAPPA
def write(name,d): (D/name).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
j=json.loads((D/'jax_result.json').read_text());sh=json.loads((D/'abaqus/shell_result.json').read_text())
assert j['status']==sh['status']=='ok'
cache=np.load(R/'validation/geometry_transfer_20261006_r15/gauss_field.npz')
state=np.load(D/'linear_state.npz');q=state['q'];classids=state['class_ids'];H=state['H']
phi=cache['rho'];w=cache['JxW'];qp=cache['physical_quad_points'];v=cache['surface_vertices'];tri=cache['surface_triangles']
family,cell,_,_,degree,order=get_elements('HEX27');el=basix.create_element(family,cell,degree)
local=np.rint(2*el.points[order]).astype(int); origins=2*np.indices((32,)*3).reshape(3,-1).T
idx=origins[:,None,:]+local[None,:,:];cells=(idx[:,:,0]*65+idx[:,:,1])*65+idx[:,:,2]
ref=(qp[0]-origins[0]/64)*32
g=el.tabulate(1,ref)[1:,:,:,:][:,:,:,0].transpose(1,2,0)[:,order,:]*32
grad=H+np.einsum('cni,qnj->cqij',q[classids[cells]],g,optimize=True)
eps=.5*(grad+grad.swapaxes(-1,-2));scale=1e-4+(1-1e-4)*phi;lam=KAPPA-2*MU/3
stress=scale[:,:,None,None]*(lam*np.trace(eps,axis1=-2,axis2=-1)[:,:,None,None]*np.eye(3)+2*MU*eps)
energy=.5*np.sum(stress*eps,axis=(-1,-2))*w*1000
recomputed=float(energy.sum());assert abs(recomputed/j['energy_N_mm']-1)<1e-10
bins=[('deep_void',0,.001),('void_transition',.001,.01),('interface',.01,.9),('solid_core',.9,1.0000001)]
regions=[]
for name,lo,hi in bins:
 m=(phi>=lo)&(phi<hi)
 regions.append({'region':name,'phi_lower':lo,'phi_upper':min(hi,1),'points':int(m.sum()),
 'geometric_quadrature_volume_mm3':float(w[m].sum()*1000),'occupancy_volume_mm3':float((phi*w)[m].sum()*1000),
 'stiffness_weighted_volume_mm3':float((scale*w)[m].sum()*1000),
 'energy_N_mm':float(energy[m].sum()),'energy_fraction':float(energy[m].sum()/recomputed),
 'macro_Fz_N':float((stress[:,:,2,2]*w)[m].sum()*100)})
# Local closest-triangle normals only index a saved-state diagnostic. No geometry derivative.
surface=PeriodicSurfaceDistance(v,tri);m=phi>=.01
_,face,_=surface.query(qp[m],details=True)
normal=np.cross(v[tri[:,1]]-v[tri[:,0]],v[tri[:,2]]-v[tri[:,0]])
normal/=np.linalg.norm(normal,axis=1)[:,None];n=normal[face]
st=stress[m];sn=np.einsum('nij,nj->ni',st,n);snn=np.sum(sn*n,axis=1);snt=sn-snn[:,None]*n
normal_release=snn**2/(2*scale[m]*(lam+2*MU))*w[m]*1000
shear_release=np.sum(snt*snt,axis=1)/(2*scale[m]*MU)*w[m]*1000
local={'occupied_point_count':int(m.sum()),'normal_stress_relaxation_energy_N_mm':float(normal_release.sum()),
       'normal_stress_relaxation_fraction_of_total_U':float(normal_release.sum()/recomputed),
       'transverse_shear_relaxation_energy_N_mm':float(shear_release.sum()),
       'transverse_shear_relaxation_fraction_of_total_U':float(shear_release.sum()/recomputed),
       'interpretation':'Pointwise independent strain relaxation at fixed in-plane strain; generally incompatible. Not a globally admissible correction, not causal proof, not a replacement 3D constitutive law.'}
# Evaluate exact Q2 displacement at the existing shell nodes and remove one rigid translation.
coords=np.array([row['xyz_mm'] for row in sh['nodes']])/10
cellindex=np.minimum(np.floor(coords*32).astype(int),31);refnode=coords*32-cellindex
tab=el.tabulate(0,refnode)[0,:,:,0][:,order];cindex=(cellindex[:,0]*32+cellindex[:,1])*32+cellindex[:,2]
bg=(np.einsum('vn,vni->vi',tab,q[classids[cells[cindex]]])+coords@H.T)*10
su=np.array([row['U_mm'] for row in sh['nodes']]);shift=np.mean(bg-su,axis=0);diff=bg-su-shift
alignment={'rigid_translation_removed_mm':shift.tolist(),'RMS_displacement_difference_mm':float(np.sqrt(np.mean(np.sum(diff*diff,axis=1)))),
           'max_displacement_difference_mm':float(np.max(np.linalg.norm(diff,axis=1))),
           'normalization_delta_mm':.001,'RMS_over_applied_delta':float(np.sqrt(np.mean(np.sum(diff*diff,axis=1)))/.001),
           'interpretation':'Unweighted existing-node samples, after removing the different pin translations. Diagnostic of displacement field, not independent mesh accuracy certification.'}
area=float(np.linalg.norm(np.cross(v[tri[:,1]]-v[tri[:,0]],v[tri[:,2]]-v[tri[:,0]]),axis=1).sum()/2)*100
result={'status':'ok','K_background_N_per_mm':j['stiffness_N_per_mm'],'K_shell_N_per_mm':sh['stiffness_N_per_mm'],
 'relative_stiffness_difference':j['stiffness_N_per_mm']/sh['stiffness_N_per_mm']-1,
 'denominator':'new matched initial S3R Standard stiffness','background_energy_N_mm':j['energy_N_mm'],
 'shell_total_energy_N_mm':sh['total_energy_N_mm'],'occupancy_volume_mm3':float((phi*w).sum()*1000),
 'stiffness_weighted_volume_mm3':float((scale*w).sum()*1000),'shell_area_times_thickness_mm3':area*.5,
 'volume_definition_note':'Area times thickness is a shell measure; closest-distance bands on a curved surface are a different 3D geometry. Their volumes are not assumed identical.',
 'regions':regions,'local_normal_diagnostic':local,'displacement_diagnostic':alignment,
 'full_20pct_accuracy_certified':False,'design_gradient_certified':False}
write('comparison.json',result)
print(json.dumps(result,indent=2))
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
k=[result['K_shell_N_per_mm'],result['K_background_N_per_mm']]
ax[0].bar(['S3R / Standard','HEX27 / background'],k,color=['#6c8191','#235c91'],width=.55)
for i,kv in enumerate(k):ax[0].text(i,kv+.06,f'{kv:.4f}',ha='center')
ax[0].set_ylim(0,5.8);ax[0].set_ylabel('Initial stiffness (N/mm)');ax[0].set_title('diverse_04: no inertia, 0.01% compression')
ax[0].text(.5,5.5,f'Background: +{result["relative_stiffness_difference"]:.2%}',ha='center')
labels=['Deep void\nphi < .001','Void gate\n.001-.01','Interface\n.01-.9','Solid core\nphi >= .9']
values=[row['energy_fraction']*100 for row in regions]
ax[1].bar(labels,values,color=['#adb6bc','#93a9b9','#5987a8','#235c91'])
for i,value in enumerate(values):ax[1].text(i,value+1,f'{value:.2f}%',ha='center',fontsize=10)
ax[1].set_ylim(0,105);ax[1].set_ylabel('Fraction of saved-state strain energy (%)')
ax[1].set_title('Regional fractions are not causal contributions')
fig.savefig(D/'initial_stiffness.png',dpi=180);plt.close(fig)
for name,h in json.loads((D/'frozen_before.json').read_text()).items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h,name

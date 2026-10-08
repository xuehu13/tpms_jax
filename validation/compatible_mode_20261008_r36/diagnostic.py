"""One frozen compatible normal bubble; fixed old nodes, no new equilibrium."""
from pathlib import Path
import sys,time,json,hashlib
R=Path('/home/xuehu/projects/tpms_jax');D=Path(__file__).resolve().parent
sys.path.insert(0,str(R))
import numpy as np,basix
from scipy.special import expit
from jax_fem.basis import get_elements
from surface_distance import PeriodicSurfaceDistance
L=10.;N=32;T=.5;ETA=1e-4;NU=.3;E=10.;WIDTH=.05
MU=E/(2*(1+NU));LAM=E*NU/((1+NU)*(1-2*NU))
def write(n,v):(D/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def freeze():
 for n,h in json.loads((D/'frozen_before.json').read_text()).items():assert sha(R/n)==h,n
def main():
 start=time.perf_counter();freeze();cfg=json.loads((D/'protocol.json').read_text())
 sel=np.load(R/'validation/local_quadrature_20261008_r33/selected_cells.npy')
 assert sha(R/'validation/local_quadrature_20261008_r33/selected_cells.npy')==cfg['selected_cells_sha256']
 c=np.load(R/'validation/geometry_transfer_20261006_r15/gauss_field.npz')
 old=np.load(R/'validation/local_reequilibrium_20261008_r34/linear_state.npz')
 base=json.loads((R/'validation/local_reequilibrium_20261008_r34/result.json').read_text())
 surface=PeriodicSurfaceDistance(c['surface_vertices'],c['surface_triangles'])
 family,cell,_,_,degree,order=get_elements('HEX27');el=basix.create_element(family,cell,degree)
 local=np.rint(2*el.points[order]).astype(int)
 origins=np.indices((N,)*3).reshape(3,-1).T/N
 ij=2*np.indices((N,)*3).reshape(3,-1).T[:,None,:]+local[None,:,:]
 cells=(ij[:,:,0]*65+ij[:,:,1])*65+ij[:,:,2]
 uc=old['q'][old['class_ids'][cells]];H=old['H']
 _,faces,_=surface.query(origins[sel]+.5/N,details=True)
 tri=c['surface_vertices'][c['surface_triangles']]
 normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
 lengths=np.linalg.norm(normals,axis=1);assert np.all(lengths>1e-14)
 normals=normals[faces]/lengths[faces,None]
 np.savez_compressed(D/'frozen_mode_geometry.npz',selected_cells=sel,cell_center_faces=faces,normals=normals)
 def strains(ref,which):
  g=el.tabulate(1,ref)[1:,:,:,:][:,:,:,0].transpose(1,2,0)[:,order,:]*N
  grad=H+np.einsum('cni,qnj->cqij',uc[which],g,optimize=True)
  return .5*(grad+grad.swapaxes(-1,-2))
 def mode(ref,n):
  # psi = b(ξ) (n dot (ξ-.5)) n; sign of n cancels.
  v=ref*(1-ref);b=64*np.prod(v,axis=1)
  db=np.empty_like(ref)
  for j in range(3):db[:,j]=64*(1-2*ref[:,j])*np.prod(v[:,[k for k in range(3) if k!=j]],axis=1)
  z=np.einsum('qj,cj->cq',ref-.5,n)
  psi=b[None,:,None]*z[:,:,None]*n[:,None,:]
  grad=n[:,None,:,None]*(db[None,:,None,:]*z[:,:,None,None]+b[None,:,None,None]*n[:,None,None,:])*N
  return psi,.5*(grad+grad.swapaxes(-1,-2))
 def elastic(ep):return .5*LAM*np.trace(ep,axis1=-2,axis2=-1)**2+MU*np.sum(ep*ep,axis=(-1,-2))
 g0ref=c['physical_quad_points'][0]*N
 ep0=strains(g0ref,np.arange(N**3))
 full0=float(np.sum((ETA+(1-ETA)*c['rho'])*elastic(ep0)*c['JxW'])*L**3)
 selected0=float(np.sum((ETA+(1-ETA)*c['rho'][sel])*elastic(ep0[sel])*c['JxW'][sel])*L**3)
 # Independent kinematic identities at every original node and face samples.
 nodes=el.points[order]
 nodeerr=float(np.max(abs(mode(nodes,normals)[0])))
 rng=np.random.default_rng(36);edge=rng.random((128,3));bd=[]
 for j in range(3):
  for v in [0.,1.]:
   a=edge.copy();a[:,j]=v;bd.append(a)
 bderr=float(np.max(abs(mode(np.vstack(bd),normals)[0])))
 signerr=float(np.max(abs(mode(g0ref,normals)[0]-mode(g0ref,-normals)[0])))
 prep=time.perf_counter()-start;rows=[];raw={}
 for level in cfg['points_per_axis']:
  tick=time.perf_counter();z,w=np.polynomial.legendre.leggauss(level);z=(z+1)/2;w=w/2
  ref=np.array([[a,b,c] for a in z for b in z for c in z])
  wt=np.array([a*b*c for a in w for b in w for c in w])*(L/N)**3
  out=np.zeros((len(sel),6));ident=0.
  for pos in range(0,len(sel),32):
   if time.perf_counter()-start>cfg['total_budget_seconds']:raise TimeoutError('Frozen total budget exceeded')
   which=sel[pos:pos+32];n=normals[pos:pos+len(which)]
   d=surface.query(origins[which,None,:]+ref[None,:,:]/N)*L
   phi=expit((T/2-d)/(WIDTH/(2*np.log(9))))
   weight=(ETA+(1-ETA)*phi)*wt
   ep=strains(ref,which);psi,em=mode(ref,n)
   tr=np.trace(ep,axis1=-2,axis2=-1);tm=np.trace(em,axis1=-2,axis2=-1)
   gg=np.sum((LAM*tr*tm+2*MU*np.sum(ep*em,axis=(-1,-2)))*weight,axis=1)
   kk=np.sum((LAM*tm*tm+2*MU*np.sum(em*em,axis=(-1,-2)))*weight,axis=1)
   assert np.all(kk>0) and np.isfinite(kk).all()
   alpha=-gg/kk;release=gg*gg/(2*kk)
   uu=np.sum(elastic(ep)*weight,axis=1)
   newu=np.sum(elastic(ep+alpha[:,None,None,None]*em)*weight,axis=1)
   ident=max(ident,float(np.max(abs(newu-(uu-release)))/base['energy_N_mm']))
   maxdisp=np.max(np.linalg.norm(psi,axis=-1),axis=1)*abs(alpha)*L
   maxstrain=np.max(abs(alpha[:,None,None,None]*em),axis=(1,2,3))
   out[pos:pos+len(which)]=np.stack([uu,gg,kk,alpha,release,maxdisp],axis=1)
   assert np.max(maxstrain)<.02,'Trial left frozen small-strain range'
  raw[level]=out;np.savez_compressed(D/f'level_{level}.npz',per_cell=out,selected_cells=sel,
    columns=np.array(['U_Nmm','g_Nmm','k_Nmm','alpha_normalized','release_Nmm','max_displacement_mm']))
  rows.append({'points_per_axis':level,'selected_energy_N_mm':float(out[:,0].sum()),
   'release_N_mm':float(out[:,4].sum()),'release_over_r34_full_energy':float(out[:,4].sum()/base['energy_N_mm']),
   'minimum_mode_stiffness_N_mm':float(out[:,2].min()),'maximum_mode_amplitude_normalized':float(abs(out[:,3]).max()),
   'maximum_sampled_extra_displacement_mm':float(out[:,5].max()),'energy_identity_relative_to_full':ident,
   'seconds':time.perf_counter()-tick})
  print(json.dumps(rows[-1]),flush=True)
 a,b=raw[8],raw[12]
 stable_u=abs(rows[0]['selected_energy_N_mm']/rows[1]['selected_energy_N_mm']-1)
 stable_release=abs(rows[0]['release_N_mm']/rows[1]['release_N_mm']-1)
 # Sign-sensitive g and positive k checked as whole-region norms, avoiding tiny-cell ratios.
 stable_g=float(np.linalg.norm(a[:,1]-b[:,1])/np.linalg.norm(b[:,1]))
 stable_k=float(np.linalg.norm(a[:,2]-b[:,2])/np.linalg.norm(b[:,2]))
 full12=full0-selected0+rows[1]['selected_energy_N_mm'];repro=abs(full12/base['energy_N_mm']-1)
 checks={'original_nodes_zero':nodeerr<1e-14,'entire_boundary_zero':bderr<1e-14,
  'normal_sign_invariance':signerr<1e-14,'r34_same_rule_energy_reproduced':repro<1e-10,
  'positive_mode_stiffness':all(x['minimum_mode_stiffness_N_mm']>0 for x in rows),
  'energy_identity':max(x['energy_identity_relative_to_full'] for x in rows)<1e-10,
  'two_level_U_g_k_release_stable':max(stable_u,stable_g,stable_k,stable_release)<=cfg['integration_relative_gate'],
  'budget':time.perf_counter()-start<=cfg['total_budget_seconds']}
 result={'status':'ok' if all(checks.values()) else 'not_accepted','checks':checks,'levels':rows,
  'node_zero_error':nodeerr,'boundary_zero_error':bderr,'normal_sign_error':signerr,
  'same_rule_total_energy_N_mm':full12,'r34_full_energy_N_mm':base['energy_N_mm'],
  'reproduction_relative_error':repro,'dense_relative_changes':{'U':stable_u,'g':stable_g,'k':stable_k,'release':stable_release},
  'clear_release_gate_fraction':cfg['clear_release_fraction'],
  'supports_one_global_confirmation':all(checks.values()) and rows[-1]['release_over_r34_full_energy']>=cfg['clear_release_fraction'],
  'is_new_global_stiffness':False,'original_5pct_root_cause_certified':False,
  'preparation_seconds':prep,'wall_seconds':time.perf_counter()-start,'production_changed':False}
 write('result.json',result);freeze();print(json.dumps(result,indent=2),flush=True)
 if not all(checks.values()):sys.exit(2)
if __name__=='__main__':main()

"""Small independent static bending solves using the existing material tangent."""
from pathlib import Path
import os,sys,time,json,hashlib,argparse
os.environ.setdefault('JAX_PLATFORMS','cpu')
R=Path('/home/xuehu/projects/tpms_jax');D=Path(__file__).resolve().parent;sys.path.insert(0,str(R))
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.integrate import quad
import jax,jax.numpy as jnp,basix
from jax_fem.generate_mesh import Mesh
from jax_fem.basis import get_elements
from hyperelastic_fem import DensityHyperelasticity
from surface_distance import thickness_occupancy
E=10.;NU=.3;L=10.;B=1.25;T=.5;H=2.5;ETA=1e-4;ELL=.05/(2*np.log(9));M=1e-4
N=np.array([32,4,8]);Iphysical=B*T**3/12
def write(p,data):p.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def verify_freeze():
 for n,h in json.loads((D/'frozen_before.json').read_text()).items():
  assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
def mesh_problem(zc):
 family,cell,_,_,degree,order=get_elements('HEX27')
 local=np.rint(2*basix.create_element(family,cell,degree).points[order]).astype(int)
 levels=2*N+1;points=np.indices(tuple(levels)).reshape(3,-1).T/64
 origins=2*np.indices(tuple(N)).reshape(3,-1).T
 idx=origins[:,None,:]+local[None,:,:]
 cells=(idx[...,0]*levels[1]+idx[...,1])*levels[2]+idx[...,2]
 p=DensityHyperelasticity(Mesh(points,cells),vec=3,dim=3,ele_type='HEX27',
   quadrature_order=4,dirichlet_bc_info=[[],[],[]])
 p.material_model='objective_void'
 qp=np.asarray(p.physical_quad_points)*L
 rho=thickness_occupancy(np.abs(qp[:,:,2]-zc),T,ELL)
 p.set_params(jnp.zeros((3,3)),rho,ETA)
 return p,local
def continuous_section(zc):
 from scipy.special import expit
 scale=lambda z:ETA+(1-ETA)*expit((T/2-abs(z-zc))/ELL)
 opts={'epsabs':1e-12,'epsrel':1e-12,'points':[zc-T/2,zc,zc+T/2]}
 a=B*quad(lambda z:scale(z),0,H,**opts)[0]
 centroid=B*quad(lambda z:z*scale(z),0,H,**opts)[0]/a
 moment=B*quad(lambda z:(z-centroid)**2*scale(z),0,H,**opts)[0]
 return {'area_mm2':a,'centroid_z_mm':centroid,'I_mm4':moment,'physical_I_mm4':Iphysical}
def tip_load(p,local,zc):
 # Integrate the same 3x3 face quadrature and node convention as Q2 basis.
 family,cell,_,_,degree,order=get_elements('HEX27')
 el=basix.create_element(family,cell,degree)
 x,w=np.polynomial.legendre.leggauss(3);x=(x+1)/2;w=w/2
 ref=np.array([[1.,a,b] for a in x for b in x]);fw=np.array([a*b for a in w for b in w])*(L/32)**2
 shape=el.tabulate(0,ref)[0,:,:,0][:,order]
 origins=2*np.indices(tuple(N)).reshape(3,-1).T/64*L
 selected=np.flatnonzero(np.isclose(origins[:,0],L-L/32))
 fp=origins[selected,None,:]+ref[None,:,:]*(L/32)
 s=np.asarray(ETA+(1-ETA)*thickness_occupancy(abs(fp[:,:,2]-zc),T,ELL))
 area=np.sum(s*fw);centroid=np.sum(s*fw*fp[:,:,2])/area
 I=np.sum(s*fw*(fp[:,:,2]-centroid)**2)
 traction=M*s*(fp[:,:,2]-centroid)/I
 localforce=np.einsum('qn,cq,q->cn',shape,traction,fw)
 f=np.zeros((len(p.fe.points),3));np.add.at(f[:,0],np.asarray(p.fe.cells)[selected].ravel(),localforce.ravel())
 pp=np.asarray(p.fe.points)*L
 resultant=f.sum(axis=0);moment=np.cross(pp-np.array([0,0,centroid]),f).sum(axis=0)
 assert np.linalg.norm(resultant)/(M/L)<1e-8
 assert abs(moment[1]/M-1)<1e-8
 return f.ravel(),{'area_mm2':float(area),'centroid_z_mm':float(centroid),'I_mm4':float(I),
   'resultant_force_N':resultant.tolist(),'applied_moment_N_mm':moment.tolist()}
def solve_phase(name,zc):
 out=D/name;out.mkdir(exist_ok=False);start=time.perf_counter()
 p,local=mesh_problem(zc);prep=time.perf_counter()-start;cells=np.asarray(p.fe.cells)
 scale=np.asarray(p.stiffness_scale);w=np.asarray(p.fe.JxW);grad=np.asarray(p.shape_grads)
 # Exact existing shared constitutive AD, at the initial state only.
 tangent=jax.jit(jax.vmap(jax.jacfwd(p.material_stress,argnums=0)))
 F0=jnp.broadcast_to(jnp.eye(3),(scale.size,3,3))
 A=np.asarray(tangent(F0,jnp.asarray(scale.ravel()))).reshape((*scale.shape,3,3,3,3))
 lam=E*NU/((1+NU)*(1-2*NU));mu=E/(2*(1+NU));delta=np.eye(3)
 analytic=(lam*np.einsum('ij,kl->ijkl',delta,delta)+mu*np.einsum('ik,jl->ijkl',delta,delta)+mu*np.einsum('il,jk->ijkl',delta,delta))
 tangent_error=float(np.max(np.abs(A-scale[:,:,None,None,None,None]*analytic)))
 assert tangent_error<1e-11
 assembling=time.perf_counter()
 Kcell=np.einsum('cqijkl,cqaj,cqbl,cq->caibk',A,grad,grad,w,optimize=True).reshape(len(cells),81,81)*L
 dofs=(3*cells[:,:,None]+np.arange(3)).reshape(len(cells),81)
 rows=np.repeat(dofs,81,axis=1).ravel();cols=np.tile(dofs,(1,81)).ravel()
 K=sp.coo_matrix((Kcell.ravel(),(rows,cols)),shape=(len(p.fe.points)*3,)*2).tocsr()
 K.sum_duplicates();K.eliminate_zeros();assembly=time.perf_counter()-assembling
 f,load=tip_load(p,local,zc)
 volume=np.sum(scale*w)*L**3;z=np.asarray(p.physical_quad_points)[:,:,2]*L
 centroid=np.sum(scale*w*z)*L**3/volume
 Iq=np.sum(scale*w*(z-centroid)**2)*L**3/L
 assert abs(Iq/load['I_mm4']-1)<1e-10
 root=np.flatnonzero(np.isclose(np.asarray(p.fe.points)[:,0],0))
 fixed=(3*root[:,None]+np.arange(3)).ravel();free=np.setdiff1d(np.arange(K.shape[0]),fixed)
 solving=time.perf_counter();u=np.zeros(K.shape[0]);u[free]=spsolve(K[free][:,free].tocsc(),f[free]);solve_seconds=time.perf_counter()-solving
 if solve_seconds>900:raise TimeoutError('Predeclared linear solve budget exceeded; result not accepted')
 residual=K@u-f;relative=float(np.linalg.norm(residual[free])/np.linalg.norm(f[free]))
 U=float(.5*u@(K@u));work=float(.5*f@u);Q=float(f@u/M);compliance=Q/M;EI=M*L/Q
 points=np.asarray(p.fe.points)*L;reaction=residual.reshape(-1,3)[root]
 rf=reaction.sum(axis=0);rm=np.cross(points[root]-[0,0,load['centroid_z_mm']],reaction).sum(axis=0)
 F=np.eye(3)+np.einsum('cni,cqnj->cqij',u.reshape(-1,3)[cells]/L,grad)
 J=np.linalg.det(F);eps=.5*(F+np.swapaxes(F,-1,-2))-np.eye(3)
 symmetry=float(sp.linalg.norm(K-K.T)/sp.linalg.norm(K))
 checks={'free_residual':relative<=1e-8,'load_net_force':np.linalg.norm(load['resultant_force_N'])/(M/L)<=1e-8,
   'reaction_moment':abs(rm[1]/M+1)<=1e-8,'reaction_net_force':np.linalg.norm(rf)/(M/L)<=1e-8,
   'energy_work':abs(U/work-1)<=1e-8,'finite_positive_small_J':np.isfinite(F).all() and J.min()>.98,
   'matrix_symmetry':symmetry<1e-12,'shared_tangent':tangent_error<1e-11}
 section=continuous_section(zc)
 result={'scope':'independent linear static equilibrium of original shared initial material tangent, not finite-strain path or AD',
   'phase_name':name,'midsurface_z_mm':zc,'mesh_cells':len(cells),'mesh_nodes':len(points),'cells_per_axis':N.tolist(),
   'h_mm':L/32,'quad_per_cell':int(p.fe.num_quads),'thickness_mm':T,'E_MPa':E,'nu':NU,'eta':ETA,
   'interface_10_90_mm':.05,'moment_N_mm':M,'Q_rad':Q,'compliance_per_N_mm':compliance,'effective_EI_N_mm2':EI,
   'linear_strain_energy_N_mm':U,'energy_work_relative':float(U/work-1),'free_residual_relative':relative,
   'root_reaction_force_N':rf.tolist(),'root_reaction_moment_N_mm':rm.tolist(),'actual_J_min':float(J.min()),
   'small_strain_norm_max':float(np.linalg.norm(eps,axis=(-2,-1)).max()),'K_symmetry_relative':symmetry,
   'shared_tangent_absolute_error':tangent_error,'load':load,'continuous_section':section,'actual_Gauss_section_I_mm4':float(Iq),
   'Gauss_vs_dense_I_relative':float(Iq/section['I_mm4']-1),'effective_EI_vs_Gauss_EI_relative':float(EI/(E*Iq)-1),
   'checks':{k:bool(v) for k,v in checks.items()},'all_checks_pass':all(checks.values()),
   'cost_seconds':{'preparation':prep,'assembly':assembly,'linear_solve':solve_seconds,'total':time.perf_counter()-start}}
 np.savez_compressed(out/'field.npz',points_mm=points,cells=cells,u_mm=u.reshape(-1,3),rho=np.asarray(p.rho),J=J)
 write(out/'result.json',result)
 print(json.dumps(result),flush=True)
 if not all(checks.values()):raise RuntimeError('Predeclared bending validity check failed; do not retry/relax')
def export_shell():
 out=D/'abaqus';out.mkdir(exist_ok=False);nx=64;ny=8;lines=['*Heading','r27 small initial-tangent pure-moment plate strip','*Node']
 nid=lambda i,j:i*(ny+1)+j+1
 for i in range(nx+1):
  for j in range(ny+1):lines.append(f'{nid(i,j)}, {i*L/nx:.16g}, {j*B/ny:.16g}, 1.25')
 lines.append('*Element, type=S4R, elset=PLATE')
 for i in range(nx):
  for j in range(ny):lines.append(f'{i*ny+j+1}, {nid(i,j)}, {nid(i+1,j)}, {nid(i+1,j+1)}, {nid(i,j+1)}')
 lines+=['*Nset, nset=ROOT',', '.join(str(nid(0,j)) for j in range(ny+1)),
   '*Nset, nset=TIP',', '.join(str(nid(nx,j)) for j in range(ny+1)),
   '*Material, name=INITIAL_NH_TANGENT','*Elastic','10., 0.3',
   '*Shell Section, elset=PLATE, material=INITIAL_NH_TANGENT','0.5, 5',
   '*Boundary','ROOT, 1, 6, 0.', '*Step, name=SMALL_BENDING, nlgeom=NO','*Static','1., 1.', '*Cload']
 tip=[]
 for j in range(ny+1):
  weight=(.5 if j in (0,ny) else 1.)/ny;tip.append({'node':nid(nx,j),'weight':weight})
  lines.append(f'{nid(nx,j)}, 5, {M*weight:.16g}')
 lines+=['*Output, field','*Node Output','U, UR, RF, RM','*Element Output','S, E',
   '*Output, history','*Energy Output','ALLSE, ALLIE, ALLAE, ALLWK','*End Step']
 (out/'plate.inp').write_text('\n'.join(lines)+'\n')
 write(out/'input.json',{'L_mm':L,'B_mm':B,'t_mm':T,'E_MPa':E,'nu':NU,'M_N_mm':M,
   'tip_weights':tip,'physical_I_mm4':Iphysical,'analytic_Q_rad':M*L/(E*Iphysical),
   'analytic_compliance_per_N_mm':L/(E*Iphysical),'nx':nx,'ny':ny,
   'material_scope':'linear elastic initial NH tangent; no finite strain certification'})
 print('SHELL_PACKAGE_READY',out)
def analyze():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 data=[json.loads((D/n/'result.json').read_text()) for n in ['phase_layer','phase_half']]
 shell=json.loads((D/'abaqus/shell.json').read_text());C=shell['compliance_per_N_mm']
 comparison={'shell':shell,'phases':data,'shell_reference_gate':shell['reference_gate_pass'],
   'background_validity':all(p['all_checks_pass'] for p in data),
   'phase_compliance_relative':data[1]['compliance_per_N_mm']/data[0]['compliance_per_N_mm']-1,
   'phase_sensitivity_gt_5pct':abs(data[1]['compliance_per_N_mm']/data[0]['compliance_per_N_mm']-1)>.05,
   'background_relative_shell_compliance':{p['phase_name']:p['compliance_per_N_mm']/C-1 for p in data},
   'background_within_10pct':{p['phase_name']:bool(abs(p['compliance_per_N_mm']/C-1)<=.1) for p in data},
   'no_TPMS_peak_or_20pct_or_design_AD_conclusion':True}
 freeze=json.loads((D/'frozen_before.json').read_text());comparison['frozen_files_unchanged']={n:hashlib.sha256((R/n).read_bytes()).hexdigest()==h for n,h in freeze.items()}
 write(D/'comparison.json',comparison)
 fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
 labels=['Binary beam theory','Abaqus shell','Background layer','Background half-cell']
 C0=L/(E*Iphysical);v=[1.,C/C0]+[p['compliance_per_N_mm']/C0 for p in data]
 axes[0].bar(np.arange(4),v,color=['grey','C2','C0','C3']);axes[0].set_xticks(np.arange(4),labels,rotation=18,ha='right')
 axes[0].set_ylabel('Moment-rotation compliance / binary beam theory');axes[0].axhline(1,color='black',linewidth=.7)
 for i,x in enumerate(v):axes[0].text(i,x+.015,f'{x:.3f}',ha='center')
 groups=[['Dense sigmoid + void',p['continuous_section']['I_mm4']/Iphysical] for p in data]
 ix=np.arange(2);width=.24
 for offset,key,label,col in [(-width,'continuous_section','Dense same field','grey'),(0,'actual_Gauss_section_I_mm4','Current 27-point field','C1'),(width,'effective_EI_N_mm2','Solved effective EI','C0')]:
  val=[(p[key]['I_mm4'] if key=='continuous_section' else (p[key]/E if key=='effective_EI_N_mm2' else p[key]))/Iphysical for p in data]
  axes[1].bar(ix+offset,val,width,label=label,color=col)
 axes[1].set_xticks(ix,['Layer phase','Half-cell phase']);axes[1].set_ylabel('Section I or effective EI / binary reference');axes[1].axhline(1,color='black',linewidth=.7);axes[1].set_ylim(0,1.32);axes[0].set_ylim(0,1.40);axes[1].legend(fontsize=8)
 for ax in axes:ax.grid(axis='y',alpha=.25);ax.set_axisbelow(True)
 fig.suptitle('r27: same 0.5 mm plate, h=0.3125 mm; linear initial tangent only')
 fig.savefig(D/'bending_comparison.png',dpi=180);plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
 for p,c in zip(data,['C0','C3']):
  zc=p['midsurface_z_mm'];z=np.linspace(0,H,2001)
  phi=np.asarray(thickness_occupancy(abs(z-zc),T,ELL));ax.plot(z-zc,phi,label=p['phase_name'],color=c)
  x,w=np.polynomial.legendre.leggauss(3);q=(np.arange(8)[:,None]+(x+1)[None,:]/2)*(L/32)
  values=np.asarray(thickness_occupancy(abs(q-zc),T,ELL));ax.scatter((q-zc).ravel(),values.ravel(),color=c,s=23)
 ax.axvline(-T/2,color='grey',linestyle=':');ax.axvline(T/2,color='grey',linestyle=':')
 ax.set_xlim(-.6,.6);ax.set_xlabel('Distance from the actual midsurface (mm)');ax.set_ylabel('Reference occupancy');ax.legend();ax.grid(alpha=.2)
 ax.set_title('Same continuous field; different fixed Gauss sampling positions')
 fig.savefig(D/'occupancy_samples.png',dpi=180);plt.close(fig)
 print(json.dumps({k:v for k,v in comparison.items() if k not in ('phases','shell','frozen_files_unchanged')},indent=2))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['jax','export','analyze']);args=parser.parse_args();verify_freeze()
 if args.action=='jax':
  for name,zc in [('phase_layer',1.25),('phase_half',1.25+L/32/2)]:solve_phase(name,zc)
 elif args.action=='export':export_shell()
 else:analyze()

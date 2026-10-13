"""One frozen cylinder patch, shared initial tangent; no production modification."""
from pathlib import Path
import os, sys, json, hashlib, time, argparse
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
RROOT=Path('/home/xuehu/projects/tpms_jax'); D=Path(__file__).resolve().parent
sys.path.insert(0,str(RROOT))
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.special import expit
import jax, jax.numpy as jnp, basix
from jax_fem.generate_mesh import Mesh
from jax_fem.basis import get_elements
from hyperelastic_fem import DensityHyperelasticity
L=10.; N=32; B=L/N; RAD=2.5; T=.5; E=10.; NU=.3; ETA=1e-4
WIDTH=.05; ELL=WIDTH/(2*np.log(9)); LOAD=1e-4; RP=1000001

def write(name,obj):
    p=D/name; p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f: f.write(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def frozen():
    for name,h in json.loads((D/'frozen_before.json').read_text()).items():
        assert hashlib.sha256((RROOT/name).read_bytes()).hexdigest()==h,name

def analytic():
    a=RAD-T/2; b=RAD+T/2; mu=E/(2*(1+NU)); lam=E*NU/((1+NU)*(1-2*NU)); c=2*(lam+mu)
    pressure=LOAD/(2*np.pi*RAD*B)
    mat=np.array([[c,-2*mu/a**2,0,0],[0,0,c,-2*mu/b**2],
                  [RAD,1/RAD,-RAD,-1/RAD],[-c,2*mu/RAD**2,c,-2*mu/RAD**2]])
    coeff=np.linalg.solve(mat,[0,0,0,-pressure]); u=coeff[0]*RAD+coeff[1]/RAD
    shell=2*np.pi*B*E*T/((1-NU**2)*RAD)
    return {'solid_K_N_per_mm':float(LOAD/u),'membrane_shell_K_N_per_mm':shell,
            'solid_over_membrane_minus_one':float(LOAD/u/shell-1),'mid_u_mm':float(u),
            'inner_outer_coefficients':coeff.tolist(),'mid_pressure_MPa':pressure,
            'equation_residual':float(np.linalg.norm(mat@coeff-np.array([0,0,0,-pressure]))/pressure)}

def prepare():
    out=D/'abaqus'; out.mkdir(exist_ok=False)
    material=['*Material, name=INITIAL','*Elastic','10., 0.3']
    tail=['*Output, field','*Node Output','U, RF','*Element Output','S, E',
          '*Output, history','*Energy Output','ALLSE, ALLIE, ALLAE, ALLWK','*End Step']
    # CAX8 is an exact axisymmetric reduction of 3D elasticity, retaining hoop strain.
    nodes={}; elems=[]; counter=0
    def node(i,j):
        nonlocal counter
        if (i,j) not in nodes:counter+=1;nodes[i,j]=counter
        return nodes[i,j]
    order=[(0,0),(2,0),(2,2),(0,2),(1,0),(2,1),(1,2),(0,1)]
    for i in range(4): elems.append([node(2*i+a,b) for a,b in order])
    lines=['*Heading','r39 exact cylindrical plane-strain patch; concentrated mid-surface ring load','*Node']
    for (i,j),n in nodes.items():lines.append(f'{n}, {RAD-T/2+i*T/8:.16g}, {j*B/2:.16g}')
    lines.append(f'{RP}, {RAD:.16g}, 0.'); lines.append('*Element, type=CAX8, elset=SOLID')
    lines.extend(f'{i+1}, '+', '.join(map(str,e)) for i,e in enumerate(elems))
    lines+=['*Nset, nset=PHYSICAL']
    labels=list(nodes.values())
    lines += [', '.join(map(str,labels[i:i+16])) for i in range(0,len(labels),16)]
    lines+=material
    lines+=['*Solid Section, elset=SOLID, material=INITIAL',',']
    equations=[]
    for i in range(9):
        labels=sorted([(j,n) for (k,j),n in nodes.items() if k==i]); master=labels[0][1]
        for j,n in labels:
            if i==4: equations.append([(n,1,1.),(RP,1,-1.)])
            elif n!=master:equations.append([(n,1,1.),(master,1,-1.)])
    for terms in equations:
        lines+=['*Equation',str(len(terms)),', '.join(f'{n}, {d}, {c:.16g}' for n,d,c in terms)]
    lines+=['*Boundary','PHYSICAL, 2, 2, 0.',f'{RP}, 2, 2, 0.',
            '*Step, name=INITIAL, nlgeom=NO','*Static','1., 1.','*Cload',f'{RP}, 1, {LOAD:.16g}']+tail
    (out/'cylinder_solid.inp').write_text('\n'.join(lines)+'\n')
    # S3R cylinder: exact axisymmetric admissible mode imposed by one radial control.
    count=128; nid=lambda i,j:(i%count)*3+j+1
    lines=['*Heading','r39 S3R cylindrical plane-strain patch; same generalized mid-surface load','*Node']
    for i in range(count):
        th=2*np.pi*i/count
        for j in range(3):lines.append(f'{nid(i,j)}, {RAD*np.cos(th):.16g}, {RAD*np.sin(th):.16g}, {j*B/2:.16g}')
    lines.append(f'{RP}, 0., 0., 0.');lines.append('*Element, type=S3R, elset=SHELL'); eid=0
    for i in range(count):
        for j in range(2):
            for e in [(nid(i,j),nid(i+1,j),nid(i+1,j+1)),(nid(i,j),nid(i+1,j+1),nid(i,j+1))]:
                eid+=1;lines.append(f'{eid}, '+', '.join(map(str,e)))
    lines+=['*Nset, nset=PHYSICAL']
    labels=list(range(1,count*3+1))
    lines += [', '.join(map(str,labels[i:i+16])) for i in range(0,len(labels),16)]
    lines+=material
    lines+=['*Shell Section, elset=SHELL, material=INITIAL','0.5, 5']
    for i in range(count):
        th=2*np.pi*i/count
        for j in range(3):
            for d,c in [(1,np.cos(th)),(2,np.sin(th))]:
                if abs(c)<1e-14:lines+=['*Boundary',f'{nid(i,j)}, {d}, {d}, 0.']
                else:lines+=['*Equation','2',f'{nid(i,j)}, {d}, 1., {RP}, 1, {-c:.16g}']
    lines+=['*Boundary','PHYSICAL, 3, 6, 0.',f'{RP}, 2, 3, 0.',
            '*Step, name=INITIAL, nlgeom=NO','*Static','1., 1.','*Cload',f'{RP}, 1, {LOAD:.16g}']+tail
    (out/'cylinder_shell.inp').write_text('\n'.join(lines)+'\n')
    write('analytic.json',analytic())
    write('abaqus/input.json',{'R_mm':RAD,'t_mm':T,'B_mm':B,'E_MPa':E,'nu':NU,'generalized_force_N':LOAD,
                            'control_label':RP,'solid_radial_elements':4,'shell_circumferential':count,
                            'solid_reference':'CAX8 axisymmetric 3D reduction, NOT plane stress / full 3D C3D mesh',
                            'shell_rotations':'zero for exact axisymmetric plane-strain radial mode; no constitutive shear deleted'})

def background():
    frozen(); start=time.perf_counter()
    family,cell,_,_,degree,order=get_elements('HEX27'); element=basix.create_element(family,cell,degree)
    local=np.rint(2*element.points[order]).astype(int)
    points=np.indices((65,65,3)).reshape(3,-1).T/64
    orig=2*np.indices((N,N,1)).reshape(3,-1).T
    ix=orig[:,None,:]+local[None,:,:];cells=(ix[:,:,0]*65+ix[:,:,1])*3+ix[:,:,2]
    problem=DensityHyperelasticity(Mesh(points,cells),vec=3,dim=3,ele_type='HEX27',quadrature_order=4,dirichlet_bc_info=[[],[],[]])
    problem.material_model='objective_void';qp=np.asarray(problem.physical_quad_points)*L
    distance=abs(np.linalg.norm(qp[:,:,:2]-L/2,axis=-1)-RAD)
    rho=expit((T/2-distance)/ELL);problem.set_params(jnp.zeros((3,3)),rho,ETA)
    scale=np.asarray(problem.stiffness_scale);g=np.asarray(problem.shape_grads);w=np.asarray(problem.fe.JxW)
    tangent=jax.jit(jax.vmap(jax.jacfwd(problem.material_stress,argnums=0)))
    aa=np.asarray(tangent(jnp.broadcast_to(jnp.eye(3),(scale.size,3,3)),jnp.asarray(scale.ravel()))).reshape(*scale.shape,3,3,3,3)
    mu=E/(2*(1+NU));lam=E*NU/((1+NU)*(1-2*NU));eye=np.eye(3)
    C=lam*np.einsum('ij,kl->ijkl',eye,eye)+mu*(np.einsum('ik,jl->ijkl',eye,eye)+np.einsum('il,jk->ijkl',eye,eye))
    tangent_error=float(np.max(abs(aa-scale[:,:,None,None,None,None]*C)))
    # Exact extrusion reduction: each xy column has identical Ux/Uy and Uz=0.
    xy_local=local[:,0]*3+local[:,1];gg=np.zeros((len(cells),len(w[0]),9,3))
    for n in range(27):gg[:,:,xy_local[n]]+=g[:,:,n]
    gz=float(np.max(abs(gg[:,:,:,2])))
    kc=np.einsum('cqijkl,cqaj,cqbl,cq->caibk',aa[:,:,:2,:,:2,:],gg,gg,w,optimize=True).reshape(-1,18,18)*L
    origins=orig[:,:2]//2
    ids=np.array([[(2*i+a)*65+2*j+b for a in range(3) for b in range(3)] for i,j in origins])
    dofs=(2*ids[:,:,None]+np.arange(2)).reshape(-1,18)
    K=sp.coo_matrix((kc.ravel(),(np.repeat(dofs,18,axis=1).ravel(),np.tile(dofs,(1,18)).ravel())),shape=(65*65*2,)*2).tocsr()
    K.sum_duplicates();K.eliminate_zeros();xy=np.indices((65,65)).reshape(2,-1).T*L/64
    # Split the exact circle at background cell faces; integrate the same trace functional.
    cuts=[0.,2*np.pi]
    for s in np.arange(N+1)*L/N:
        v=(s-L/2)/RAD
        if abs(v)<=1:
            q=np.arccos(np.clip(v,-1,1));cuts.extend([q,2*np.pi-q,(np.pi/2-q)%(2*np.pi),(np.pi/2+q)%(2*np.pi)])
    cuts=np.unique(np.round(cuts,14)); cuts.sort()
    def load(level):
        z,wt=np.polynomial.legendre.leggauss(level);f=np.zeros((65*65,2))
        for a,b in zip(cuts[:-1],cuts[1:]):
            theta=(a+b)/2+(b-a)*z/2;normal=np.stack([np.cos(theta),np.sin(theta)],axis=1)
            pp=L/2+RAD*normal;cell=np.clip(np.floor(pp/(L/N)).astype(int),0,N-1);ref=pp/(L/N)-cell
            vals=[np.stack([2*(v-.5)*(v-1),4*v*(1-v),2*v*(v-.5)],axis=1) for v in ref.T]
            shape=(vals[0][:,:,None]*vals[1][:,None,:]).reshape(-1,9)
            nodes=np.array([[(2*i+k)*65+2*j+l for k in range(3) for l in range(3)] for i,j in cell])
            ff=shape[:,:,None]*normal[:,None,:]*(LOAD/(2*np.pi)*(b-a)*wt/2)[:,None,None]
            np.add.at(f,nodes.ravel(),ff.reshape(-1,2))
        return f.ravel()
    f8=load(8);f=load(12);trace_error=float(np.linalg.norm(f-f8)/np.linalg.norm(f))
    gauge=np.array([2*(32*65+32),2*(32*65+32)+1,2*(64*65+32)+1]);free=np.setdiff1d(np.arange(K.shape[0]),gauge)
    prep=time.perf_counter()-start;u=np.zeros(K.shape[0]);u[free]=spsolve(K[free][:,free].tocsc(),f[free])
    residual=K@u-f;rel=float(np.linalg.norm(residual[free])/np.linalg.norm(f[free]));U=float(.5*u@(K@u));delta=float(f@u/LOAD)
    full=np.zeros((len(points),3));full[:,:2]=u.reshape(-1,2)[np.arange(len(points))//3]
    grad=np.einsum('cni,cqnj->cqij',full[cells]/L,g,optimize=True);eps=.5*(grad+grad.swapaxes(-1,-2));J=np.linalg.det(eye+grad)
    affine=(xy-L/2)/RAD;virtual=float(f.reshape(-1,2).ravel()@affine.ravel()/LOAD)
    net=f.reshape(-1,2).sum(axis=0);torque=float(np.sum((xy[:,0]-L/2)*f.reshape(-1,2)[:,1]-(xy[:,1]-L/2)*f.reshape(-1,2)[:,0]))
    checks={'shared_tangent':tangent_error<1e-11,'extrusion_gradient_z':gz<1e-12,
            'trace_load_8_12':trace_error<=1e-10,'affine_virtual_work':abs(virtual-1)<=1e-10,
            'load_net_force':float(np.linalg.norm(net))/LOAD<=1e-10,'load_net_torque':abs(torque)/(LOAD*RAD)<=1e-10,
            'free_residual':rel<=1e-8,'gauge_reaction':float(np.linalg.norm(residual[gauge]))/LOAD<=1e-8,
            'energy_work':abs(U/(.5*LOAD*delta)-1)<=1e-8,'finite_small_strain':bool(np.isfinite(u).all() and J.min()>.98 and np.max(abs(eps))<.02),
            'matrix_symmetry':float(sp.linalg.norm(K-K.T)/sp.linalg.norm(K))<1e-12,'budget':time.perf_counter()-start<=900}
    result={'status':'ok' if all(checks.values()) else 'not_accepted','checks':checks,'K_N_per_mm':LOAD/delta,
            'generalized_force_N':LOAD,'mid_trace_mean_radial_u_mm':delta,'energy_N_mm':U,'true_free_residual':rel,
            'gauge_reaction_over_load':float(np.linalg.norm(residual[gauge]))/LOAD,'shared_tangent_error':tangent_error,
            'trace_8_12_relative':trace_error,'virtual_radial_affine_work_over_load':virtual,'minimum_sampled_J':float(J.min()),
            'max_abs_strain':float(np.max(abs(eps))),'weighted_volume_mm3':float(np.sum(scale*w)*L**3),
            'binary_volume_mm3':2*np.pi*RAD*T*B,'cells':len(cells),'physical_nodes':len(points),'reduced_xy_dofs':K.shape[0],
            'quad_per_cell':len(w[0]),'material_rule_unchanged':True,'actual_hex27_extrusion_reduction':True,
            'preparation_seconds':prep,'wall_seconds':time.perf_counter()-start,'production_changed':False,
            'root_cause_TPMS_certified':False,'load_is_internal_mid_surface_stress_jump_not_TPMS_macro_compression':True}
    np.savez_compressed(D/'background_state.npz',xy_mm=xy,u_xy_mm=u.reshape(-1,2),rho=rho)
    write('background_result.json',result);frozen();print(json.dumps(result,indent=2),flush=True)
    if result['status']!='ok':raise RuntimeError('Frozen checks failed; no retry')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','background']);a=ap.parse_args()
    if a.action=='prepare':prepare()
    else:background()

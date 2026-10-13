"""Same existing midsurface, same membrane energy rule, two saved displacement traces."""
from pathlib import Path
import sys,json,hashlib
R=Path('/home/xuehu/projects/tpms_jax');D=Path(__file__).resolve().parent;sys.path.insert(0,str(R))
import numpy as np,basix
from jax_fem.basis import get_elements
cache=np.load(R/'validation/geometry_transfer_20261006_r15/gauss_field.npz')
v=cache['surface_vertices']*10;tri=cache['surface_triangles'];node_labels=cache['node_ids']
sh=json.loads((R/'validation/initial_tangent_20261008_r30/abaqus/shell_result.json').read_text())
u_by_id={row['label']:row['U_mm'] for row in sh['nodes']};us=np.array([u_by_id[int(n)] for n in node_labels])
st=np.load(R/'validation/initial_tangent_20261008_r30/linear_state.npz');q=st['q'];ci=st['class_ids'];H=st['H']
family,cell,_,_,degree,order=get_elements('HEX27');el=basix.create_element(family,cell,degree)
local=np.rint(2*el.points[order]).astype(int);o=2*np.indices((32,)*3).reshape(3,-1).T
idx=o[:,None,:]+local[None,:,:];cells=(idx[:,:,0]*65+idx[:,:,1])*65+idx[:,:,2]
x=v/10;c=np.minimum(np.floor(x*32).astype(int),31);ref=x*32-c;cind=(c[:,0]*32+c[:,1])*32+c[:,2]
shape=el.tabulate(0,ref)[0,:,:,0][:,order];ub=(np.einsum('vn,vni->vi',shape,q[ci[cells[cind]]])+x@H.T)*10
edges=np.stack([v[tri[:,1]]-v[tri[:,0]],v[tri[:,2]]-v[tri[:,0]]],axis=-1)
inv=np.linalg.inv(np.einsum('cik,cil->ckl',edges,edges));pseudo=np.einsum('ckl,cil->cki',inv,edges)
cross=np.cross(edges[:,:,0],edges[:,:,1]);norm=np.linalg.norm(cross,axis=1);n=cross/norm[:,None];area=.5*norm
P=np.eye(3)-n[:,:,None]*n[:,None,:];mu=10/(2*1.3);lps=10*.3/(1-.3**2)
def energy(u):
    du=np.stack([u[tri[:,1]]-u[tri[:,0]],u[tri[:,2]]-u[tri[:,0]]],axis=-1)
    gu=np.einsum('cik,ckj->cij',du,pseudo);ep=.5*(gu+gu.swapaxes(-1,-2))
    ep=np.einsum('cij,cjk,ckl->cil',P,ep,P)
    return (.5*lps*np.trace(ep,axis1=-2,axis2=-1)**2+mu*np.sum(ep*ep,axis=(1,2)))*area*.5
es=energy(us);eb=energy(ub)
prior=json.loads((D/'analysis.json').read_text());mem=prior['shell']['membrane_N_mm'];validation=abs(es.sum()/mem-1)
threshold=.01
result={'status':'ok' if validation<=threshold else 'diagnostic_mapping_not_validated',
 'shell_trace_membrane_N_mm':float(es.sum()),'original_ODB_membrane_N_mm':mem,
 'shell_trace_energy_relative_difference':float(es.sum()/mem-1),'shell_trace_agreement_limit':threshold,
 'background_trace_membrane_N_mm':float(eb.sum()),
 'background_over_shell_trace_energy_minus_one':float(eb.sum()/es.sum()-1),
 'definition':'Piecewise-linear triangle trace of the saved physical displacement on the exact same midsurface; homogeneous plane-stress membrane energy with true t=0.5mm. No equilibrium is recomputed, no background constitutive law is replaced.',
 'interpretation':'Only compare trace energies as corresponding shell membrane quantities when shell own displacement energy is within the fixed 1% ODB reconstruction gate. Difference is a diagnostic of saved trace and approximation, not a causal stiffness correction or background membrane/bending split.',
 'causal_factor_identified':False}
(D/'membrane_trace.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
np.savez_compressed(D/'membrane_trace.npz',shell_membrane=es,background_membrane=eb)
for name,h in json.loads((D/'frozen_before.json').read_text()).items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h,name
print(json.dumps(result,indent=2))

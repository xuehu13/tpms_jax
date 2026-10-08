"""Read-only reproduction of extrusion roundoff in the existing r39 state."""
from pathlib import Path
import os,sys,json
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
R=Path('/home/xuehu/projects/tpms_jax');sys.path.insert(0,str(R))
import numpy as np,basix
from jax_fem.basis import get_elements
from jax_fem.generate_mesh import Mesh
from hyperelastic_fem import DensityHyperelasticity
D=Path(__file__).resolve().parent
f,cell,_,_,degree,order=get_elements('HEX27')
local=np.rint(2*basix.create_element(f,cell,degree).points[order]).astype(int)
points=np.indices((65,65,3)).reshape(3,-1).T/64
orig=2*np.indices((32,32,1)).reshape(3,-1).T
ix=orig[:,None,:]+local;cells=(ix[:,:,0]*65+ix[:,:,1])*3+ix[:,:,2]
p=DensityHyperelasticity(Mesh(points,cells),vec=3,dim=3,ele_type='HEX27',quadrature_order=4,dirichlet_bc_info=[[],[],[]])
g=np.asarray(p.shape_grads);gg=np.zeros((1024,27,9,3))
for n in range(27):gg[:,:,local[n,0]*3+local[n,1]]+=g[:,:,n]
s=np.load(D/'background_state.npz');u=np.zeros((len(points),3))
u[:,:2]=s['u_xy_mm'][np.arange(len(points))//3]
grad=np.einsum('cni,cqnj->cqij',u[cells]/10,g,optimize=True)
print(json.dumps({'basis_sum_gz_abs':float(np.max(abs(gg[:,:,:,2]))),
                  'basis_sum_gz_over_max_g':float(np.max(abs(gg[:,:,:,2]))/np.max(abs(gg))),
                  'actual_saved_displacement_grad_z_max':float(np.max(abs(grad[:,:,:,2]))),
                  'no_new_solve':True,'no_original_gate_relaxed':True},indent=2))

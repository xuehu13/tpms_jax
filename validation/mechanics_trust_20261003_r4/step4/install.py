"""Minimal extension of step3 material; prior solvers remain unchanged."""
from pathlib import Path
import json,hashlib
P=Path('/home/xuehu/projects/tpms_jax');O=P/'validation/mechanics_trust_20261003_r4/step4';H=Path(__file__).resolve().parent
name='hyperelastic_fem.py';source=(P/name).read_text()
(O/'source_before').mkdir();(O/'source_before'/name).write_text(source)
key='    energy=jax.vmap(neo_hookean_energy)(flat).reshape(F.shape[:2])\n'
assert key in source
source=source.replace(key,key+"    scale=getattr(problem,'stiffness_scale',jnp.ones(F.shape[:2]))\n    P=P*scale[...,None,None];energy=energy*scale\n")
source+='''


class DensityHyperelasticity(PeriodicHyperelasticity):
    """Reference Gauss occupancy scales the matched energy, P and tangent."""

    def set_params(self,H_macro,rho,eta=1e-4):
        rho=jnp.broadcast_to(jnp.asarray(rho),(self.fe.num_cells,self.fe.num_quads))
        if not 0<eta<=1 or not np.isfinite(np.asarray(rho)).all() or np.any((np.asarray(rho)<0)|(np.asarray(rho)>1)):
            raise ValueError('Expected 0<=rho<=1, 0<eta<=1')
        self.H_macro=jnp.asarray(H_macro);self.rho=rho;self.eta=eta
        self.stiffness_scale=eta+(1-eta)*rho
        self.internal_vars=[jnp.broadcast_to(self.H_macro,(*rho.shape,3,3)),self.stiffness_scale]

    def get_tensor_map(self):
        def stress(w_grad,H,scale):
            return scale*first_piola(jnp.eye(3)+H+w_grad)
        return stress

    def get_mass_map(self):
        def mass(u,x,H,scale):return jnp.zeros(self.vec)
        return mass

    def detF_stats(self,w):
        J=np.asarray(jnp.linalg.det(jnp.eye(3)+self.H_macro+self.fe.sol_to_grad(w)))
        rho=np.asarray(self.rho)
        result={'all_min':float(np.nanmin(J)),'all_max':float(np.nanmax(J)),'finite':bool(np.isfinite(J).all())}
        for tag,mask in [('solid',rho>=.95),('void',rho<=.05),('transition',(rho>.05)&(rho<.95))]:
            values=J[mask]
            result[tag]={'points':int(mask.sum()),'min':float(values.min()) if values.size else None,
                         'p01':float(np.quantile(values,.01)) if values.size else None,
                         'max':float(values.max()) if values.size else None}
        return result

    def newton_update(self,sol):
        # Guard validity before AD assembly; reject the trial, never alter J.
        self.trial_detF=self.detF_stats(sol[0]);self.trial_calls=getattr(self,'trial_calls',0)+1
        if not self.trial_detF['finite'] or self.trial_detF['all_min']<=0:
            self.invalid_trial_w=np.asarray(sol[0])
            raise ValueError('Nonpositive/nonfinite detF in Newton trial; no clipping')
        if self.trial_calls>16:raise RuntimeError('Locked maximum of 15 Newton corrections exceeded')
        return super().newton_update(sol)


def make_density_hyperelastic_problem(n,c=.541062,beta=40.,eta=1e-4):
    from geometry import density
    meshio_mesh=box_mesh(n,n,n,1.,1.,1.)
    mesh=Mesh(meshio_mesh.points,meshio_mesh.cells_dict[get_meshio_cell_type(ELE_TYPE)])
    problem=DensityHyperelasticity(mesh,vec=3,dim=3,ele_type=ELE_TYPE,dirichlet_bc_info=[[],[],[]])
    problem.set_params(jnp.zeros((3,3)),density(problem.physical_quad_points,c,beta),eta)
    fixed=xy_compression_fixed_dofs(problem.fe.points,n,n,n)
    problem.P_mat,problem.class_ids,problem.fixed_class_id=periodic_p_mat(
        problem.fe.points,n,n,n,periodic_axes=(0,1),fixed_class=None,fixed_dofs=fixed)
    return problem
'''
(P/name).write_text(source)
new=['scripts/finite_strain_gyroid.py','tests/test_density_hyperelastic.py']
for n in new:
    assert not (P/n).exists();(P/n).write_bytes((H/'newfiles'/n).read_bytes())
files=[name]+new
for n in files:
    f=O/'source_at_run'/n;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes((P/n).read_bytes())
(O/'source_at_run.json').write_text(json.dumps({n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in files},indent=2)+'\n')
(O/'source_changes.json').write_text(json.dumps({'modified':[name],'added':new,'prior_solvers_modified':False},indent=2)+'\n')
(O/'install.py').write_bytes(Path(__file__).read_bytes())
print('Minimal reference-Gauss energy extension installed; old solvers unchanged')

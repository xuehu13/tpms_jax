"""Matched compressible Neo-Hookean material on existing periodic HEX8 kinematics.

Energy and weak form use the reference configuration. No contact, plasticity,
void stabilization, or new equilibrium solver is implemented here.
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax_fem.generate_mesh import box_mesh, get_meshio_cell_type, Mesh
from fem import E, NU, ELE_TYPE, internal_force
from pbc import PeriodicLinearElasticityCube, periodic_p_mat, xy_compression_fixed_dofs

jax.config.update("jax_enable_x64", True)
MU = E/(2*(1+NU))
KAPPA = E/(3*(1-2*NU))


def neo_hookean_energy(F, mu=MU, kappa=KAPPA):
    """W per reference volume; J<=0 remains invalid rather than clipped."""
    J = jnp.linalg.det(F)
    return mu/2*(J**(-2/3)*jnp.sum(F*F)-3)+kappa/2*(J-1)**2


first_piola = jax.grad(neo_hookean_energy)


class PeriodicHyperelasticity(PeriodicLinearElasticityCube):
    """Reuse H runtime field, zero body force, mesh and constraint interfaces."""

    def get_tensor_map(self):
        def stress(w_grad, H):
            return first_piola(jnp.eye(3)+H+w_grad)
        return stress


def _make_periodic_problem(n, problem_type):
    """One mesh/constraint definition for full and density-weighted solids."""
    meshio_mesh = box_mesh(n,n,n,1.,1.,1.)
    mesh = Mesh(meshio_mesh.points,meshio_mesh.cells_dict[get_meshio_cell_type(ELE_TYPE)])
    problem = problem_type(mesh,vec=3,dim=3,ele_type=ELE_TYPE,
                           dirichlet_bc_info=[[],[],[]])
    fixed = xy_compression_fixed_dofs(problem.fe.points,n,n,n)
    problem.P_mat,problem.class_ids,problem.fixed_class_id = periodic_p_mat(
        problem.fe.points,n,n,n,periodic_axes=(0,1),fixed_class=None,fixed_dofs=fixed)
    return problem


def make_hyperelastic_problem(n):
    """Full solid; XY-periodic fluctuation with the existing compression pins."""
    problem = _make_periodic_problem(n, PeriodicHyperelasticity)
    problem.set_params(jnp.zeros((3,3)))
    return problem


def reduced_guess(problem, w):
    """Installed JAX-FEM expects a reduced vector when P_mat is present."""
    projection=problem.P_mat
    counts=np.asarray(projection.power(2).sum(axis=0)).ravel()
    return jnp.asarray((projection.T@np.asarray(w).ravel())/counts)


def finite_response(problem, sol):
    """Recover reactions and reference/current-configuration physical outputs."""
    F=jnp.eye(3)+problem.H_macro+problem.fe.sol_to_grad(sol[0])
    flat=F.reshape((-1,3,3))
    P=jax.vmap(first_piola)(flat).reshape(F.shape)
    energy=jax.vmap(neo_hookean_energy)(flat).reshape(F.shape[:2])
    scale=getattr(problem,'stiffness_scale',jnp.ones(F.shape[:2]))
    P=P*scale[...,None,None];energy=energy*scale
    J=jnp.linalg.det(F)
    weights=jnp.asarray(problem.JxW)[:,0,:]
    if not np.isfinite(np.asarray(F)).all() or float(J.min())<=0:
        raise ValueError("Nonfinite or nonpositive detF; result is not physical")
    sigma=P@jnp.swapaxes(F,-1,-2)/J[...,None,None]
    V0=weights.sum();V=(J*weights).sum()
    meanP=(P*weights[...,None,None]).sum(axis=(0,1))/V0
    meanSigma=(sigma*(J*weights)[...,None,None]).sum(axis=(0,1))/V
    residual=internal_force(problem,sol)
    reduced=np.asarray(problem.P_mat.T@residual.ravel())
    points=np.asarray(problem.fe.points)
    top=np.isclose(points[:,2],1.,atol=1e-12);bottom=np.isclose(points[:,2],0.,atol=1e-12)
    return {'energy':float((energy*weights).sum()),'reference_volume':float(V0),
            'current_volume':float(V),'J_min':float(J.min()),'J_max':float(J.max()),
            'mean_first_piola':np.asarray(meanP).tolist(),'mean_cauchy':np.asarray(meanSigma).tolist(),
            'Fz_top':float(residual[top,2].sum()),'Fz_bottom':float(residual[bottom,2].sum()),
            'reduced_residual_l2':float(np.linalg.norm(reduced))}



class DensityHyperelasticity(PeriodicHyperelasticity):
    """Reference Gauss occupancy scales the matched energy, P and tangent."""

    def set_params(self,H_macro,rho,eta=1e-4):
        """Validate concrete inputs before forward solves or design tracing."""
        rho_host=np.asarray(rho)
        H_host=np.asarray(H_macro)
        if (not np.isfinite(eta) or not 0<eta<=1
                or not np.isfinite(rho_host).all()
                or np.any((rho_host<0)|(rho_host>1))):
            raise ValueError('Expected 0<=rho<=1, 0<eta<=1')
        if H_host.shape!=(3,3) or not np.isfinite(H_host).all():
            raise ValueError('Expected finite H_macro with shape (3,3)')
        self._set_params_jax(H_macro,rho,eta)

    def _set_params_jax(self,H_macro,rho,eta=1e-4):
        """Traceable field update for prevalidated inputs, not an adjoint solver.

        An AD caller must validate its concrete inputs outside tracing and
        restore concrete parameters afterwards. Equilibrium-gradient checks
        and admissibility of design perturbations remain separate tasks.
        """
        rho=jnp.broadcast_to(jnp.asarray(rho),(self.fe.num_cells,self.fe.num_quads))
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
    """Historical Gyroid field on the shared periodic HEX8 background.

    c is an implicit-field band parameter, not constant physical shell thickness.
    """
    from geometry import density
    problem=_make_periodic_problem(n,DensityHyperelasticity)
    problem.set_params(jnp.zeros((3,3)),density(problem.physical_quad_points,c,beta),eta)
    return problem

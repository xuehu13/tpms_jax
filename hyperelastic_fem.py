"""Matched compressible Neo-Hookean material on periodic HEX8/HEX27 kinematics.

Energy and weak form use the reference configuration. An opt-in objective
void extension shares this material kernel; no contact, plasticity or new solver.
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



def stable_neo_hookean_extension(F, mu=MU, kappa=KAPPA):
    """Objective all-F extension, Smith et al. (2018), Eq. 14 / Sec. 3.4.

    Parameters are converted to preserve the original small-strain mu/kappa;
    the constant is subtracted to set W(I)=0. This is an artificial void
    extension, not the matched true-solid NH law and not a contact barrier.
    No SVD, inverse F or determinant clipping is used.
    """
    I1 = jnp.sum(F*F)
    # Polynomial determinant avoids inverse-based higher AD at singular F.
    J = jnp.dot(F[:,0],jnp.cross(F[:,1],F[:,2]))
    mu_s = 4*mu/3
    lambda_s = kappa+mu/6
    return (mu_s/2*(I1-3-jnp.log((I1+1)/4))
            -mu*(J-1)+lambda_s/2*(J-1)**2)


def void_nh_weight(rho):
    """C2 gate: zero through .001 occupancy, original NH from .01 upward.

    Fixed numerical convention for this single candidate, not a fit to shell
    forces. Occupancy and the geometric interface width remain unchanged.
    """
    x = jnp.clip((rho-.001)/(.01-.001),0.,1.)
    return x**3*(10-15*x+6*x**2)


def objective_void_energy(F, rho, eta=1e-4, mu=MU, kappa=KAPPA):
    """Opt-in objective virtual energy; default material remains matched NH.

    Deep void has an all-F energy. Wherever NH has nonzero weight, actual
    detF must stay positive. F_nh=I skips an *exactly inactive* branch; it
    does not repair or reinterpret actual F, J, or the true wall geometry.
    Scale and gate both participate in occupancy derivatives.
    """
    weight = void_nh_weight(rho)
    F_nh = jnp.where(weight>0,F,jnp.eye(3,dtype=F.dtype))
    return (eta+(1-eta)*rho)*(
        weight*neo_hookean_energy(F_nh,mu,kappa)
        +(1-weight)*stable_neo_hookean_extension(F,mu,kappa))


objective_void_first_piola = jax.grad(objective_void_energy)


class PeriodicHyperelasticity(PeriodicLinearElasticityCube):
    """Reuse H runtime field, zero body force, mesh and constraint interfaces."""

    def get_tensor_map(self):
        def stress(w_grad, H):
            return first_piola(jnp.eye(3)+H+w_grad)
        return stress


def _make_periodic_problem(n, problem_type, periodic_axes=(0,1),
                           element_degree=1, quadrature_order=None):
    """One mesh/constraint definition for full and density-weighted solids."""
    if element_degree == 1:
        meshio_mesh = box_mesh(n,n,n,1.,1.,1.)
        mesh = Mesh(meshio_mesh.points,meshio_mesh.cells_dict[get_meshio_cell_type(ELE_TYPE)])
        ele_type = ELE_TYPE
    elif element_degree == 2:
        # Reuse installed Basix/JAX-FEM basis and its mesh node convention.
        import basix
        from jax_fem.basis import get_elements
        family, cell, _, _, degree, order = get_elements('HEX27')
        local = np.rint(2*basix.create_element(family,cell,degree).points[order]).astype(int)
        levels = 2*n+1
        points = np.indices((levels,)*3).reshape(3,-1).T/(2*n)
        origins = 2*np.indices((n,)*3).reshape(3,-1).T
        indices = origins[:,None,:]+local[None,:,:]
        cells = (indices[...,0]*levels+indices[...,1])*levels+indices[...,2]
        mesh = Mesh(points,cells); ele_type = 'HEX27'
        # 3x3x3 volume points, rather than the library's expensive default.
        if quadrature_order is None: quadrature_order = 4
    else:
        raise ValueError('Supported element degrees are 1 (HEX8) and 2 (HEX27)')
    problem = problem_type(mesh,vec=3,dim=3,ele_type=ele_type,
                           quadrature_order=quadrature_order,
                           dirichlet_bc_info=[[],[],[]])
    periodic_axes=tuple(periodic_axes)
    if periodic_axes not in ((0,1),(0,1,2)):
        raise ValueError('Supported periodic axes are XY or XYZ')
    nodes_per_axis=n*element_degree
    fixed=xy_compression_fixed_dofs(problem.fe.points,nodes_per_axis,nodes_per_axis,nodes_per_axis) if periodic_axes==(0,1) else ()
    gauge=None if periodic_axes==(0,1) else (0,0,0)
    problem.P_mat,problem.class_ids,problem.fixed_class_id = periodic_p_mat(
        problem.fe.points,nodes_per_axis,nodes_per_axis,nodes_per_axis,
        periodic_axes=periodic_axes,fixed_class=gauge,fixed_dofs=fixed)
    problem.element_degree=element_degree
    return problem


def make_hyperelastic_problem(n, periodic_axes=(0,1)):
    """Full solid; historical XY pins or XYZ with only translation gauge."""
    problem = _make_periodic_problem(n, PeriodicHyperelasticity, periodic_axes)
    problem.set_params(jnp.zeros((3,3)))
    return problem


def reduced_guess(problem, w):
    """Installed JAX-FEM expects a reduced vector when P_mat is present."""
    projection=problem.P_mat
    counts=np.asarray(projection.power(2).sum(axis=0)).ravel()
    return jnp.asarray((projection.T@np.asarray(w).ravel())/counts)


def finite_response(problem, sol):
    """Recover reactions and reference/current-configuration physical outputs."""
    if getattr(problem,'material_model','nh')!='nh':
        raise ValueError('Objective void is currently Explicit-only; use its reference energy/P observables')
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

    def material_energy(self,F,scale):
        if getattr(self,'material_model','nh')=='objective_void':
            return objective_void_energy(F,(scale-self.eta)/(1-self.eta),self.eta)
        return scale*neo_hookean_energy(F)

    def material_stress(self,F,scale):
        if getattr(self,'material_model','nh')=='objective_void':
            return objective_void_first_piola(F,(scale-self.eta)/(1-self.eta),self.eta)
        return scale*first_piola(F)

    def nh_active(self,scale):
        if getattr(self,'material_model','nh')=='objective_void':
            return void_nh_weight((scale-self.eta)/(1-self.eta))>0
        return jnp.ones_like(scale,dtype=bool)

    def get_tensor_map(self):
        def stress(w_grad,H,scale):
            return self.material_stress(jnp.eye(3)+H+w_grad,scale)
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
        # PETSc copied the previous COO values into its matrix. The next
        # assembly replaces V entirely; retaining N64's ~1.13 GiB old V
        # during that assembly serves no purpose. No tangent is approximated.
        if hasattr(self,'V'):del self.V
        return super().newton_update(sol)


def make_density_hyperelastic_problem(n,c=.541062,beta=40.,eta=1e-4,
                                      rho_quad=None,periodic_axes=(0,1),
                                      element_degree=1,quadrature_order=None,material_model="nh"):
    """Shared background with optional actual-Gauss input and XY/XYZ constraints.

    Without rho_quad, preserve the historical Gyroid c/beta field. Otherwise
    evaluate the callable on this unique Problem or accept its Gauss array;
    c/beta are unused. Geometry is fixed in reference/material coordinates.
    """
    if material_model not in ('nh','objective_void'):
        raise ValueError('Expected nh or objective_void material model')
    if material_model=='objective_void' and not 0<eta<1:
        raise ValueError('Objective void requires a fixed 0<eta<1')
    from geometry import density
    problem=_make_periodic_problem(n,DensityHyperelasticity,periodic_axes,
                                   element_degree,quadrature_order)
    # Fixed before the first JIT trace; never change model on an existing Problem.
    problem.material_model=material_model
    if rho_quad is None:rho=density(problem.physical_quad_points,c,beta)
    else:rho=rho_quad(problem) if callable(rho_quad) else rho_quad
    problem.set_params(jnp.zeros((3,3)),rho,eta)
    return problem

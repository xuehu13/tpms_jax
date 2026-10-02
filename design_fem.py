"""Fixed-lateral Gyroid design derivatives using installed JAX-FEM's adjoint.

theta = (c0, ax, ay, az) controls a periodic positive level-set width.
This adapter owns one ordinary density problem. Its instance set_params is
bound to the pure-JAX design setter required by ad_wrapper; the existing
material law, mesh, P_mat and installed library remain unchanged.
Only reverse-mode first derivatives on this fixed mesh are supported here.
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax_fem.solver import ad_wrapper
from density_fem import DensityLinearElasticityPeriodic, make_density_problem
from fem import NU
from geometry import density
from pbc import xy_compression_fixed_dofs

jax.config.update('jax_enable_x64', True)
SOLVER_OPTIONS = {'spsolve_solver': {}, 'tol': 1e-12, 'rel_tol': 1e-10}
ADJOINT_OPTIONS = {'spsolve_solver': {}}
OUTPUT_NAMES = ('Eapp', 'Vf', 'Qw')


def width_basis(xyz):
    """Unit-period basis (1, cos X, cos Y, cos Z), also periodic on all faces."""
    xyz = jnp.asarray(xyz)
    return jnp.concatenate((jnp.ones(xyz.shape[:-1]+(1,)), jnp.cos(2*jnp.pi*xyz)), axis=-1)


class GyroidDesign:
    def __init__(self, N, beta=20., emin_ratio=.001, eps_z=-.01):
        if N < 2 or not np.all(np.isfinite([beta, emin_ratio, eps_z])) or beta <= 0 or not 0 < emin_ratio < 1 or eps_z == 0:
            raise ValueError('Expected N>=2, beta>0, 0<Emin/Es<1 and nonzero finite strain')
        self.N, self.beta, self.eps_z = N, beta, eps_z
        self.E_s, self.E_min = 10., 10.*emin_ratio
        self.H = jnp.diag(jnp.array([0., 0., eps_z]))
        self.problem = make_density_problem(
            N, N, N, self.H, None, E_s=self.E_s, E_min=self.E_min,
            periodic_axes=(0, 1), fixed_class=None,
            fixed_dofs=lambda pts: xy_compression_fixed_dofs(pts, N, N, N))
        self.points = self.problem.physical_quad_points
        self.basis = width_basis(self.points)
        self.weights = jnp.asarray(self.problem.JxW)[:, 0, :]
        self.volume = self.weights.sum()
        # ad_wrapper expects set_params(theta), not set_params(H, rho, ...).
        self.problem.set_params = self._set_design
        self._predict = ad_wrapper(self.problem, dict(SOLVER_OPTIONS), dict(ADJOINT_OPTIONS))

    @staticmethod
    def validate_theta(theta):
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (4,) or not np.all(np.isfinite(theta)) or theta[0] <= np.abs(theta[1:]).sum():
            raise ValueError('Expected four finite parameters and positive width everywhere')
        return jnp.asarray(theta)

    def density_field(self, theta):
        return density(self.points, self.basis @ theta, self.beta)

    def _set_design(self, theta):
        DensityLinearElasticityPeriodic.set_params(
            self.problem, self.H, self.density_field(theta), self.E_s, self.E_min)

    def _fields(self, theta, w):
        grad = self.problem.fe.sol_to_grad(w)
        eps = .5*(grad+jnp.swapaxes(grad, -1, -2)) + self.H
        rho = self.density_field(theta)
        E = self.E_min + rho*(self.E_s-self.E_min)
        lam = E*NU/((1+NU)*(1-2*NU))
        mu = E/(2*(1+NU))
        sigma = lam[..., None, None]*jnp.trace(eps, axis1=-2, axis2=-1)[..., None, None]*jnp.eye(3) + 2*mu[..., None, None]*eps
        U = .5*jnp.sum(sigma*eps*self.weights[..., None, None])
        return rho, sigma, U

    def responses(self, theta, w):
        rho, _, U = self._fields(theta, w)
        return jnp.array([2*U/(self.volume*self.eps_z**2),
                          jnp.sum(rho*self.weights)/self.volume,
                          jnp.sum(w*w)/(w.shape[0]*self.eps_z**2)])

    def _outputs(self, theta):
        w = self._predict(theta)[0]
        return self.responses(theta, w), w

    def observables(self, theta):
        """Differentiable Eapp, Vf, Qw; call outside jit/vmap with validated theta."""
        return self._outputs(theta)[0]

    def derivatives(self, theta):
        theta = self.validate_theta(theta)
        try:
            (values, w), pullback = jax.vjp(self._outputs, theta)
            # Sequential pullbacks: the installed sparse solver is not vmappable.
            jac = jnp.stack([pullback((jnp.eye(3)[i], jnp.zeros_like(w)))[0] for i in range(3)])
            envelope = jax.grad(lambda t: self.responses(t, jax.lax.stop_gradient(w))[0])(theta)
            return {'values': np.asarray(values), 'jacobian': np.asarray(jac),
                    'energy_envelope_gradient': np.asarray(envelope)}
        finally:
            # implicit_vjp traces set_params; restore concrete state for diagnostics.
            self._set_design(theta)

    def forward(self, theta):
        theta = self.validate_theta(theta)
        values, w = self._outputs(theta)
        rho, sigma, U = self._fields(theta, w)
        avg = jnp.sum(sigma*self.weights[..., None, None], axis=(0, 1))/self.volume
        residual = np.asarray(self.problem.compute_residual([w])[0])
        pts = np.asarray(self.problem.fe.points)
        top, bottom = (np.isclose(pts[:, 2], z, atol=1e-8) for z in (1., 0.))
        ftop, fbottom = float(residual[top, 2].sum()), float(residual[bottom, 2].sum())
        row = {'theta': np.asarray(theta).tolist(), 'values': np.asarray(values).tolist(),
               'Fz_top': ftop, 'Fz_bottom': fbottom, 'U_internal': float(U),
               'red_res': float(np.abs(self.problem.P_mat.T @ residual.ravel()).max()),
               'balance': ftop+fbottom, 'reaction_consistency_err': abs(ftop-float(avg[2, 2])),
               'work_identity_err': abs(float(U)-.5*float(self.volume)*float(jnp.sum(avg*self.H)))}
        finite = all(np.isfinite(np.asarray(v)).all() for v in row.values())
        row['checks'] = {'finite': bool(finite), 'red_res<=1e-8': row['red_res'] <= 1e-8,
                         'balance<=1e-8': abs(row['balance']) <= 1e-8,
                         'reaction<=1e-6': row['reaction_consistency_err'] <= 1e-6,
                         'work_identity<=1e-8': row['work_identity_err'] <= 1e-8}
        row['status'] = 'ok' if all(row['checks'].values()) else 'check_failed'
        return row

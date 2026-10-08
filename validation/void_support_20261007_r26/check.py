"""Meaningful local consistency checks, not design-gradient certification."""
import importlib.util,json
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('r26run',Path(__file__).with_name('run.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
jax=m.jax;jnp=m.jnp
from hyperelastic_fem import make_density_hyperelastic_problem,objective_void_energy,objective_void_first_piola,void_nh_weight
from types import SimpleNamespace
checks=[]
def record(name,ok,**kw):
 checks.append({'name':name,'pass':bool(ok),**kw})
 if not ok:raise AssertionError(name)
F=jnp.array([[.85,.17,.03],[.02,1.07,.04],[.03,.01,.76]])
for rho in [0.,.0005,.001,.004,.0099,.01,.4,1.]:
 scale=1e-4+(1-1e-4)*rho
 p=SimpleNamespace(material_model='objective_void',eta=1e-4,
   material_energy=lambda X,s:objective_void_energy(X,(s-1e-4)/(1-1e-4),1e-4))
 m.apply_support(p,.1);s=.1+.9*float(void_nh_weight(rho))
 W=float(p.material_energy(F,scale));P=p.material_stress(F,scale);A=jax.jacfwd(p.material_stress)(F,scale)
 base=lambda X:objective_void_energy(X,rho,1e-4)
 errors=[abs(W-s*float(base(F))),float(jnp.max(jnp.abs(P-s*jax.grad(base)(F)))),
         float(jnp.max(jnp.abs(A-s*jax.hessian(base)(F))))]
 record('energy_P_tangent_scale_phi_'+str(rho),max(errors)<2e-12,absolute_errors=errors)
# Folds in the artificial domain stay finite and frame indifferent.
fold=F.at[2,2].set(-.3);Q=jnp.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
for rho in [.0005,.004]:
 p=SimpleNamespace(material_model='objective_void',eta=1e-4,
   material_energy=lambda X,s:objective_void_energy(X,(s-1e-4)/(1-1e-4),1e-4))
 m.apply_support(p,.1);scale=1e-4+(1-1e-4)*rho
 A=jax.jacfwd(p.material_stress)(fold,scale)
 record('fold_finite_objective_phi_'+str(rho),bool(jnp.all(jnp.isfinite(A))) and
   abs(float(p.material_energy(Q@fold,scale)-p.material_energy(fold,scale)))<1e-12)
# Two small periodic HEX27 problems, mass bitwise same; unchanged true material.
rho=np.linspace(0,1,8*27).reshape(8,27)
def build(factor):
 p=make_density_hyperelastic_problem(2,rho_quad=rho,periodic_axes=(0,1,2),element_degree=2,
   quadrature_order=4,material_model='objective_void')
 return m.entry.ExplicitXYZ(m.apply_support(p,factor),force_batch_cells=8)
original=build(1.);modified=build(.1)
mass_error=float(np.max(np.abs(np.asarray(original.mass)/np.asarray(modified.mass)-1)))
record('HRZ_prescription_same_nodal_mass_bitwise_same',np.array_equal(np.asarray(original.nodem),np.asarray(modified.nodem)) and mass_error<2e-14, periodic_scatter_relative_roundoff=mass_error)
q=jnp.asarray(np.random.default_rng(26).normal(0,.00001,(modified.nc,3))).at[modified.pin].set(0.)
h=-.04
coords,grads,weights,_,scale=modified.kernel_geometry
def total_energy(x):
 w=x[modified.ids];H=jnp.zeros((3,3)).at[2,2].set(h)
 FF=jnp.eye(3)+H+jnp.einsum('cni,qnj->cqij',w[modified.device_cells],grads)
 W=jax.vmap(modified.p.material_energy)(FF.reshape(-1,3,3),scale.ravel()).reshape(scale.shape)
 return jnp.sum(W*weights)
grad=jax.grad(total_energy)(q);force=modified.reduce(modified.force(q,h))
err=float(jnp.max(jnp.abs(grad-force)));record('assembled_shared_force_is_energy_gradient',err<1e-11,absolute_error=err)
bound=m.entry.conservative_step_bound(modified,8)(q,h)
record('same_bound_uses_finite_new_tangent',bool(bound['all_material_tangents_finite']) and
 int(bound['invalid_material_points'])==0 and float(bound['row_sum_bound_s_minus2'])>0)
source=m.adapted_controller()
namespace={};exec(compile(source,'r26_controller_check','exec'),m.entry.__dict__,namespace)
from tempfile import TemporaryDirectory
with TemporaryDirectory() as td:
 a=m.arguments(Path(td));a.stability_batch_cells=8;a.control_budget_seconds=60.
 namespace['state_step_target'](original,a,{},2)
 result=json.loads((Path(td)/'result.json').read_text())
 record('adapted_controller_completes_original_motion_at_17pct',abs(result['path'][-1]['compression']-.17)<1e-12 and all(row['invalid_material_points']==0 for row in result['path']))
record('controller_only_two_checked_changes',source.count('if False and high_KE_time')==1 and
 'original_end/math.ceil(original_end/ex.dt_estimate)' in source)
Path(__file__).with_name('local_checks.json').write_text(json.dumps({'checks':checks,'all_pass':all(c['pass'] for c in checks),
 'scope':'Local/shared-kernel consistency only; no complete trajectory or design AD certification'},indent=2)+'\n')
print(json.dumps({'all_pass':True,'checks':len(checks),'force_gradient_error':err}))

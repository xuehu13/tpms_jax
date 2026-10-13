"""Interpret all near-zero modes, evaluate the first deformation-dominated candidate."""
from pathlib import Path
import json,ast,sys,os,time,shutil
os.environ['JAX_PLATFORMS']='cuda,cpu'
R=Path('/home/xuehu/projects/tpms_jax');D=R/'validation/critical_mode_20261007_r21';O=D/'physical_metric'
W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
classification=json.loads((O/'translation_classification.json').read_text())
protocol={'question':'Which extracted near-zero direction represents deformation rather than gross translation, and how stationary is it in the original fine space?',
 'selection':'Report all six translation fractions; take first in increasing absolute eigenvalue with deformation kinetic share greater than uniform translation kinetic share. No use of reference force or eigenvalue shift to select.',
 'states':['accepted_a0.1200.npz','accepted_a0.1600.npz'],'no_new_eigensolve_or_time_advance':True,
 'same_original_material_Gauss_mass_and_pin':True,'full_residual':'Kfine v-lambda MHRZ v on original free periodic classes; this is mode residual, not forward state imbalance.',
 'limit':'A converged restricted mode with a large fine residual is not a certified fine-space critical mode. Domain curvature shares refer to this candidate only.'}
(O/'interpretation_protocol.json').write_text(json.dumps(protocol,indent=2))
sys.path.insert(0,str(R));import numpy as np;import jax;import jax.numpy as jnp
jax.config.update('jax_enable_x64',True)
source=(D/'diagnose.py').read_text();source=source[:source.index('spaces={};checks=[]')]
source=source.replace("assert not (D/'results/result.json').exists() and not (D/'results/partial_result.json').exists()",'None')
ns={};exec(compile(source,str(D/'diagnose.py'),'exec'),ns)
P=ns['prolongation'](8);mass=ns['mass'];rho=ns['rho'];weights=ns['weights'];cid=ns['cid'];grads=ns['grads'];scale=ns['scale'];p=ns['p'];interp=ns['interpolate']
cj=jnp.asarray(cid);gj=jnp.asarray(grads);ww=jnp.asarray(weights)
make_grad=jax.jit(lambda q:jnp.einsum('cni,qnj->cqij',q[cj],gj))
jvp=jax.jit(jax.vmap(lambda f,s,d:jax.jvp(lambda x:p.material_stress(x,s),(f,),(d,))[1]))
assemble=jax.jit(lambda dp:jnp.zeros((len(mass),3)).at[cj.ravel()].add(jnp.einsum('cqij,qnj,cq->cni',dp,gj,ww).reshape(-1,3)))
base=R/'validation/step_control_20261007_r18/controlled_forward';rows=[];started=time.perf_counter()
for info in classification['rows']:
 name=info['state'];candidates=[x for x in info['modes'] if x['uniform_translation_kinetic_fraction']<.5]
 if not candidates:continue
 candidate=candidates[0];rank=candidate['rank_nearest_zero']
 with np.load(O/f'{name[:-4]}_modes.npz') as f:vec=f['vectors'][:,rank];lam=float(f['eigenvalues'][rank])
 qc=np.zeros((4096,3));qc[1:]=vec.reshape(-1,3);mode=P@qc
 sw,_=interp(mode);factor=1/float(np.linalg.norm(sw,axis=1).max());mode*=factor;sw*=factor
 with np.load(base/name) as f:q=f['q'];t=float(f['time'])
 s=np.clip(t/.004,0,1);a=float(.2*(10*s**3-15*s**4+6*s**5))
 F=np.eye(3)+np.diag([0.,0.,-a])+np.asarray(make_grad(jnp.asarray(q)));df=np.asarray(make_grad(jnp.asarray(mode)));parts_dp=[]
 for first in range(0,len(cid),1024):
  parts_dp.append(np.asarray(jvp(jnp.asarray(F[first:first+1024].reshape(-1,3,3)),jnp.asarray(scale[first:first+1024].ravel()),jnp.asarray(df[first:first+1024].reshape(-1,3,3)))).reshape(-1,27,3,3))
 dp=np.concatenate(parts_dp);density=np.sum(df*dp,axis=(-2,-1))
 parts={key:float(np.sum(density*weights*mask)*1000) for key,mask in {'deep_void':rho<=.001,'mixed_tail':(rho>.001)&(rho<.01),'main_interface':(rho>=.01)&(rho<.5),'geometric_core':rho>=.5}.items()}
 grad=np.asarray(assemble(jnp.asarray(dp)));res=grad[1:]-lam*mass[1:,None]*mode[1:]
 fine_res=float(np.linalg.norm(res)/max(np.linalg.norm(grad[1:]),np.linalg.norm(lam*mass[1:,None]*mode[1:]),1e-30))
 delta=np.einsum('qn,cni->cqi',np.asarray(p.fe.shape_vals),mode[cid]);d2=np.sum(delta**2,axis=-1);part=float(np.sum(d2*weights*rho)/np.sum(d2*weights))
 projected=P.T@grad;Mrqc=P.T@(mass[:,None]*mode)
 proj_res=float(np.linalg.norm(projected[1:]-lam*Mrqc[1:])/max(np.linalg.norm(projected[1:]),np.linalg.norm(lam*Mrqc[1:]),1e-30))
 expected=lam*float(np.sum(mode**2*mass[:,None]))*1000;err=abs(sum(parts.values())-expected)/max(abs(expected),1e-12);assert err<1e-6
 row={'state':name,'compression':a,'rank_nearest_zero':rank,'eigenvalue_s_minus2':lam,'uniform_translation_kinetic_fraction':candidate['uniform_translation_kinetic_fraction'],'curvature_for_1mm_max_surface_mode_N_mm':sum(parts.values()),'parts_N_mm':parts,'occupancy_weighted_displacement_fraction':part,'surface_RMS_mm':float(np.sqrt(np.sum(ns['aw']*np.sum(sw**2,axis=1))/ns['aw'].sum())),'full_background_max_mode_mm':float(np.linalg.norm(mode,axis=1).max()*10),'generalized_fine_eigen_residual':fine_res,'projected_generalized_residual':proj_res,'direct_curvature_vs_eigen_mass_relative_error':err}
 np.savez_compressed(O/f'{name[:-4]}_deformation_candidate.npz',mode=mode,surface_mode_mm=sw,surface=ns['surface'],triangles=ns['triangles'])
 rows.append(row);print(json.dumps(row),flush=True)
(O/'deformation_candidate.json').write_text(json.dumps({'protocol':protocol,'rows':rows,'wall_seconds':time.perf_counter()-started,'not_a_full_space_or_static_critical_certification':True},indent=2))
for name in ['classify.py','interpret_modes.py']:shutil.copy2(W/'work/critical_mode_20261007'/name,D/name)

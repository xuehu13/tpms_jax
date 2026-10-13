"""Read-only r17 diagnostics; shared material/basis, no second FEM solver."""
from pathlib import Path
import hashlib, json, os, sys, time
os.environ['JAX_PLATFORMS']='cpu'
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
R=Path('/home/xuehu/projects/tpms_jax')
sys.path.insert(0,str(R))
import numpy as np
import jax
import jax.numpy as jnp
import basix
from jax_fem.basis import get_elements
from hyperelastic_fem import (make_density_hyperelastic_problem,
    objective_void_energy, objective_void_first_piola, void_nh_cutoff)
jax.config.update('jax_enable_x64',True)
O=R/'validation/mechanism_20261007_r17'
D=O/'original_diagnostic';old=R/'validation/geometry_transfer_20261006_r15'
A=O/'analysis';A.mkdir(exist_ok=False)
started=time.perf_counter();N=32;L=10.;levels=65
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def finite(x):return float(x) if np.isfinite(x) else None
def compression(t):
    s=np.clip(t/.004,0,1)
    return .2*(10*s**3-15*s**4+6*s**5)

# Reuse installed one-cell reference map and formal node ordering. The full
# cached quadrature coordinates verify the translation/scaling convention.
p=make_density_hyperelastic_problem(1,rho_quad=1.,eta=1e-4,
    periodic_axes=(0,1,2),element_degree=2,quadrature_order=4,
    material_model='objective_void')
grads=np.asarray(p.fe.shape_grads[0])*N
family,cell,_,_,degree,order=get_elements('HEX27')
local=np.rint(2*basix.create_element(family,cell,degree).points[order]).astype(int)
origins=2*np.indices((N,)*3).reshape(3,-1).T
indices=origins[:,None,:]+local[None,:,:]
cells=(indices[...,0]*levels+indices[...,1])*levels+indices[...,2]
with np.load(old/'gauss_field.npz') as f:
    rho=f['rho'];weights=f['JxW'];points=f['physical_quad_points']
with np.load(old/'hrz_mass.npz') as f:ids=f['class_ids']
predicted=origins[:,None,:]/(2*N)+np.asarray(p.physical_quad_points)[0][None,:,:]/N
geometry_error=float(np.max(np.abs(predicted-points)))
assert geometry_error<5e-14
write(A/'definitions.json',{'shared_basis':'installed JAX-FEM/Basix HEX27, one-cell Cartesian map scaled by N',
    'cached_quadrature_coordinate_max_abs_error':geometry_error,
    'material_source_sha256':sha(R/'hyperelastic_fem.py'),
    'Gauss_sha256':sha(old/'gauss_field.npz'),
    'scope':'Read-only field/material diagnostics. Local Hessians are mechanical tangents, not design/path AD certification. No alternative FEM or new trajectory.'})

evalbatch=jax.jit(jax.vmap(lambda F,r:(objective_void_energy(F,r),objective_void_first_piola(F,r))))
tangent=jax.jit(jax.jacfwd(objective_void_first_piola,argnums=0))
def fields(statefile):
    with np.load(statefile) as f:q=f['q'];v=f['vhalf'];t=float(f['time']);dt=float(f['dt'])
    a=float(compression(t))
    F=np.eye(3)+np.diag([0.,0.,-a])+np.einsum('cni,qnj->cqij',q[ids][cells],grads)
    J=np.linalg.det(F);flat=F.reshape(-1,3,3);rf=rho.ravel()
    W=np.empty(len(flat));P=np.empty_like(flat)
    for start in range(0,len(flat),4096):
        ww,pp=evalbatch(flat[start:start+4096],rf[start:start+4096])
        W[start:start+4096]=np.asarray(ww);P[start:start+4096]=np.asarray(pp)
    W=W.reshape(rho.shape);P=P.reshape(F.shape)
    pn=np.linalg.norm(P,axis=(-2,-1));fn=np.linalg.norm(F,axis=(-2,-1))
    domains={'deep_void':rho<=.001,'mixed_virtual_tail':(rho>.001)&(rho<.01),
             'uncontinued_NH':rho>=.01,'wall_core':rho>=.5}
    summary={'state':str(statefile.relative_to(O)),'time_s':t,'compression':a,'dt_seconds':dt,
      'q_finite':bool(np.isfinite(q).all()),'vhalf_finite':bool(np.isfinite(v).all()),
      'nonfinite_F_points':int((~np.isfinite(F).all(axis=(-2,-1))).sum()),
      'nonfinite_P_points':int((~np.isfinite(P).all(axis=(-2,-1))).sum()),
      'nonfinite_W_points':int((~np.isfinite(W)).sum()),
      'invalid_uncontinued_NH_points':int(((rho>=.01)&(J<=0)).sum()),
      'total_energy_N_mm':finite(np.sum(W*weights)*L**3),
      'q_norm_max_mm':finite(np.linalg.norm(q,axis=-1).max()*L),
      'vhalf_norm_max_mm_s':finite(np.linalg.norm(v,axis=-1).max()*L), 'domains':{}}
    candidates=[]
    for name,mask in domains.items():
        jm=J[mask];pm=pn[mask];fm=fn[mask];wm=W[mask]
        summary['domains'][name]={'count':int(mask.sum()),'negative_J_count':int((jm<=0).sum()),
          'nonfinite_J_count':int((~np.isfinite(jm)).sum()),
          'nonfinite_P_count':int((~np.isfinite(pm)).sum()),'nonfinite_W_count':int((~np.isfinite(wm)).sum()),
          'finite_J_min':finite(jm[np.isfinite(jm)].min()) if np.isfinite(jm).any() else None,
          'finite_P_norm_max_MPa':finite(pm[np.isfinite(pm)].max()) if np.isfinite(pm).any() else None,
          'finite_F_norm_max':finite(fm[np.isfinite(fm)].max()) if np.isfinite(fm).any() else None,
          'energy_N_mm':finite(np.sum(wm*weights[mask])*L**3)}
        # Extremes plus a typical point, not an exhaustive stability estimate.
        valid=mask&np.isfinite(J)&np.isfinite(pn)
        if valid.any():
            validflat=np.flatnonzero(valid)
            candidates.extend([int(validflat[np.argmin(J.ravel()[validflat])]),
                               int(validflat[np.argmax(pn.ravel()[validflat])]),
                               int(validflat[len(validflat)//2])])
    bad=(~np.isfinite(W))|(~np.isfinite(P).all(axis=(-2,-1)))|((rho>=.01)&(J<=0))
    candidates.extend(np.flatnonzero(bad)[:12].tolist())
    samples=[]
    for idx in sorted(set(candidates)):
        c,g=divmod(idx,27);FF=flat[idx];rr=float(rf[idx])
        sample={'cell_index':c,'Gauss_index':g,'reference_xyz_mm':(points[c,g]*L).tolist(),
          'rho':rr,'actual_J':finite(J[c,g]),'F_norm':finite(fn[c,g]),
          'P_norm_MPa':finite(pn[c,g]),'W_MPa':finite(W[c,g]),
          'NH_continuation_cutoff':float(void_nh_cutoff(rr)),
          'F':[[finite(x) for x in row] for row in FF]}
        if np.isfinite(FF).all() and np.isfinite(P[c,g]).all():
            H=np.asarray(tangent(FF,rr)).reshape(9,9)
            sample['tangent_finite']=bool(np.isfinite(H).all())
            if np.isfinite(H).all():
                eig=np.linalg.eigvalsh((H+H.T)/2)
                sample['material_tangent_eigen_min_MPa']=float(eig.min())
                sample['material_tangent_eigen_max_MPa']=float(eig.max())
        samples.append(sample)
    summary['selected_material_points']=samples
    write(A/(statefile.stem+'_material.json'),summary)
    return q,a,summary

with np.load(O/'shell_frames/reference.npz') as f:
    sx=f['xyz_mm'];tri=f['triangles'];area=f['reference_area_weights_mm2'];normals=f['reference_vertex_normals']
sr=json.loads((O/'shell_frames/modes.json').read_text())['rows']
def interpolate(q):
    x=sx/L;ci=np.minimum(np.floor(x*N).astype(int),N-1);ci=np.maximum(ci,0)
    s=x*N-ci
    basis=np.stack(((2*s-1)*(s-1),4*s*(1-s),s*(2*s-1)),axis=-1)
    w=np.zeros_like(x)
    for i in range(3):
      for j in range(3):
       for k in range(3):
        ijk=2*ci+np.array([i,j,k]);node=(ijk[:,0]*levels+ijk[:,1])*levels+ijk[:,2]
        w+=q[ids[node]]*(basis[:,0,i]*basis[:,1,j]*basis[:,2,k])[:,None]
    w*=L;translation=np.average(w,axis=0,weights=area)
    return w-translation,translation

states=sorted(D.glob('accepted_a*.npz'))
for name in ('last_accepted.npz','first_rejection_replay/before_first_invalid.npz','first_rejection_replay/first_invalid.npz'):
    if (D/name).exists():states.append(D/name)
summary=[];comparisons=[]
for statefile in states:
    q,a,row=fields(statefile);summary.append(row)
    if not statefile.name.startswith('accepted_a'):continue
    w,translation=interpolate(q);normal=np.einsum('ni,ni->n',w,normals)
    near=min(sr,key=lambda r:abs(r['compression']-a))
    with np.load(O/'shell_frames'/near['file']) as f:sw=f['fluctuation_mm']
    rms=lambda x:float(np.sqrt(np.average(np.sum(x*x,axis=1),weights=area)))
    cosine=float(np.sum(area[:,None]*w*sw)/np.sqrt(np.sum(area[:,None]*w*w)*np.sum(area[:,None]*sw*sw)))
    name=statefile.stem+'_midsurface.npz'
    np.savez_compressed(A/name,xyz_mm=sx,triangles=tri,jax_fluctuation_mm=w,
      jax_normal_fluctuation_mm=normal,jax_current_xyz_mm=sx+w+sx*np.array([0.,0.,-a]),
      removed_translation_mm=translation,compression=a)
    comparisons.append({'state':str(statefile.relative_to(O)),'file':name,'compression':a,
      'JAX_fluctuation_RMS_mm':rms(w),'shell_nearest_file':near['file'],
      'shell_nearest_compression':near['compression'], 'compression_difference':a-near['compression'],
      'shell_nearest_fluctuation_RMS_mm':rms(sw),'area_weighted_pattern_cosine':cosine,
      'scope':'Actual accepted JAX field Q2-interpolated on original midsurface; compared with nearest saved shell frame. Unequal compression; cosine is a pattern diagnostic, not eigenmode or accuracy certification.'})
    print(json.dumps({'state':statefile.name,'compression':a,'invalid_NH':row['invalid_uncontinued_NH_points'],
                      'nonfinite_P':row['nonfinite_P_points']}),flush=True)
write(A/'material_summary.json',summary)
write(A/'mode_comparison.json',comparisons)
write(A/'analysis_receipt.json',{'wall_seconds':time.perf_counter()-started,'states_analyzed':len(states),
  'new_solver_jobs':0,'new_design_AD_jobs':0,'only_shared_material_mechanical_tangents':True})
print(json.dumps({'wall_seconds':time.perf_counter()-started,'states_analyzed':len(states)}))

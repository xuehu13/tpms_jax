"""Read-only Gauss localization of full AD vs finite-perturbation state strain."""
from pathlib import Path
import json,os,shutil,time
os.environ.setdefault('JAX_PLATFORMS','cpu')
import numpy as np
import basix
from jax_fem.basis import get_elements
R=Path('/home/xuehu/projects/tpms_jax');B=Path(__file__).resolve().parent
E=R/'validation/large_compression_20261005_r6/gradient20_path_20261005'
out=E/'tangent_localization';out.mkdir(exist_ok=False);started=time.perf_counter()
with np.load(E/'path_ad_full/field.npz') as f:q=f['q']
with np.load(E/'path_ad_full/tangent_field.npz') as f:dq=f['q']
with np.load(E/'adaptive_plus/field.npz') as p,np.load(E/'adaptive_minus/field.npz') as m:fd=(p['q']-m['q'])/.005
with np.load(R/'validation/large_compression_20261005_r6/quadratic_candidate/gauss_field.npz') as f:phi=f['rho'];xi=f['physical_quad_points'][0]*32
family,cell,_,_,degree,order=get_elements('HEX27');el=basix.create_element(family,cell,degree)
local=np.rint(2*el.points[order]).astype(int)
grads=el.tabulate(1,xi)[1:4,:,: ,0].take(order,axis=2).transpose(1,2,0)*32
weights=np.prod(np.where(np.abs(xi-.5)<1e-10,4/9,5/18),axis=1)/32**3
groups={name:{'volume':0.,'AD_strain_square_integral':0.,'FD_strain_square_integral':0.} for name in ['soft_phi_lt_0p01','interface','solid_phi_ge_0p9']}
minimum=np.inf;maxstrain=0.;maximum_phi=None
for start in range(0,32**3,1024):
    k=np.arange(start,min(start+1024,32**3));cellid=np.stack([k//32**2,(k//32)%32,k%32],axis=1)
    grid=(2*cellid[:,None,:]+local[None,:,:])%64;ids=(grid[...,0]*64+grid[...,1])*64+grid[...,2]
    F=np.eye(3)+np.diag([0,0,-.2])+np.einsum('cni,qnj->cqij',q[ids],grads)
    D=np.einsum('cni,qnj->cqij',dq[ids],grads);DF=np.einsum('cni,qnj->cqij',fd[ids],grads)
    d2=np.sum(D**2,axis=(2,3));f2=np.sum(DF**2,axis=(2,3));rho=phi[k]
    minimum=min(minimum,float(np.linalg.det(F).min()));i=np.unravel_index(np.argmax(d2),d2.shape)
    if d2[i]>maxstrain:maxstrain=float(d2[i]);maximum_phi=float(rho[i])
    masks={'soft_phi_lt_0p01':rho<.01,'solid_phi_ge_0p9':rho>=.9,'interface':(rho>=.01)&(rho<.9)}
    for name,mask in masks.items():
        groups[name]['volume']+=float(np.sum(mask*weights))
        groups[name]['AD_strain_square_integral']+=float(np.sum(d2*mask*weights))
        groups[name]['FD_strain_square_integral']+=float(np.sum(f2*mask*weights))
adtotal=sum(g['AD_strain_square_integral'] for g in groups.values());fdtotal=sum(g['FD_strain_square_integral'] for g in groups.values())
for g in groups.values():
    g['AD_squared_strain_fraction']=g['AD_strain_square_integral']/adtotal
    g['FD_squared_strain_fraction']=g['FD_strain_square_integral']/fdtotal
    g['AD_Gauss_strain_sensitivity_RMS_per_mm']=np.sqrt(g['AD_strain_square_integral']/g['volume'])
    g['FD_Gauss_strain_sensitivity_RMS_per_mm']=np.sqrt(g['FD_strain_square_integral']/g['volume'])
record=json.loads((E/'path_ad_full/result.json').read_text())['path'][-1]['J_min']
assert abs(minimum-record)<1e-12,(minimum,record)
result={'status':'saved_tangent_localization_complete','groups':groups,'Gauss_min_J':minimum,
        'J_reconstruction_absolute_difference':abs(minimum-record),'phi_at_maximum_AD_Gauss_strain':maximum_phi,
        'scope':'Reference-volume-weighted strain sensitivity, not energy fraction or definitive causal diagnosis',
        'body_seconds':time.perf_counter()-started}
(out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');shutil.copy2(__file__,out/'experiment.py')
print(json.dumps(result,indent=2))

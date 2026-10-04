from pathlib import Path
import json,time
import numpy as np
from scipy.special import expit
R=Path('/home/xuehu/projects/tpms_jax');O=R/'validation/learning_bridge_20261003_r3'
assert json.loads((O/'step1/summary.json').read_text())['passed']
S=O/'step2';S.mkdir(exist_ok=False);start=time.perf_counter()
N=64;target=.3505280533
offset=(1+np.array([-1.,1.])/np.sqrt(3))/2
x=(np.arange(N)[:,None]+offset).ravel()/N
xyz=np.stack(np.meshgrid(x,x,x,indexing='ij'),axis=-1)
phase=2*np.pi*xyz
g=np.sin(phase[...,0])*np.cos(phase[...,1])+np.sin(phase[...,1])*np.cos(phase[...,2])+np.sin(phase[...,2])*np.cos(phase[...,0])
basis=np.cos(phase)
def fraction(c0,modulation):
 c=c0+modulation
 return float(np.mean(expit(40*(g+c))-expit(40*(g-c))))
anchors=[]
for label,a in [('ax035',np.array([.035,0.,0.])),('az035',np.array([0.,0.,.035]))]:
 modulation=basis@a;lo=.525;hi=.555
 assert fraction(lo,modulation)<target<fraction(hi,modulation)
 for iteration in range(50):
  c0=(lo+hi)/2;vf=fraction(c0,modulation)
  if abs(vf-target)<1e-10:break
  if vf<target:lo=c0
  else:hi=c0
 q=np.r_[c0,a];assert .48<q[0]-abs(a).sum() and q[0]+abs(a).sum()<.60
 anchors.append({'label':label,'theta':q.tolist(),'projected_N64_Vf':vf,'Vf_target':target,'geometric_volume_absolute_error':abs(vf-target),'binary_Gauss_Vf':float(np.mean(abs(g)<=c0+modulation)),'global_min_c':float(c0-abs(a).sum()),'global_max_c':float(c0+abs(a).sum()),'definition':'|G|<=c0+ax*cos(2pi*x)+ay*cos(2pi*y)+az*cos(2pi*z)','base_material':'single homogeneous E=10 nu=.3','geometry_only_volume_root':True})
(S/'anchors.json').write_text(json.dumps(anchors,indent=2)+'\n')
(S/'plan.json').write_text(json.dumps({'anchor_count':2,'N':[48,64],'G':[32,48],'beta':40,'eta':1e-4,'target_Vf':target,'c0_calibration':'Geometric N64 eight-Gauss uniform weights only, no force or Abaqus fitting','max_JAX_forward':4,'max_Abaqus_analysis':4,'background_mesh_relative_tolerance':.01,'binary_mesh_relative_tolerance':.01,'reference_relative_tolerance':.02,'binary_volume_absolute_tolerance':.003,'physical_span_rule':'>3 times max relevant absolute mesh changes','locked_before_solves':True},indent=2)+'\n')
(S/'geometry_setup.json').write_text(json.dumps({'seconds':time.perf_counter()-start,'uniform_c0_projected_Vf':fraction(.541062,0.),'full_geometry_mean_uses_equally_weighted_actual_uniform_HEX8_Gauss_set':True},indent=2)+'\n')
print(json.dumps(anchors),flush=True)

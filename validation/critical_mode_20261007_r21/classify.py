"""Report all near-zero modes' translation participation before selecting a folding candidate."""
from pathlib import Path
import json,ast,time
import numpy as np
from scipy import sparse
R=Path('/home/xuehu/projects/tpms_jax');D=R/'validation/critical_mode_20261007_r21';O=D/'physical_metric'
with np.load(R/'validation/geometry_transfer_20261006_r15/hrz_mass.npz') as f:mass=f['periodic_class_mass_normalized']
tree=ast.parse((D/'diagnose.py').read_text());names=['shape','dshape','coarse_basis','prolongation']
fns=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names]
ns={'np':np,'sparse':sparse,'fine_coordinates':np.indices((64,)*3).reshape(3,-1).T/64,'lex':np.indices((3,)*3).reshape(3,-1).T}
exec(compile(ast.Module(body=fns,type_ignores=[]),'pure_prolongation','exec'),ns)
P=ns['prolongation'](8);records=[]
for name in ['accepted_a0.1200.npz','accepted_a0.1600.npz']:
 with np.load(O/f'{name[:-4]}_modes.npz') as f:values=f['eigenvalues'];vectors=f['vectors']
 rows=[]
 for i,value in enumerate(values):
  qc=np.zeros((4096,3));qc[1:]=vectors[:,i].reshape(-1,3);mode=P@qc
  mean=np.sum(mode*mass[:,None],axis=0)/mass.sum()
  inertia=float(np.sum(mode**2*mass[:,None]));translation=float(mass.sum()*np.sum(mean**2));share=translation/inertia
  rows.append({'rank_nearest_zero':i,'eigenvalue_s_minus2':float(value),'uniform_translation_kinetic_fraction':share,
    'Rayleigh_after_mass_mean_removal_s_minus2':float(value/(1-share))})
 records.append({'state':name,'modes':rows})
(O/'translation_classification.json').write_text(json.dumps({'rows':records,'all_modes_reported':True,'not_new_eigenvalues_after_mean_removal':True,'no_time_advance_no_design_AD':True},indent=2))
print(json.dumps(records,indent=2))

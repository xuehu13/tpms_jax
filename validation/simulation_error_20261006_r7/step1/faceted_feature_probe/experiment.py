"""Check the three already flagged faceted locations; no new mechanics."""
from pathlib import Path
import json,shutil,sys
import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import brentq
from scipy.special import expit
R=Path('/home/xuehu/projects/tpms_jax');sys.path.insert(0,str(R))
from surface_distance import PeriodicSurfaceDistance
O=R/'validation/simulation_error_20261006_r7/step1/faceted_feature_probe';O.mkdir(exist_ok=False)
with np.load(R/'validation/thin_target_20261004_r5/gauss_field.npz') as f:v=f['surface_vertices'];tri=f['surface_triangles']
flags=json.loads((R/'validation/thin_target_20261004_r5/local_geometry_diagnostics.json').read_text())['facet_flags']
query=PeriodicSurfaceDistance(v,tri);z=np.linspace(-.45,.45,3601);ell=.05/(2*np.log(9));rows=[]
for index in sorted({x['facet_index_zero_based'] for x in flags}):
    corners=v[tri[index]];p=corners.mean(axis=0);normal=np.cross(corners[1]-corners[0],corners[2]-corners[0]);normal/=np.linalg.norm(normal)
    d=query.query(p+z[:,None]/10*normal)*10
    root=lambda x:float(query.query(p+x/10*normal)*10)-.25
    low=brentq(root,-.4,-.1);high=brentq(root,.1,.4)
    phi=expit((.25-d)/ell);m0=trapezoid(phi,z);m1=trapezoid(phi*z,z);m2=trapezoid(phi*z*z,z)-m1*m1/m0
    rows.append({'facet_index':index,'sharp_width_mm':high-low,'sharp_width_over_target':(high-low)/.5,
                 'smooth_M0_mm':float(m0),'smooth_central_M2_mm3':float(m2),
                 'smooth_bending_ratio_vs_nominal_t3_over_12':float(m2/(.5**3/12))})
result={'status':'flagged_feature_profiles_complete','rows':rows,
        'scope':'Deliberately selected worst faceting indicators; no surface-wide weighting or global force correction.'}
(O/'result.json').write_text(json.dumps(result,indent=2)+'\n');shutil.copy2(__file__,O/'experiment.py');print(json.dumps(result,indent=2))

from pathlib import Path
import json
import sys
import numpy as np
from scipy.stats import qmc

P=Path('/home/xuehu/projects/tpms_jax')
O=P/'validation/mechanics_trust_20261003_r4/step2'
sys.path.insert(0,str(P))
from binary_gyroid import implicit_numpy, linear_domain, audit_linear


def grad(x,family):
    a,b,c=np.moveaxis(2*np.pi*x,-1,0)
    if family=='primitive':
        return -2*np.pi*np.sin(2*np.pi*x)
    return 2*np.pi*np.column_stack((np.cos(a)*np.cos(b)-np.sin(c)*np.sin(a),
                                    np.cos(b)*np.cos(c)-np.sin(a)*np.sin(b),
                                    np.cos(c)*np.cos(a)-np.sin(b)*np.sin(c)))


records=[]
for family,c,label in [('gyroid',.25,'thin_gyroid'),('primitive',.6,'primitive')]:
    fractions=[]
    for seed in range(4):
        x=qmc.Sobol(3,scramble=True,seed=20261004+seed).random_base2(18)
        fractions.append(float(np.mean(np.abs(implicit_numpy(x,family))<=c)))
    x=qmc.Sobol(3,scramble=True,seed=20261004).random_base2(12)
    for _ in range(15):
        g=implicit_numpy(x,family);v=grad(x,family)
        step=g[:,None]*v/np.sum(v*v,axis=1)[:,None]
        x=(x-np.clip(step,-.1,.1))%1
    valid=np.abs(implicit_numpy(x,family))<1e-10
    x=x[valid];normal=grad(x,family);normal/=np.linalg.norm(normal,axis=1)[:,None]
    distances=[]
    for sign in (-1,1):
        lo=np.zeros(len(x));hi=np.full(len(x),.2)
        assert np.all(sign*implicit_numpy(x+sign*hi[:,None]*normal,family)>c)
        for _ in range(45):
            middle=(lo+hi)/2
            outside=sign*implicit_numpy(x+sign*middle[:,None]*normal,family)>c
            hi=np.where(outside,middle,hi);lo=np.where(outside,lo,middle)
        distances.append((lo+hi)/2)
    thickness=np.sum(distances,axis=0)
    pts,tets=linear_domain(16,c=c,family=family)
    audit=audit_linear(pts,tets,c,family)
    record={'label':label,'family':family,'c':c,'volume_samples':fractions,'binary_vf_estimate':float(np.mean(fractions)),
            'volume_replicate_std':float(np.std(fractions,ddof=1)),
            'thickness_method':'sampled mid-surface normal, roots at both signed levels; not a rigorous global minimum',
            'mid_surface_samples':len(x),'thickness_min_sampled':float(thickness.min()),
            'thickness_quantiles':np.quantile(thickness,[.01,.5,.99]).tolist(),
            'thinnest_sample_position':x[np.argmin(thickness)].tolist(),
            'minimum_wall_cells_N48':float(thickness.min()*48),'minimum_wall_cells_N64':float(thickness.min()*64),
            'G16_clipped_geometry_audit':audit}
    records.append(record)
    print(json.dumps({k:v for k,v in record.items() if k!='volume_samples'},indent=2),flush=True)
assert records[0]['binary_vf_estimate']<.22
assert .25<records[1]['binary_vf_estimate']<.45
(O/'geometry_precheck.json').write_text(json.dumps(records,indent=2)+'\n')
(O/'precheck.py').write_bytes(Path(__file__).read_bytes())

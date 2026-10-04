"""Read-only reproduction of the new thin-case rejection; no mesh is accepted."""
from pathlib import Path
import hashlib,json
import numpy as np

P=Path('/home/xuehu/projects/tpms_jax');O=P/'validation/mechanics_trust_20261003_r4/step2'
source=(P/'binary_gyroid.py').read_text()
needle='raise ValueError("Zero-volume clipped element")'
assert source.count(needle)==1
diagnostic='''raise ValueError(json.dumps({"origin":origin.tolist(),"permutation":list(permutation),"det":float(det),"tet_node_ids":tet,"tet_points":xyz.tolist(),"polygon_node_ids":ids,"original_tetra_G":g.tolist(),"threshold":float(c)}))'''
namespace={'json':json,'__name__':'read_only_clip_rejection_diagnostic'}
exec(compile(source.replace(needle,diagnostic),'read_only_clip_diagnostic','exec'),namespace)
try:
    namespace['linear_domain'](48,c=.25)
    raise RuntimeError('Rejection was not reproduced')
except ValueError as exc:
    result=json.loads(str(exc))
result['module_sha256']=hashlib.sha256((P/'binary_gyroid.py').read_bytes()).hexdigest()
result['criterion']='abs(mapping determinant)<1e-20; rejection message alone does not distinguish zero from tiny positive volume'
result['new_FEM_or_Abaqus_solves']=0
(O/'thin_gyroid/reference_failure_detail.json').write_text(json.dumps(result,indent=2)+'\n')
(O/'debug_clip.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(result,indent=2))

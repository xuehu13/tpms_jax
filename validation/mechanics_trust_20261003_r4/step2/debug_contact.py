"""Inspect the second thin-reference rejection in memory only."""
from pathlib import Path
import json,hashlib
P=Path('/home/xuehu/projects/tpms_jax');O=P/'validation/mechanics_trust_20261003_r4/step2'
source=(P/'binary_gyroid.py').read_text()
needle='raise ValueError("Degenerate clipped tetrahedron")'
assert source.count(needle)==1
diagnostic='raise ValueError(json.dumps({"origin":origin.tolist(),"permutation":list(permutation),"original_tetra_G":g.tolist(),"threshold":float(c),"unique_points":[coordinates[i].tolist() for i in unique],"polygons":[[v[:3].tolist() for v in face] for face in polygons]}))'
namespace={'json':json,'__name__':'read_only_contact_diagnostic'}
exec(compile(source.replace(needle,diagnostic),'read_only_contact_diagnostic','exec'),namespace)
try:
    namespace['linear_domain'](48,c=.25)
    raise RuntimeError('Rejection not reproduced')
except ValueError as exc:
    result=json.loads(str(exc))
result['module_sha256']=hashlib.sha256((P/'binary_gyroid.py').read_bytes()).hexdigest()
result['new_FEM_or_Abaqus_solves']=0
(O/'thin_gyroid/reference_contact_failure_detail.json').write_text(json.dumps(result,indent=2)+'\n')
(O/'debug_contact.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(result,indent=2))

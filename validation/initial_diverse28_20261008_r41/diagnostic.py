"""One current N32 diverse_28 initial static solve, reusing frozen r30."""
from pathlib import Path
import hashlib,json
R=Path('/home/xuehu/projects/tpms_jax');D=Path(__file__).resolve().parent
source_path=R/'validation/initial_tangent_20261008_r30/diagnostic.py'
source=source_path.read_text(encoding='utf-8')
old="validation/geometry_transfer_20261006_r15/gauss_field.npz"
assert source.count(old)==1
source=source.replace(old,"validation/large_compression_20261005_r6/quadratic_candidate/gauss_field.npz")
assert source.count("'case':'diverse_04'")==1
source=source.replace("'case':'diverse_04'","'case':'diverse_28'")
(D/'source_adaptation.json').write_text(json.dumps({
 'source_path':str(source_path.relative_to(R)),
 'source_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest(),
 'compiled_source_sha256':hashlib.sha256(source.encode()).hexdigest(),
 'changes':'Only current diverse_28 N32 Gauss cache and case label; r30 shared tangent and all solver gates unchanged',
 'production_changed':False},indent=2)+'\n')
exec(compile(source,str(D/'adapted_source.py'),'exec'),
 {'__file__':str(D/'adapted_source.py'),'__name__':'__main__'})

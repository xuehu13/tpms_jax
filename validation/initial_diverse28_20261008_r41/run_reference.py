"""One native r41 job using the frozen r30 runner with paths changed only."""
from pathlib import Path
import hashlib,json
R=Path(r'\\wsl.localhost\Ubuntu-24.04\home\xuehu\projects\tpms_jax')
D=R/'validation/initial_diverse28_20261008_r41'
old=R/'validation/initial_tangent_20261008_r30/run_abaqus.py'
source=old.read_text(encoding='utf-8')
assert source.count('initial_tangent_20261008_r30')==2
assert source.count('diverse04')==1
source=source.replace('initial_tangent_20261008_r30','initial_diverse28_20261008_r41').replace('diverse04','diverse28')
(D/'abaqus/launch_source.json').write_text(json.dumps({
 'original_runner_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),
 'adapted_runner_sha256':hashlib.sha256(source.encode()).hexdigest(),
 'changes':'Only formal and native paths/case label; one cpus=1 Standard job and one extractor, no retry',
 'source':str(old)},indent=2)+'\n',encoding='utf-8')
exec(compile(source,str(Path(__file__).resolve()),'exec'),{'__file__':str(Path(__file__).resolve()),'__name__':'__main__'})

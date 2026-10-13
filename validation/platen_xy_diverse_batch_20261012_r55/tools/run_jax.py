"""Data-only adapter; same bounded r52 loop and r51 shared mechanics."""
from pathlib import Path
import hashlib,json,sys
B=Path(__file__).resolve().parents[1];R=B.parents[1];kind=sys.argv[1]
source=R/'validation/platen_xy_PD_20261010_r52/tools/run_jax.py';original=source.read_text()
code=original.replace("choices=['P','D']",'choices='+repr(json.loads((B/'protocol.json').read_text())['cases']))
assert code!=original
(B/kind/'checks/runner_recipe.json').write_text(json.dumps({'source':str(source),'source_sha256':hashlib.sha256(original.encode()).hexdigest(),'adapted_sha256':hashlib.sha256(code.encode()).hexdigest(),'change':'argument choices only'},indent=2)+'\n')
exec(compile(code,str(source),'exec'),{'__name__':'__main__','__file__':str(Path(__file__).resolve())})

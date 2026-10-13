from pathlib import Path
import hashlib,json
B=Path(__file__).resolve().parents[1];R=B.parents[1]
source=R/'validation/platen_xy_PD_20261010_r52/tools/run_jax.py'
original=source.read_text();assert original.count("choices=['P','D']")==1
code=original.replace("choices=['P','D']","choices=['diverse_04']")
(B/'checks/run_jax_recipe.json').write_text(json.dumps({'source':str(source),'source_sha256':hashlib.sha256(original.encode()).hexdigest(),'substitution':"choices=['diverse_04']",'adapted_sha256':hashlib.sha256(code.encode()).hexdigest(),'same_mechanics_and_100min_budget':True},indent=2)+'\n')
exec(compile(code,str(source),'exec'),{'__name__':'__main__','__file__':str(Path(__file__).resolve())})

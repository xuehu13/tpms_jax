"""Reuse bounded r52 physical loop, injecting only HEX8 build and this budget."""
from pathlib import Path
import hashlib,json,sys
B=Path(__file__).resolve().parents[1];R=B.parents[1]
source=R/'validation/platen_xy_PD_20261010_r52/tools/run_jax.py';original=source.read_text();code=original
changes=[("choices=['P','D']","choices="+repr(json.loads((B/'protocol.json').read_text())['cases'])),
 ("build,estimate_initial_dt,jax,jnp,np=record.build,record.estimate_initial_dt,record.jax,record.jnp,record.np", "estimate_initial_dt,jax,jnp,np=record.estimate_initial_dt,record.jax,record.jnp,record.np\nfrom hex8_model import build"),
 ("deadline=min(START+3900,clock['started_unix']+6000-300)","deadline=START+2400")]
for old,new in changes:assert code.count(old)==1,(old,code.count(old));code=code.replace(old,new)
E=B/sys.argv[1]
(E/'checks/runner_recipe.json').write_text(json.dumps({'source':str(source),'source_sha256':hashlib.sha256(original.encode()).hexdigest(),'substitutions':changes,'adapted_sha256':hashlib.sha256(code.encode()).hexdigest()},indent=2)+'\n')
exec(compile(code,str(source),'exec'),{'__name__':'__main__','__file__':str(Path(__file__).resolve())})

"""Reuse the completed r54 single-pair coordinator with only case paths changed."""
from pathlib import Path
import hashlib,json,sys
K=sys.argv[1];B=Path(__file__).resolve().parents[1];R=B.parents[1]
cases=json.loads((B/'protocol.json').read_text())['cases'];assert K in cases
W=Path(r'C:\Users\xuehu\Documents\Codex\2026-10-01\referenced-chatgpt-conversation-this-is-an')
out=W/'output/r55_platen_xy_diverse_batch_20261012'/K;out.mkdir(exist_ok=True)
source=R/'validation/platen_xy_diverse04_20261012_r54/tools/run_pair.py';original=source.read_text();code=original
changes=[('platen_xy_diverse04_20261012_r54','platen_xy_diverse_batch_20261012_r55'),
 ('output/r54_platen_xy_diverse04_20261012',f'output/r55_platen_xy_diverse_batch_20261012/{K}'),
 ('r54_diverse04_midsurface_end_20pct_20261012',f'r55_{K}_midsurface_end_20pct_20261012'),
 ('diverse_04',K),
 ("B/'checks/pair_started.json'","E/'checks/pair_started.json'"),
 ("wsl('analyze.py',lim)",f"wsl('analyze.py',lim,{K!r})"),
 ("wsl('compare_peaks.py',lim)",f"wsl('compare_peaks.py',lim,{K!r})"),
 ("s.update(stage=stage,active_jobs=active or [],**kw);", "s.update(stage=stage,current_case=K,active_jobs=active or [],**kw);s['cases'][K].update(stage=stage,**kw);"),
 ("shutil.copy2(p,O/p.name)","shutil.copy2(p,O/p.name)"),
 ("write(O/'status.json',s)","write(O/'status.json',s);write(O.parent/'status.json',s)")]
for old,new in changes:
 assert old in code,old;code=code.replace(old,new)
(B/'checks'/f'{K}_coordinator_recipe.json').write_text(json.dumps({'source':str(source),'source_sha256':hashlib.sha256(original.encode()).hexdigest(),'substitutions':changes,'adapted_sha256':hashlib.sha256(code.encode()).hexdigest(),'same_100min_budget_and_single_attempt':True},indent=2)+'\n')
exec(compile(code,str(source),'exec'),{'__name__':'__main__','__file__':str(Path(__file__).resolve()),'K':K})

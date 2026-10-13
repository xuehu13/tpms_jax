from pathlib import Path
import hashlib,json
B=Path(__file__).resolve().parents[1];R=B.parents[1]
source=R/'validation/platen_xy_PD_20261010_r52/tools/prepare_case.py'
original=source.read_text();code=original
replacements=[("choices=['P','D']","choices=['diverse_04']"),
 ("old=R/'validation/canonical_tpms_20261009_r50'/a.kind","old=R/'validation/geometry_transfer_20261006_r15'"),
 ("'surface.npz'","'surface_geometry.npz'"),('r52','r54')]
for old,new in replacements:
 assert old in code,old
 code=code.replace(old,new)
(B/'checks/prepare_recipe.json').write_text(json.dumps({'source':str(source),'source_sha256':hashlib.sha256(original.encode()).hexdigest(),'substitutions':replacements,'adapted_sha256':hashlib.sha256(code.encode()).hexdigest()},indent=2)+'\n')
exec(compile(code,str(source),'exec'),{'__name__':'__main__','__file__':str(Path(__file__).resolve())})

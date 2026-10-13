"""Reuse r51 field/health postprocessing; no solver/model rebuild."""
from pathlib import Path
import hashlib,json,sys
B=Path(__file__).resolve().parents[1];R=B.parents[1];kind=sys.argv[1];E=B/kind;G=R/'validation/platen_xy_G_20261010_r51'
cfg=json.loads((E/'input/protocol.json').read_text())
source=G/'experiments/midsurface_rim_20pct_20261010/tools/analyze_rim20.py';original=source.read_text();code=original
substitutions=[("E=Path(__file__).resolve().parents[1];D=E.parents[1];R=D.parents[1]",f"E=Path({str(E)!r});D=Path({str(G)!r});R=Path({str(R)!r})"),
 ("R/'validation/canonical_tpms_20261009_r50/G/surface_shell_quality.npz'",f"Path({cfg['source_shell_surface']!r})"),
 ("D/'input/cut_face_contact.npz'","E/'input/cut_face_contact.npz'"),
 ("fig.suptitle('G | L10 mm",f"fig.suptitle('{kind} | L10 mm"),
 (",'contextual_original2pct_peak_N':fixed",''),( ",'curve_RMS_over_old_fixed_peak':primary/fixed",'')]
for old,new in substitutions:assert code.count(old)==1,(old,code.count(old));code=code.replace(old,new)
(E/'checks/postprocess_recipe.json').write_text(json.dumps({'source':str(source),'source_sha256':hashlib.sha256(original.encode()).hexdigest(),'substitutions':substitutions,'adapted_sha256':hashlib.sha256(code.encode()).hexdigest(),'new_FEM_calls':False},indent=2)+'\n')
exec(compile(code,str(source),'exec'),{'__name__':'__main__','__file__':str(source)})

"""Apply the frozen G postprocessor with explicit data/path/title substitutions."""
from pathlib import Path
import argparse, hashlib, json
B=Path(__file__).resolve().parents[1];R=B.parents[1];G=R/'validation/platen_xy_G_20261010_r51'
ap=argparse.ArgumentParser();ap.add_argument('kind',choices=['P','D']);args=ap.parse_args();E=B/args.kind
cfg=json.loads((E/'input/protocol.json').read_text())
source=G/'experiments/midsurface_rim_20pct_20261010/tools/analyze_rim20.py';original=source.read_text();code=original
substitutions=[
    ("E=Path(__file__).resolve().parents[1];D=E.parents[1];R=D.parents[1]",f"E=Path({str(E)!r});D=Path({str(G)!r});R=Path({str(R)!r})"),
    ("fixed=.7241182386842194", "fixed=4.123236290287"),
    ("'contextual_original2pct_peak_N'", "'contextual_G_full20_peak_N'"),
    ("'curve_RMS_over_old_fixed_peak'", "'curve_RMS_over_fixed_G_peak'"),
    ("R/'validation/canonical_tpms_20261009_r50/G/surface_shell_quality.npz'", f"Path({cfg['source_shell_surface']!r})"),
    ("D/'input/cut_face_contact.npz'", "E/'input/cut_face_contact.npz'"),
    ("fig.suptitle('G | L10 mm", f"fig.suptitle('{args.kind} | L10 mm"),
]
for old,new in substitutions:
    assert code.count(old)==1,(old,code.count(old));code=code.replace(old,new)
recipe={'source':str(source),'source_sha256':hashlib.sha256(original.encode()).hexdigest(),'substitutions':substitutions,
        'adapted_sha256':hashlib.sha256(code.encode()).hexdigest(),'mechanics_or_FEM_calls':False}
(E/'checks/postprocess_recipe.json').write_text(json.dumps(recipe,indent=2)+'\n')
exec(compile(code,str(source),'exec'),{'__name__':'__main__','__file__':str(source)})

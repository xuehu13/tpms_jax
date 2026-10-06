"""Retain matched shell evidence. ODBs remain in their native job directories."""
from pathlib import Path
import argparse, hashlib, json, shutil, re
R=Path('/home/xuehu/projects/tpms_jax');O=R/'validation/thickness_range_20261006_r13'
p=argparse.ArgumentParser();p.add_argument('tag',choices=['t0p45','t0p55']);a=p.parse_args()
prep=json.loads((O/'preparation.json').read_text())[a.tag];src=Path(prep['native_job_directory'])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(src/'input.json')==prep['input_sha256']
assert sha(src/'thin_shell.inp')==prep['deck_sha256']
assert sha(src/'extract_thin_explicit.py')==sha(R/'scripts/extract_thin_explicit.py')
assert 'THE ANALYSIS HAS COMPLETED SUCCESSFULLY' in (src/'thin_shell.sta').read_text()
assert (src/'shell.json').exists()
command=(src/'thin_shell.com').read_text()
assert "'cpus':4" in command and "'double_precision':BOTH" in command
dest=O/a.tag/'abaqus/explicit_T0p040';dest.mkdir(parents=True,exist_ok=False)
records={}
for f in sorted(src.iterdir()):
    if not f.is_file() or f.suffix not in ['.json','.npz','.inp','.py','.sta','.dat','.msg','.odb']:continue
    records[f.name]={'native_path':str(f),'bytes':f.stat().st_size,'sha256':sha(f)}
    if f.suffix!='.odb':shutil.copy2(f,dest/f.name)
sta=(src/'thin_shell.sta').read_text()
elapsed=re.findall(r'\b(\d\d:\d\d:\d\d)\b',sta)
receipt={'files':records,'native_directory':str(src),'ODB_retained_native':True,
         'command':'abaqus job=thin_shell input=thin_shell.inp cpus=4 double=both interactive',
         'last_reported_solver_elapsed_HHMMSS':elapsed[-1] if elapsed else None,
         'physical_change_only':'shell section thickness; unchanged nodes/elements, XYZ equations, NH, density, loading and hold'}
(dest/'retention.json').write_text(json.dumps(receipt,indent=2)+'\n')
j=json.loads((dest/'shell.json').read_text())
print(json.dumps({k:j[k] for k in ['status','checks','hold_mean_Fz_N','global_energy_drift_relative_work','loading_time_fraction_KE_below_5pct']},indent=2))

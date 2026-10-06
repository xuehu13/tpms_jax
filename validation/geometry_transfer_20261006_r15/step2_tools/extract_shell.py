"""Correct only the read-only CLI call; preserve the completed job and first log."""
from pathlib import Path
import hashlib,json,subprocess,time
D=Path(__file__).resolve().parent
P=Path(r'E:\ABAQUS\2026temp\Abaqus_Work\tpms_jax_abaqus\geometry_transfer_20261006_r15_diverse04_explicit_T0p040')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt=json.loads((P/'launch_receipt.json').read_text())
assert receipt['analysis_completed_successfully'] and receipt['returncode']==0
assert not receipt['shell_json_exists'] and not (P/'shell.json').exists()
assert 'the following arguments are required: out' in (P/'extraction.log').read_text()
assert 'THE ANALYSIS HAS COMPLETED SUCCESSFULLY' in (P/'thin_shell.sta').read_text()
before={n:sha(P/n) for n in ['input.json','thin_shell.inp','thin_shell.odb','extract_thin_explicit.py','launch_manifest.json','launch_receipt.json','extraction.log']}
command=['cmd','/d','/c',r'E:\ABAQUS\2026\Commands\abaqus.bat','python',
         'extract_thin_explicit.py','thin_shell.odb','input.json','shell.json']
assert not (P/'extraction_corrected.log').exists()
start=time.perf_counter()
with (P/'extraction_corrected.log').open('x',encoding='utf-8') as log:
    result=subprocess.run(command,cwd=P,stdout=log,stderr=subprocess.STDOUT)
assert all(sha(P/n)==h for n,h in before.items())
note={'returncode':result.returncode,'seconds':time.perf_counter()-start,'command':command,
 'preexisting_files_sha256':before,'preexisting_files_unchanged':True,'new_Abaqus_jobs':0,
 'original_extraction_failure':'Launcher omitted input.json; Abaqus wrapper returned 0 despite argparse error. Missing JSON and original log identify the failure.',
 'shell_json_exists':(P/'shell.json').exists(),'launcher_sha256':sha(Path(__file__))}
assert not (P/'extraction_corrected_receipt.json').exists()
(P/'extraction_corrected_receipt.json').write_text(json.dumps(note,indent=2)+'\n')
print(json.dumps(note));assert result.returncode==0 and (P/'shell.json').exists() and (P/'shell_field.npz').exists()

"""One Windows reference job and its extraction; source lives in formal WSL."""
from pathlib import Path
import json,time,subprocess,shutil,hashlib,sys
sys.stdout.reconfigure(encoding='utf-8')
D=Path(r'\\wsl.localhost\Ubuntu-24.04\home\xuehu\projects\tpms_jax\validation\plate_bending_20261008_r27')
P=Path(r'E:\ABAQUS\2026temp\Abaqus_Work\tpms_jax_abaqus\plate_bending_20261008_r27')
bat=Path(r'E:\ABAQUS\2026\Commands\abaqus.bat');P.mkdir(exist_ok=False)
for source in [D/'abaqus/plate.inp',D/'abaqus/input.json',D/'extract_shell.py']:shutil.copy2(source,P/source.name)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
before={n:sha(P/n) for n in ['plate.inp','input.json','extract_shell.py']}
command=[str(bat),'job=plate','input=plate.inp','cpus=1','interactive','ask_delete=OFF']
receipt={'native_directory':str(P),'command':command,'one_new_job':True,'input_sha256':before}
started=time.perf_counter()
with (P/'launch.log').open('xb') as log:
 r=subprocess.run(command,cwd=P,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW,timeout=900)
receipt['job_seconds']=time.perf_counter()-started;receipt['returncode']=r.returncode
sta=(P/'plate.sta').read_text(errors='replace') if (P/'plate.sta').exists() else ''
receipt['analysis_completed_successfully']='THE ANALYSIS HAS COMPLETED SUCCESSFULLY' in sta
if receipt['analysis_completed_successfully']:
 started=time.perf_counter()
 with (P/'extraction.log').open('xb') as log:
  x=subprocess.run([str(bat),'python','extract_shell.py'],cwd=P,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW,timeout=180)
 receipt['extract_seconds']=time.perf_counter()-started;receipt['extract_returncode']=x.returncode
 receipt['extract_artifact_exists']=(P/'shell.json').exists()
 if receipt['extract_artifact_exists']:
  parsed=json.loads((P/'shell.json').read_text())
  receipt['extract_artifact_valid']=bool(parsed.get('checks')) and isinstance(parsed.get('reference_gate_pass'),bool)
 if (P/'shell.json').exists():shutil.copy2(P/'shell.json',D/'abaqus/shell.json')
receipt['input_files_unchanged']=all(sha(P/n)==h for n,h in before.items())
for dest in [P/'receipt.json',D/'abaqus/receipt.json']:dest.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
sys.exit(0 if receipt['analysis_completed_successfully'] and receipt.get('extract_returncode')==0 and receipt.get('extract_artifact_valid',False) else 1)

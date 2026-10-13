"""Run one new small-strain shell Standard reference, preserving all earlier files."""
from pathlib import Path
import subprocess, shutil, json, time, hashlib
D=Path(r'\\wsl.localhost\Ubuntu-24.04\home\xuehu\projects\tpms_jax\validation\initial_tangent_20261008_r30')
P=Path(r'E:\ABAQUS\2026temp\Abaqus_Work\tpms_jax_abaqus\initial_tangent_20261008_r30_diverse04')
BAT=r'E:\ABAQUS\2026\Commands\abaqus.bat'
P.mkdir(exist_ok=False)
for name in ['shell.inp','shell_input.json','extract_shell.py']: shutil.copy2(D/'abaqus'/name,P/name)
receipt={'job_directory':str(P),'input_sha256':hashlib.sha256((P/'shell.inp').read_bytes()).hexdigest()}
for label,args,timeout in [('job',['job=shell','input=shell.inp','cpus=1','interactive','ask_delete=OFF'],900),
                         ('extract',['python','extract_shell.py','shell.odb','shell_input.json','shell_result.json'],120)]:
    start=time.perf_counter()
    with (P/(label+'.log')).open('w') as log:
        proc=subprocess.run([BAT,*args],cwd=P,stdout=log,stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW,timeout=timeout)
    receipt[label+'_seconds']=time.perf_counter()-start; receipt[label+'_returncode']=proc.returncode
    if proc.returncode: break
    if label=='job' and 'THE ANALYSIS HAS COMPLETED SUCCESSFULLY' not in (P/'shell.sta').read_text(): break
receipt['input_unchanged']=hashlib.sha256((P/'shell.inp').read_bytes()).hexdigest()==receipt['input_sha256']
for name in ['job.log','extract.log','shell_result.json','shell.sta','shell.msg','shell.dat']:
    if (P/name).exists(): shutil.copy2(P/name,D/'abaqus'/name)
(D/'abaqus'/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2),flush=True)

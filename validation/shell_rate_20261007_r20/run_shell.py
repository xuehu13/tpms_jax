"""Launch exactly one prepared shell-rate job, then reuse read-only extractors."""
from pathlib import Path
import hashlib,json,subprocess,time,sys,platform,shutil
sys.stdout.reconfigure(encoding='utf-8')
W=Path(__file__).resolve().parent
formal=Path(r'\\wsl.localhost\Ubuntu-24.04\home\xuehu\projects\tpms_jax\validation\shell_rate_20261007_r20\abaqus\explicit_T0p004')
old=Path(r'E:\ABAQUS\2026temp\Abaqus_Work\tpms_jax_abaqus\geometry_transfer_20261006_r15_diverse04_explicit_T0p040')
P=old.parent/'shell_rate_20261007_r20_diverse04_explicit_T0p004'
bat=Path(r'E:\ABAQUS\2026\Commands\abaqus.bat')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):
    assert not p.exists(),p
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

assert not P.exists();assert bat.exists()
oldcfg=json.loads((old/'input.json').read_text());cfg=json.loads((formal/'input.json').read_text())
assert cfg['shell_inp_sha256']==sha(formal/'thin_shell.inp')
assert cfg['extractor_sha256']==sha(formal/'extract_thin_explicit.py')==sha(old/'extract_thin_explicit.py')
oldfiles={n:sha(old/n) for n in ('thin_shell.inp','input.json','extract_thin_explicit.py','thin_shell.odb')}
P.mkdir()
for name in ('thin_shell.inp','input.json','extract_thin_explicit.py','extract_shell_modes.py'):
    shutil.copy2(formal/name,P/name)
before={p.name:sha(p) for p in P.iterdir()}
cmd=['cmd','/d','/c',str(bat),'job=thin_shell','input=thin_shell.inp','cpus=4','double=both','interactive']
extract=['cmd','/d','/c',str(bat),'python','extract_thin_explicit.py','thin_shell.odb','input.json','shell.json']
modes=['cmd','/d','/c',str(bat),'python','extract_shell_modes.py','thin_shell.odb','input.json','shell_frames']
write(P/'launch_manifest.json',{'plan_step':2,'user_authorization':'好的，开始按照规划进行吧',
    'job_command':cmd,'extraction_command':extract,'mode_extraction_command':modes,
    'working_directory':str(P),'input_sha256':before,'old_native_sha256':oldfiles,
    'launcher_sha256':sha(Path(__file__)),'platform':platform.platform(),
    'one_physical_factor':'load/hold times divided by 10','new_jobs':1})
start=time.perf_counter()
with (P/'launch.log').open('xb') as log:
    result=subprocess.run(cmd,cwd=P,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
wall=time.perf_counter()-start
complete=(P/'thin_shell.sta').exists() and 'THE ANALYSIS HAS COMPLETED SUCCESSFULLY' in (P/'thin_shell.sta').read_text(errors='replace')
codes={};times={}
if result.returncode==0 and complete:
    for label,command in (('extraction',extract),('mode_extraction',modes)):
        start=time.perf_counter()
        with (P/(label+'.log')).open('xb') as log:
            r=subprocess.run(command,cwd=P,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        codes[label]=r.returncode;times[label]=time.perf_counter()-start
unchanged=all(sha(P/name)==h for name,h in before.items())
old_unchanged=all(sha(old/name)==h for name,h in oldfiles.items())
receipt={'returncode':result.returncode,'analysis_completed_successfully':complete,
    'wall_seconds':wall,'postprocess_returncodes':codes,'postprocess_seconds':times,
    'input_package_unchanged':unchanged,'old_native_files_unchanged':old_unchanged,
    'shell_json_exists':(P/'shell.json').exists(),'shell_field_exists':(P/'shell_field.npz').exists(),
    'shell_modes_exists':(P/'shell_frames/modes.json').exists(),'native_directory':str(P),
    'ODB_retained_native':True,'new_JAX_forward':False,'design_AD':False}
write(P/'launch_receipt.json',receipt);write(W/'launch_receipt.json',receipt)
print(json.dumps(receipt,ensure_ascii=False));assert unchanged and old_unchanged
sys.exit(0 if complete and result.returncode==0 and all(c==0 for c in codes.values()) and receipt['shell_json_exists'] and receipt['shell_modes_exists'] else 1)

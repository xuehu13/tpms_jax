"""One prepared Abaqus Explicit shell job plus read-only result extraction."""
from pathlib import Path
import hashlib,json,subprocess,sys,time
D=Path(__file__).resolve().parent
P=Path(r'E:\ABAQUS\2026temp\Abaqus_Work\tpms_jax_abaqus\geometry_transfer_20261006_r15_diverse04_explicit_T0p040')
bat=Path(r'E:\ABAQUS\2026\Commands\abaqus.bat')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    assert not p.exists(),p
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
cfg=json.loads((P/'input.json').read_text())
assert sorted(p.name for p in P.iterdir())==['extract_thin_explicit.py','input.json','thin_shell.inp']
assert sha(P/'thin_shell.inp')==cfg['shell_inp_sha256']
assert sha(P/'extract_thin_explicit.py')==cfg['extractor_sha256']
before={p.name:sha(p) for p in P.iterdir()}
job=['cmd','/d','/c',str(bat),'job=thin_shell','input=thin_shell.inp','cpus=4','double=both','interactive']
extract=['cmd','/d','/c',str(bat),'python','extract_thin_explicit.py','thin_shell.odb','input.json','shell.json']
write(P/'launch_manifest.json',{'main_plan_step':2,'job_command':job,'extraction_command':extract,
 'working_directory':str(P),'input_sha256':before,'launcher_sha256':sha(Path(__file__)),
 'research_question':'One matched diverse_04 0.50mm NH XYZ shell reference to 20% plus hold',
 'no_contact_no_plasticity_no_mass_scaling':True})
start=time.perf_counter()
with (P/'launch.log').open('x',encoding='utf-8') as log:
    result=subprocess.run(job,cwd=P,stdout=log,stderr=subprocess.STDOUT)
wall=time.perf_counter()-start
complete=(P/'thin_shell.sta').exists() and 'THE ANALYSIS HAS COMPLETED SUCCESSFULLY' in (P/'thin_shell.sta').read_text(errors='replace')
extraction_code=None;extract_seconds=None
if result.returncode==0 and complete:
    start=time.perf_counter()
    with (P/'extraction.log').open('x',encoding='utf-8') as log:
        extraction=subprocess.run(extract,cwd=P,stdout=log,stderr=subprocess.STDOUT)
    extraction_code=extraction.returncode;extract_seconds=time.perf_counter()-start
unchanged=all(sha(P/p)==h for p,h in before.items())
receipt={'returncode':result.returncode,'analysis_completed_successfully':complete,
 'wall_seconds':wall,'extraction_returncode':extraction_code,'extraction_seconds':extract_seconds,
 'original_package_unchanged':unchanged,'shell_json_exists':(P/'shell.json').exists(),
 'native_directory':str(P),'ODB_retained_native':True}
write(P/'launch_receipt.json',receipt);write(D/'shell_launch_receipt.json',receipt)
print(json.dumps(receipt,ensure_ascii=False));assert unchanged
sys.exit(0 if complete and result.returncode==0 and extraction_code==0 and (P/'shell.json').exists() else 1)

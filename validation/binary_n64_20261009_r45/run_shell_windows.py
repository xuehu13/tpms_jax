"""One same-rate shell reference; preserve the existing mesh and physical input."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys,time
sys.stdout.reconfigure(encoding='utf-8')
T=float(sys.argv[1]); assert T in (.001,.0005,.00025)
R=Path(r'\\wsl.localhost\Ubuntu-24.04\home\xuehu\projects\tpms_jax')
D=R/'validation/binary_n64_20261009_r45'
S=R/'validation/shell_rate_20261007_r20/abaqus/explicit_T0p004'
tag=format(T,'.6f').replace('.','p')
B=D/('abaqus/explicit_T'+tag)
native_root=Path(r'E:\ABAQUS\2026temp\Abaqus_Work\tpms_jax_abaqus')
P=native_root/('binary_n64_20261009_r45_shell_T'+tag)
bat=Path(r'E:\ABAQUS\2026\Commands\abaqus.bat')
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
protocol=json.loads((D/'protocol.json').read_text());deadline=protocol['started_unix']+7200-120
assert not P.exists() and not B.exists() and P.resolve().is_relative_to(native_root.resolve())
assert protocol['load_time_seconds'] is None and time.time()<deadline
text=(S/'thin_shell.inp').read_text()
changes={'0., 0., 0.004, 1., 0.0044, 1.':f'0., 0., {T:.9g}, 1., {1.1*T:.9g}, 1.',
         ', 0.0044':f', {1.1*T:.9g}',
         '*Output, history, time interval=4.400000000000001e-06':f'*Output, history, time interval={1.1*T/1000:.17g}'}
for old,new in changes.items():
    assert text.count(old)==1,old;text=text.replace(old,new)
B.mkdir(parents=True);(B/'thin_shell.inp').write_text(text)
shutil.copy2(S/'extract_thin_explicit.py',B/'extract_thin_explicit.py')
cfg=json.loads((S/'input.json').read_text());cfg.update(load_time_seconds=T,hold_time_seconds=.1*T,total_time_seconds=1.1*T,shell_inp_sha256=sha(B/'thin_shell.inp'),timing_only_change_from=str(S),round='r45')
write(B/'input.json',cfg);P.mkdir()
for name in ('thin_shell.inp','input.json','extract_thin_explicit.py'):shutil.copy2(B/name,P/name)
initial={n:sha(P/n) for n in ('thin_shell.inp','input.json','extract_thin_explicit.py')}
job=['cmd','/d','/c',str(bat),'job=thin_shell','input=thin_shell.inp','cpus=4','double=both','interactive']
extract=['cmd','/d','/c',str(bat),'python','extract_thin_explicit.py','thin_shell.odb','input.json','shell.json']
write(P/'launch_manifest.json',dict(job_command=job,extraction_command=extract,input_sha256=initial,physical_change='Three timing/output lines only; same S3R geometry/PBC/NH/density/default bulk viscosity; no new contact or mass scaling.'))
start=time.time();receipt=dict(started_unix=start,native_directory=str(P),load_time_seconds=T,new_shell_jobs=1)
try:
    with (P/'launch.log').open('xb') as log:
        p=subprocess.Popen(job,cwd=P,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        try:code=p.wait(timeout=min(600,deadline-time.time()))
        except subprocess.TimeoutExpired:
            subprocess.run(['cmd','/d','/c',str(bat),'job=thin_shell','terminate'],cwd=P,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW,timeout=30)
            raise TimeoutError('Matched shell resource stop; no retry')
    complete='THE ANALYSIS HAS COMPLETED SUCCESSFULLY' in (P/'thin_shell.sta').read_text(errors='replace')
    assert code==0 and complete
    with (P/'extraction.log').open('xb') as log:
        post=subprocess.run(extract,cwd=P,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW,timeout=min(120,deadline-time.time()))
    assert post.returncode==0
    shell=json.loads((P/'shell.json').read_text());assert all(shell['checks'][n] for n in ('complete','finite','target_compression'))
    assert all(sha(P/n)==h for n,h in initial.items())
    results=B/'results';results.mkdir()
    for n in ('shell.json','shell_field.npz','launch.log','extraction.log','launch_manifest.json','thin_shell.sta','thin_shell.msg','thin_shell.dat'):
        if (P/n).exists():shutil.copy2(P/n,results/n)
    peak=max((r for r in shell['force_path'] if r['time']<=T and r['compression']>=.01),key=lambda r:-r['Fz_N'])
    shutil.copy2(D/'protocol.json',D/'protocol_before_selection.json')
    protocol.update(load_time_seconds=T,matched_peak_N=-peak['Fz_N'],matched_shell_peak=peak,matched_shell_json=str(results/'shell.json'),matched_shell_sha256=sha(results/'shell.json'),selected_unix=time.time())
    write(D/'protocol.json',protocol)
    receipt.update(returncode=0,analysis_completed_successfully=complete,quality_checks=shell['checks'],matched_peak=peak,input_unchanged=True,ODB_retained_native=str(P/'thin_shell.odb'))
except BaseException as exc:
    receipt.update(returncode=1,exception_type=type(exc).__name__,exception_message=str(exc));raise
finally:
    receipt['wall_seconds']=time.time()-start;write(D/'shell_receipt.json',receipt);write(P/'launch_receipt.json',receipt);print(json.dumps(receipt,ensure_ascii=False),flush=True)

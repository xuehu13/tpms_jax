"""One matched shell and one N64 attempt, total deadline 3h; no retries."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys,time
sys.stdout.reconfigure(encoding='utf-8')
R=Path(r'\\wsl.localhost\Ubuntu-24.04\home\xuehu\projects\tpms_jax')
D=R/'validation/n64_resolution_20261008_r28'
package=D/'abaqus/explicit_T0p002'
native_root=Path(r'E:\ABAQUS\2026temp\Abaqus_Work\tpms_jax_abaqus')
P=native_root/'n64_resolution_20261008_r28_shell_T0p002'
bat=Path(r'E:\ABAQUS\2026\Commands\abaqus.bat')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,value):p.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
assert P.resolve().is_relative_to(native_root.resolve()) and not P.exists()
assert bat.exists() and json.loads((D/'launch_decision.json').read_text())['full_attempt_authorized']
cfg=json.loads((package/'input.json').read_text());assert sha(package/'thin_shell.inp')==cfg['shell_inp_sha256']
assert sha(package/'extract_thin_explicit.py')==cfg['extractor_sha256']
P.mkdir()
for name in ('thin_shell.inp','input.json','extract_thin_explicit.py','extract_shell_modes.py'):
    shutil.copy2(package/name,P/name)
initial={name:sha(P/name) for name in ('thin_shell.inp','input.json','extract_thin_explicit.py','extract_shell_modes.py')}
start=time.time();deadline=start+10800.
decision=json.loads((D/'launch_decision.json').read_text());decision.update(started_unix=start,deadline_unix=deadline)
write(D/'launch_decision.json',decision)
receipt={'started_unix':start,'deadline_unix':deadline,'total_budget_seconds':10800.,
    'native_directory':str(P),'new_shell_jobs':1,'new_JAX_from_zero_attempts':0,'no_design_AD_or_training':True}
write(D/'pipeline_progress.json',dict(stage='matched_shell_running',**receipt))
job=['cmd','/d','/c',str(bat),'job=thin_shell','input=thin_shell.inp','cpus=4','double=both','interactive']
extract=['cmd','/d','/c',str(bat),'python','extract_thin_explicit.py','thin_shell.odb','input.json','shell.json']
modes=['cmd','/d','/c',str(bat),'python','extract_shell_modes.py','thin_shell.odb','input.json','shell_frames']
write(P/'launch_manifest.json',{'job_command':job,'extraction_command':extract,'mode_command':modes,
    'input_sha256':initial,'user_authorization':'<=3h; may shorten loading, agent judges',
    'physical_change_from_r20':'duration halved, same mesh/material/PBC; 3 time/output lines only'})
code=1
try:
    t=time.perf_counter()
    with (P/'launch.log').open('xb') as log:
        process=subprocess.Popen(job,cwd=P,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        try:jobcode=process.wait(timeout=min(1200.,deadline-time.time()-120.))
        except subprocess.TimeoutExpired:
            subprocess.run(['cmd','/d','/c',str(bat),'job=thin_shell','terminate'],cwd=P,
                stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW,timeout=60)
            raise TimeoutError('Matched shell reference resource budget stop; no automatic retry')
    complete=(P/'thin_shell.sta').exists() and 'THE ANALYSIS HAS COMPLETED SUCCESSFULLY' in (P/'thin_shell.sta').read_text(errors='replace')
    shellreceipt={'returncode':jobcode,'analysis_completed_successfully':complete,'wall_seconds':time.perf_counter()-t}
    if jobcode or not complete:raise RuntimeError('Matched shell did not complete; full launch withheld')
    extraction_codes={}
    for name,command in [('extraction',extract),('mode_extraction',modes)]:
        with (P/(name+'.log')).open('xb') as log:
            result=subprocess.run(command,cwd=P,stdout=log,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW,timeout=min(180.,deadline-time.time()-120.))
        extraction_codes[name]=result.returncode
    assert all(x==0 for x in extraction_codes.values())
    shell=json.loads((P/'shell.json').read_text())
    assert shell['checks']['complete'] and shell['checks']['finite'] and shell['checks']['target_compression']
    assert all(sha(P/name)==digest for name,digest in initial.items())
    shellreceipt.update(postprocess_returncodes=extraction_codes,input_unchanged=True,
        quality_checks=shell['checks'],ODB_retained_native=str(P/'thin_shell.odb'))
    write(P/'launch_receipt.json',shellreceipt);write(D/'shell_receipt.json',shellreceipt)
    results=package/'results';results.mkdir()
    for name in ('shell.json','shell_field.npz','launch_manifest.json','launch_receipt.json','launch.log',
                 'extraction.log','mode_extraction.log','thin_shell.sta','thin_shell.msg','thin_shell.dat'):
        if (P/name).exists():shutil.copy2(P/name,results/name)
    if (P/'shell_frames').exists():shutil.copytree(P/'shell_frames',results/'shell_frames')
    loading=[r for r in shell['force_path'] if r['time']<=.002 and r['compression']>=.01]
    peak=max(loading,key=lambda r:-r['Fz_N'])
    protocol=json.loads((D/'protocol.json').read_text())
    protocol['denominators_N']['matched_shell_peak']=-peak['Fz_N']
    protocol['reference_raw_sha256']=sha(results/'shell.json')
    write(D/'protocol.json',protocol)
    decision=json.loads((D/'launch_decision.json').read_text());decision.update(
        matched_shell_ready=True,matched_shell_peak=peak,matched_shell_sha256=sha(results/'shell.json'))
    write(D/'launch_decision.json',decision)
    print('MATCHED_SHELL_READY '+json.dumps(shellreceipt,ensure_ascii=False),flush=True)
    receipt['new_JAX_from_zero_attempts']=1
    write(D/'pipeline_progress.json',dict(stage='N64_from_zero_running',**receipt))
    command=['wsl','-d','Ubuntu-24.04','--','bash','-lc',
        'cd /home/xuehu/projects/tpms_jax && .pixi/envs/default/bin/python validation/n64_resolution_20261008_r28/run.py full --budget-seconds 10800']
    with (D/'full.log').open('xb') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        receipt['launcher_pid']=process.pid
        write(D/'pipeline_progress.json',dict(stage='N64_from_zero_running',**receipt))
        # Internal controller stops with 120s reserve. External wait limits
        # unexpected stalls; a watchdog uses SIGINT on this exact recorded PID.
        try:fullcode=process.wait(timeout=max(1.,deadline-time.time()-60.))
        except subprocess.TimeoutExpired:
            launch=json.loads((D/'full_launch.json').read_text())
            subprocess.run(['wsl','-d','Ubuntu-24.04','--','kill','-INT',str(launch['pid'])],
                capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW,timeout=10)
            try:fullcode=process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                subprocess.run(['wsl','-d','Ubuntu-24.04','--','kill','-TERM',str(launch['pid'])],
                    capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW,timeout=10)
                fullcode=124
    receipt['full_returncode']=fullcode
    write(D/'pipeline_progress.json',dict(stage='extracting_completed_or_partial_result',**receipt))
    command=['wsl','-d','Ubuntu-24.04','--','bash','-lc',
        'cd /home/xuehu/projects/tpms_jax && .pixi/envs/default/bin/python validation/n64_resolution_20261008_r28/analyze.py']
    with (D/'analysis.log').open('xb') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW,timeout=max(1.,deadline-time.time()))
    receipt['analysis_returncode']=result.returncode
    receipt['full_complete']=(D/'full_from_zero/result.json').exists()
    receipt['comparison_exists']=(D/'comparison.json').exists()
    code=0 if result.returncode==0 and receipt['comparison_exists'] else 1
except Exception as exc:
    receipt.update(exception_type=type(exc).__name__,exception_message=str(exc))
finally:
    receipt.update(returncode=code,total_wall_seconds=time.time()-start,
        deadline_overrun_seconds=max(0.,time.time()-deadline))
    write(D/'pipeline_receipt.json',receipt)
    write(D/'pipeline_progress.json',dict(stage='finished',**receipt))
    print('R28_PIPELINE_RECEIPT '+json.dumps(receipt,ensure_ascii=False),flush=True)
sys.exit(code)

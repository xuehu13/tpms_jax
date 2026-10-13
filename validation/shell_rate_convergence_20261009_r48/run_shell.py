"""Submit only the two frozen r48 shell cases; reuse original extractors."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib, json, shutil, subprocess, sys, time, platform

sys.stdout.reconfigure(encoding='utf-8')
W = Path(__file__).resolve().parent
R = Path(r'\\wsl.localhost\Ubuntu-24.04\home\xuehu\projects\tpms_jax')
D = R/'validation/shell_rate_convergence_20261009_r48'
NATIVE = Path(r'E:\ABAQUS\2026temp\Abaqus_Work\tpms_jax_abaqus')
BAT = Path(r'E:\ABAQUS\2026\Commands\abaqus.bat')

def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def write(p, data):
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def status(p):
    if not p.exists():
        return 'preparing / license / packaging'
    lines=p.read_text(errors='replace').splitlines()
    data=[s.strip() for s in lines if s.strip() and s.strip()[0].isdigit()]
    return data[-1] if data else (lines[-1].strip() if lines else 'initializing')

def main():
    protocol=json.loads((D/'protocol.json').read_text(encoding='utf-8'))
    assert [c['T_seconds'] for c in protocol['cases']]==[.02,.08]
    assert protocol['new_shell_jobs']==2 and BAT.exists()
    assert not (D/'batch_launch.json').exists(), 'Already submitted: inspect receipts instead of rerunning'
    for case in protocol['cases']:
        p=NATIVE/('shell_rate_convergence_20261009_r48_T'+case['tag'])
        assert not p.exists() and p.resolve().is_relative_to(NATIVE.resolve())
    start=time.time();deadline=start+protocol['global_wall_budget_seconds']
    write(D/'batch_launch.json',{'started_unix':start,'deadline_unix':deadline,
        'new_shell_jobs':2,'launcher_sha256':sha(Path(__file__)),
        'platform':platform.platform(),'parallel_jobs':2,'CPUs_per_job':4,
        'authorization':protocol['user_authorization']})

    def run(case):
        T=case['T_seconds'];tag=case['tag']
        B=D/'abaqus'/('explicit_T'+tag)
        P=NATIVE/('shell_rate_convergence_20261009_r48_T'+tag)
        P.mkdir();initial=case['input_sha256']
        receipt={'T_seconds':T,'native_directory':str(P),'started_unix':time.time(),
                 'new_shell_jobs':1,'new_JAX_forward':False,'AD':False}
        for name,h in initial.items():
            assert sha(B/name)==h
            shutil.copy2(B/name,P/name)
        job=['cmd','/d','/c',str(BAT),'job=thin_shell','input=thin_shell.inp',
             'cpus=4','double=both','interactive']
        extract=['cmd','/d','/c',str(BAT),'python','extract_thin_explicit.py',
                 'thin_shell.odb','input.json','shell.json']
        modes=['cmd','/d','/c',str(BAT),'python','extract_shell_modes.py',
               'thin_shell.odb','input.json','shell_frames']
        write(P/'launch_manifest.json',{'job_command':job,'extraction_command':extract,
              'mode_extraction_command':modes,'input_sha256':initial,
              'formal_protocol':str(D/'protocol.json'),'reverse_byte_check':case['reverse_byte_check'],
              'physical_change':'Load/hold times and proportional history output interval only'})
        solver_start=time.perf_counter()
        try:
            with (P/'launch.log').open('xb') as log:
                proc=subprocess.Popen(job,cwd=P,stdout=log,stderr=subprocess.STDOUT,
                                      creationflags=subprocess.CREATE_NO_WINDOW)
                receipt['solver_command_pid']=proc.pid
                own_deadline=min(deadline-90,time.time()+case['maximum_solver_wall_seconds'])
                while True:
                    remaining=own_deadline-time.time()
                    if remaining<=0:
                        subprocess.run(['cmd','/d','/c',str(BAT),'job=thin_shell','terminate'],
                            cwd=P,stdout=log,stderr=subprocess.STDOUT,timeout=45,
                            creationflags=subprocess.CREATE_NO_WINDOW)
                        raise TimeoutError('Frozen wall budget reached; no retry or extension')
                    try:
                        code=proc.wait(timeout=min(30,remaining));break
                    except subprocess.TimeoutExpired:
                        progress={'T_seconds':T,'wall_seconds':time.perf_counter()-solver_start,
                                  'sta_last_record':status(P/'thin_shell.sta')}
                        write(B/'progress.json',progress)
                        print(json.dumps(progress,ensure_ascii=False),flush=True)
            receipt['solver_wall_seconds']=time.perf_counter()-solver_start
            receipt['solver_returncode']=code
            complete=(P/'thin_shell.sta').exists() and 'THE ANALYSIS HAS COMPLETED SUCCESSFULLY' in (P/'thin_shell.sta').read_text(errors='replace')
            receipt['analysis_completed_successfully']=complete
            assert code==0 and complete, 'Abaqus analysis did not complete'
            for label,command in [('extraction',extract),('mode_extraction',modes)]:
                t=time.perf_counter()
                with (P/(label+'.log')).open('xb') as log:
                    p=subprocess.run(command,cwd=P,stdout=log,stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW,timeout=max(1,min(120,deadline-time.time())))
                receipt[label+'_returncode']=p.returncode
                receipt[label+'_wall_seconds']=time.perf_counter()-t
                assert p.returncode==0, label+' failed; preserve outputs'
            shell=json.loads((P/'shell.json').read_text())
            receipt['quality_checks']=shell['checks']
            core=('complete','finite','target_compression','physical_node_count',
                  'XYZ_periodic_mm_le_1e-6','XYZ_rotation_rad_le_1e-6')
            receipt['core_valid']=all(shell['checks'][k] for k in core)
            receipt['returncode']=0 if receipt['core_valid'] else 1
            receipt['ODB_retained_native']=str(P/'thin_shell.odb')
        except BaseException as exc:
            receipt.update(returncode=1,exception_type=type(exc).__name__,exception_message=str(exc))
        finally:
            receipt['input_package_unchanged']=all(sha(P/n)==h for n,h in initial.items())
            receipt['wall_seconds']=time.time()-receipt['started_unix']
            write(P/'launch_receipt.json',receipt);write(B/'launch_receipt.json',receipt)
            results=B/'results';results.mkdir(exist_ok=True)
            for name in ['shell.json','shell_field.npz','thin_shell.sta','thin_shell.msg',
                         'thin_shell.dat','launch.log','extraction.log','mode_extraction.log',
                         'launch_manifest.json','launch_receipt.json']:
                if (P/name).exists():shutil.copy2(P/name,results/name)
            if (P/'shell_frames').exists():shutil.copytree(P/'shell_frames',results/'shell_frames',dirs_exist_ok=True)
            print(json.dumps(receipt,ensure_ascii=False),flush=True)
        return receipt

    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts=list(pool.map(run,protocol['cases']))
    batch={'new_shell_jobs':2,'new_JAX_forward':False,'AD':False,'cases':receipts,
           'UTC_wall_seconds':time.time()-start,'within_global_budget':time.time()<=deadline,
           'all_completed':all(r.get('analysis_completed_successfully',False) for r in receipts),
           'all_core_valid':all(r.get('core_valid',False) for r in receipts),
           'no_automatic_more_jobs':True}
    write(D/'batch_receipt.json',batch);write(W/'batch_receipt.json',batch)
    print(json.dumps(batch,ensure_ascii=False),flush=True)
    return 0 if batch['all_completed'] and batch['all_core_valid'] else 1

if __name__=='__main__':
    sys.exit(main())

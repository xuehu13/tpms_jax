"""Serial, finite r50 batch. Six physical dynamic cases; explicit mesh-preprocessing restart; no AD."""
from pathlib import Path
import subprocess, time, json, shutil, hashlib, sys

FORMAL='/home/xuehu/projects/tpms_jax'
LINUX=FORMAL+'/validation/canonical_tpms_20261009_r50'
D=Path(r'\\wsl.localhost\Ubuntu-24.04\home\xuehu\projects\tpms_jax\validation\canonical_tpms_20261009_r50')
NATIVE=Path(r'E:\ABAQUS\2026temp\Abaqus_Work\tpms_jax_abaqus')
BAT=r'E:\ABAQUS\2026\Commands\abaqus.bat'
PYTHON=FORMAL+'/.pixi/envs/default/bin/python'
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def wait_job(command,cwd,log_path,seconds,label,progress_path=None):
    start=time.monotonic();budget=False
    with log_path.open('xb') as log:
        proc=subprocess.Popen(command,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW)
        while proc.poll() is None:
            try:proc.wait(timeout=max(.1,min(30,seconds-(time.monotonic()-start))))
            except subprocess.TimeoutExpired:pass
            update={'task':label,'wall_seconds':time.monotonic()-start,'pid':proc.pid}
            if progress_path and progress_path.exists():
                try:update['progress']=read(progress_path)
                except (ValueError,OSError):pass
            write(D/'progress.json',update);print(json.dumps(update),flush=True)
            if time.monotonic()-start>=seconds and proc.poll() is None:
                budget=True
                if label.endswith('shell'):
                    subprocess.run(['cmd','/d','/c',BAT,'job=shell','terminate'],cwd=cwd,
                        stdout=log,stderr=subprocess.STDOUT,timeout=25,creationflags=subprocess.CREATE_NO_WINDOW)
                    try:proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:proc.terminate()
                else:
                    # Linux timeout owns the child process group and is already
                    # set to interrupt/kill within the requested wall cap.
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:proc.terminate()
                break
    return {'command':command,'returncode':proc.poll(),'budget_stop':budget,
            'wall_seconds':time.monotonic()-start,'log':str(log_path)}

def jax(kind,initial):
    c=D/kind;label=kind+('_initial_jax' if initial else '_dynamic_jax')
    if initial:
        args=[PYTHON,LINUX+'/initial.py',LINUX+'/'+kind];cap=300;limit='280s'
    else:
        args=[PYTHON,FORMAL+'/scripts/thin_target_explicit.py','target',
            '--case-input',LINUX+'/'+kind+'/input.json','--gauss-field',LINUX+'/'+kind+'/gauss_field.npz',
            '--material-model','objective_void','--element-degree','2','--cells','32','--quadrature-order','4',
            '--geometry-on-cpu','--force-batch-cells','2048','--load-time','.004',
            '--adaptive',
            '--checkpoint-compressions','.05','.10','.15','.20','--output',LINUX+'/'+kind+'/jax']
        cap=2400;limit='2380s'
    command=['wsl.exe','-d','Ubuntu-24.04','--exec','/usr/bin/timeout','--signal=INT','--kill-after=15s',limit,
        '/usr/bin/env','JAX_PLATFORMS=cuda,cpu','XLA_PYTHON_CLIENT_PREALLOCATE=false','OMP_NUM_THREADS=1',*args]
    print('START '+label,flush=True)
    receipt=wait_job(command,None,c/(label+'.log'),cap,label,None if initial else c/'jax/progress.json')
    receipt['budget_stop']=receipt['budget_stop'] or receipt['returncode'] in (124,137)
    write(c/(label+'_receipt.json'),receipt);return receipt

def shell(kind,initial):
    c=D/kind;source=c/('initial_shell' if initial else 'abaqus')
    quality=(c/'shell_quality_adopted.json').exists()
    native=NATIVE/('canonical_tpms_20261009_r50_'+kind+('_initial' if initial else '_dynamic')+('_quality' if quality else ''))
    native.mkdir(exist_ok=False)
    for name in ['shell.inp','input.json','extract.py']:shutil.copy2(source/name,native/name)
    original={name:sha(native/name) for name in ['shell.inp','input.json','extract.py']}
    label=kind+('_initial_shell' if initial else '_dynamic_shell')
    command=['cmd','/d','/c',BAT,'job=shell','input=shell.inp',
        'cpus=1' if initial else 'cpus=4','interactive','ask_delete=OFF']
    if not initial:command.insert(-2,'double=both')
    print('START '+label,flush=True)
    cap=270 if initial else (1080 if kind=='G' else 1400)
    receipt=wait_job(command,native,native/'launch.log',cap,label)
    receipt['current_launch_wall_seconds']=receipt['wall_seconds']
    receipt['prior_aborted_launch_budget_reservation_seconds']=320 if not initial and kind=='G' else 0
    receipt['wall_seconds']+=receipt['prior_aborted_launch_budget_reservation_seconds']
    sta=native/'shell.sta'
    receipt['analysis_completed_successfully']=sta.exists() and 'THE ANALYSIS HAS COMPLETED SUCCESSFULLY' in sta.read_text(errors='replace')
    if (native/'shell.odb').exists() and not (native/'shell.lck').exists():
        with (native/'extraction.log').open('xb') as log:
            t=time.monotonic()
            try:
                cp=subprocess.run(['cmd','/d','/c',BAT,'python','extract.py','shell.odb','input.json','shell_result.json'],
                    cwd=native,stdout=log,stderr=subprocess.STDOUT,timeout=45,creationflags=subprocess.CREATE_NO_WINDOW)
                receipt['extraction_returncode']=cp.returncode
            except subprocess.TimeoutExpired:receipt['extraction_timeout']=True
            receipt['extraction_seconds']=time.monotonic()-t
    receipt['native_directory']=str(native);receipt['input_unchanged']=all(sha(native/n)==h for n,h in original.items())
    for name in ['shell_result.json','shell_field.npz','shell.sta','shell.msg','shell.dat','launch.log','extraction.log']:
        if (native/name).exists():shutil.copy2(native/name,source/name)
    write(source/'receipt.json',receipt);write(native/'receipt.json',receipt);return receipt

def main():
    assert not (D/'batch_receipt.json').exists(),'Completed batch cannot be repeated'
    protocol={'cases':['G','P','D'],'new_dynamic_tasks':6,'initial_checks':6,
        'jax_wall_cap_seconds':2400,'abaqus_wall_cap_seconds':1500,
        'Abaqus_solver_budget_seconds':{'G_remaining':1080,'P':1400,'D':1400},'G_prior_aborted_launch_reserved_seconds':320,'initial_task_cap_seconds':300,
        'load_time_seconds':.004,'hold_time_seconds':.0004,'N':32,'element':'HEX27','Gauss_points':27,
        'material':'objective_void C2','thickness_mm':.5,'eta':1e-4,'interface_10_90_mm':.05,
        'no_AD':True,'no_numerical_parameter_retries':True,'mesh_preprocessing_restart_G':True,'mesh_quality_reconnected':['G','P'],'D_original_mesh_retained':True,'user_authorized':True,
        'time_control':'r12 validated adaptive invalid-block rollback, minimum initial dt/16; no F/J clipping',
        'prelaunch_choice':'State-step-control draft replaced before any dynamic run: its historical high-KE hard stop conflicts with allowed jump dynamics. Original production entry unchanged; energy/KE quality flags reported separately.'}
    write(D/'protocol.json',protocol);start=time.time();receipts=[];admitted={}
    for kind in protocol['cases']:
        assert read(D/kind/'geometry_audit.json')['pass']
        # Completed initial results are reused after a pre-solver environment
        # launch correction; dynamic attempts are never repeated.
        jr=read(D/kind/(kind+'_initial_jax_receipt.json')) if (D/kind/'jax_result.json').exists() else jax(kind,True)
        sr=read(D/kind/'initial_shell/receipt.json') if (D/kind/'initial_shell/shell_result.json').exists() else shell(kind,True)
        receipts += [{'task':kind+'_initial_jax',**jr}, {'task':kind+'_initial_shell',**sr}]
        initial=D/kind/'jax_result.json';sh=D/kind/'initial_shell/shell_result.json'
        admitted[kind]=initial.exists() and read(initial)['status']=='ok' and sh.exists()
        if sh.exists():
            checks=read(sh)['checks']
            admitted[kind]=admitted[kind] and all(checks.get(n,False) for n in ['complete_step','node_count','finite','load_delta','periodic_translation_gap_le_1e-8_mm','periodic_rotation_gap_le_1e-8_rad'])
        write(D/'batch_partial.json',{'receipts':receipts,'admitted':admitted})
    for kind in protocol['cases']:
        if not admitted[kind]:
            print('INPUT_NOT_ADMITTED '+kind,flush=True);continue
        jpath=D/kind/(kind+'_dynamic_jax_receipt.json')
        jr=read(jpath) if jpath.exists() else jax(kind,False)
        spath=D/kind/'abaqus/receipt.json'
        sr=read(spath) if spath.exists() else shell(kind,False)
        receipts += [{'task':kind+'_dynamic_jax',**jr}, {'task':kind+'_dynamic_shell',**sr}]
        write(D/'batch_partial.json',{'receipts':receipts,'admitted':admitted})
    result={'receipts':receipts,'admitted':admitted,'UTC_wall_seconds':time.time()-start,
        'dynamic_tasks_attempted':sum('_dynamic_' in x['task'] for x in receipts),'AD_jobs':0,'automatic_numerical_retries':0,'G_mesh_preprocessing_dynamic_restart':1,'prior_G_shell_cost_reserved_seconds':320}
    write(D/'batch_receipt.json',result);print('BATCH_DONE '+json.dumps(result),flush=True)

if __name__=='__main__':main()

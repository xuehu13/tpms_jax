"""Read the budget-stopped r48 T80 ODB; never submit or resume a job."""
from pathlib import Path
import json, shutil, subprocess, time

R=Path(r'\\wsl.localhost\Ubuntu-24.04\home\xuehu\projects\tpms_jax')
D=R/'validation/shell_rate_convergence_20261009_r48'
P=Path(r'E:\ABAQUS\2026temp\Abaqus_Work\tpms_jax_abaqus\shell_rate_convergence_20261009_r48_T0p080000')
BAT=Path(r'E:\ABAQUS\2026\Commands\abaqus.bat')

def main():
    start=time.time()
    batch=json.loads((D/'batch_receipt.json').read_text(encoding='utf-8'))
    case=next(c for c in batch['cases'] if c['T_seconds']==.08)
    assert not case.get('analysis_completed_successfully',False)
    assert case.get('exception_type')=='TimeoutError',case
    assert (P/'thin_shell.odb').exists()
    assert not (P/'partial_shell.json').exists(),'Already extracted; inspect existing receipt'
    stop=time.monotonic()+45
    while (P/'thin_shell.lck').exists() and time.monotonic()<stop:
        time.sleep(1)
    assert not (P/'thin_shell.lck').exists(),'ODB remains locked; no forced read or resume'
    shutil.copy2(D/'extract_partial.py',P/'extract_partial.py')
    commands=[('partial_extraction',['extract_partial.py','thin_shell.odb','input.json','partial_shell.json']),
              ('partial_modes',['extract_shell_modes.py','thin_shell.odb','input.json','shell_frames'])]
    receipt={'new_shell_jobs':0,'new_JAX_forward':0,'AD':0,'read_only_ODB':str(P/'thin_shell.odb'),'commands':[]}
    for label,args in commands:
        t=time.perf_counter()
        cmd=['cmd','/d','/c',str(BAT),'python',*args]
        with (P/(label+'.log')).open('xb') as log:
            p=subprocess.run(cmd,cwd=P,stdout=log,stderr=subprocess.STDOUT,timeout=120,
                             creationflags=subprocess.CREATE_NO_WINDOW)
        receipt['commands'].append({'command':cmd,'returncode':p.returncode,'wall_seconds':time.perf_counter()-t})
        assert p.returncode==0,(label,p.returncode)
    j=json.loads((P/'partial_shell.json').read_text(encoding='utf-8'))
    receipt.update(history_compression=j['compression'],history_time=j['force_path'][-1]['time'],
        field_compression=j['field_compression'],field_time=j['field_time'],checks=j['checks'],
        wall_seconds=time.time()-start,scope='Budget-stopped actual history; no full loading/hold invented')
    B=D/'abaqus/explicit_T0p080000/results'
    for name in ['partial_shell.json','shell_field.npz','extract_partial.py','partial_extraction.log','partial_modes.log']:
        shutil.copy2(P/name,B/name)
    shutil.copytree(P/'shell_frames',B/'shell_frames',dirs_exist_ok=True)
    for parent in [D,P,Path(__file__).resolve().parent]:
        (parent/'partial_extraction_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()

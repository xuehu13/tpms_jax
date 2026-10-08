"""Single explicit control, native Windows; data check MUST precede solve.

Each invocation writes exclusive files. Solve requires a reviewed datacheck
gate JSON. No finite-stiffness case or JAX job is automatically submitted.
"""
from pathlib import Path
import argparse,hashlib,json,subprocess,time

BAT=Path('E:/ABAQUS/2026/Commands/abaqus.bat')
P=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['datacheck','datacheck_serial','solve','datacheck_gauge','solve_gauge']);a=p.parse_args()
    assert P.name=='shell_fill_20261007_r25_nearzero'
    receipt=P/(a.phase+'_receipt.json');log=P/(a.phase+'.log')
    assert not receipt.exists() and not log.exists()
    is_solve=a.phase.startswith('solve');is_gauge=a.phase.endswith('_gauge')
    inpname='shell_fill_gauge.inp' if is_gauge else 'shell_fill.inp'
    cfgname='input_gauge.json' if is_gauge else 'input.json'
    if is_solve:
        gate=json.loads((P/('datacheck_gauge_gate.json' if is_gauge else 'datacheck_gate.json')).read_text())
        assert gate['approved_for_nearzero_solve']
        assert gate['INP_sha256']==sha(P/inpname)
    job=('nearzero_gauge' if is_solve else 'check_gauge') if is_gauge else ('check_serial' if a.phase=='datacheck_serial' else ('nearzero' if is_solve else 'shell_fill'))
    assert not (P/(job+'.odb')).exists()
    cpu=4 if a.phase=='datacheck' else 1
    cmd=['cmd','/d','/c',str(BAT),'job='+job,'input='+inpname,
         'cpus='+str(cpu),'double=both','interactive']
    if not is_solve:cmd.append('datacheck')
    inp=sha(P/inpname);cfg=sha(P/cfgname);start=time.perf_counter()
    with log.open('xb') as f:
        result=subprocess.run(cmd,cwd=P,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    logtext=log.read_text(errors='replace')
    rec={'phase':a.phase,'returncode':result.returncode,'wall_seconds':time.perf_counter()-start,
        'command':cmd,'INP_sha256':inp,'input_json_sha256':cfg,
         'input_unchanged':inp==sha(P/inpname) and cfg==sha(P/cfgname),
        'Abaqus_success_log':('Abaqus JOB '+job+' COMPLETED') in logtext,
        'Abaqus_error_log':'exited with errors' in logtext.lower() or 'exception' in logtext.lower(),
        'design_AD':False,'JAX_forward':False}
    if is_solve:
        sta=(P/(job+'.sta')).read_text(errors='replace') if (P/(job+'.sta')).exists() else ''
        rec['analysis_completed_successfully']='THE ANALYSIS HAS COMPLETED SUCCESSFULLY' in sta
        if (P/(job+'.odb')).exists():
            resultname='result_gauge.json' if is_gauge else 'result.json'
            cmd2=['cmd','/d','/c',str(BAT),'python','extract.py',job+'.odb',cfgname,resultname]
            t=time.perf_counter()
            extlog=P/('extraction_gauge.log' if is_gauge else 'extraction.log')
            with extlog.open('xb') as f:
                ext=subprocess.run(cmd2,cwd=P,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            rec['extraction_returncode']=ext.returncode;rec['extraction_wall_seconds']=time.perf_counter()-t
            rec['extraction_success']=(P/resultname).exists() and 'Traceback' not in extlog.read_text(errors='replace')
    receipt.write_text(json.dumps(rec,indent=2))
    print(json.dumps(rec,indent=2))
if __name__=='__main__':main()

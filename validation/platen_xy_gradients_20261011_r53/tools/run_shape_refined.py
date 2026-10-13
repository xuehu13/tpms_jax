"""One bounded targeted FD refinement; reuse existing complete JVP unchanged."""
from pathlib import Path
import hashlib,json,shutil,subprocess,time
R=Path('/home/xuehu/projects/tpms_jax');B=R/'validation/platen_xy_gradients_20261011_r53'
O=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an/output/r53_platen_xy_gradients_20261011')
python=str(R/'.pixi/envs/default/bin/python');start=time.time()
execution='input/G5_shape_refined_execution.json';analysis='input/G5_shape_refined_protocol.json'
spec=json.loads((B/execution).read_text())
assert not json.loads((B/'status.json').read_text())['active_jobs']
with (B/'G5_shape_refined_execution.log').open('w') as file:
    result=subprocess.run([python,'-u',str(B/'tools/run.py'),'execute','--spec',execution],cwd=R,
        stdout=file,stderr=subprocess.STDOUT,timeout=spec['wall_budget_seconds']+150)
receipt=dict(scope='One targeted finer actual-geometry FD check; original failed check retained',
    solver_exit_code=result.returncode,new_complete_dynamic_paths=4,new_JVP_paths=0,reused_complete_JVP='G5_shape/AD/shape',
    threshold_changes=False)
if result.returncode==0:
    with (B/'G5_shape_refined_analysis.log').open('w') as file:
        completed=subprocess.run([python,str(B/'tools/analyze.py'),'--spec',analysis],cwd=R,
            stdout=file,stderr=subprocess.STDOUT,timeout=300)
    receipt['analysis_exit_code']=completed.returncode
    if completed.returncode==0:
        metrics=json.loads((B/'analysis/G5_shape_refined/metrics.json').read_text())
        receipt['passed']=metrics['passed']
        # The shared analyzer includes the cost of the reused AD in its case table.
        receipt['additional_case_wall_minutes']=sum(v for k,v in metrics['additional_case_wall_minutes'].items()
                                                   if k.startswith('G5_shape_refined/'))
        shutil.copytree(B/'analysis/G5_shape_refined',O/'analysis/G5_shape_refined',dirs_exist_ok=True)
receipt['wall_minutes']=(time.time()-start)/60
receipt['protected_sources_unchanged']=all(hashlib.sha256((R/p).read_bytes()).hexdigest()==h
                                         for p,h in spec['protected_sources'].items())
for p in [B/'checks/G5_shape_refined_phase_receipt.json',O/'G5_shape_refined_phase_receipt.json']:
    p.write_text(json.dumps(receipt,indent=2)+'\n')
shutil.copyfile(B/'status.json',O/'status.json')
print(json.dumps(receipt,indent=2),flush=True)

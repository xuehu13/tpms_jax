"""Complete the originally authorized fourth solve. No new Abaqus attempt or timer reset."""
from pathlib import Path
from datetime import datetime,timezone
import importlib.util,json,shutil,sys,time
here=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('r52_coordinator',here/'run_batch.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
B,R,O=m.B,m.R,m.O;E=B/'D';N=m.NROOT/'r52_D_midsurface_end_20pct_20261010'

def main():
    state=m.read(B/'status.json');assert state['jobs_submitted']==3 and state['new_solver_datachecks']==2
    assert not (E/'jax').exists() and not (E/'checks/continue_D_started.json').exists()
    start=m.read(E/'checks/execution_clock.json')['started_unix']
    def remaining(reserve=300):return 6000-(time.time()-start)-reserve
    assert remaining()>120
    archive=E/'checks/abaqus_failure_before_JAX';archive.mkdir()
    for p in [E/'README.md',E/'status.json',E/'analysis/costs.json',B/'status.json',B/'summary.json']:
        dest=archive/(p.name if p.parent==E else p.parent.name+'_'+p.name);shutil.copy2(p,dest)
    m.write(E/'checks/continue_D_started.json',{'UTC':datetime.now(timezone.utc).isoformat(),'original_clock_preserved':True,
        'reason':'Abaqus failed at18.4pct; execute still-unsubmitted original D/JAX and analyze only pre-failure common windows',
        'new_Abaqus_attempts':0,'parameter_changes':False,'diagnostic_compression_windows':[.02,.10,.15],
        'full20_acceptance_not_evaluable_without_complete_Abaqus':True})
    state['cases']['D'].update(Abaqus='failed_excessive_distortion',JAX='not_submitted',result=None);m.write(B/'status.json',state)
    m.status('D','extracting_failed_Abaqus',[])
    shutil.copy2(here/'extract_partial_D.py',N/'extract_partial.py')
    cmd=['cmd','/d','/c',m.ABQ,'python',str(N/'extract_partial.py'),str(N)]
    ex=m.run(cmd,E/'abaqus/partial_extract.log',min(180,remaining()),E/'checks/partial_extract_driver.json')
    assert ex['returncode']==0 and (N/'result_raw.json').exists() and (N/'frames.npz').exists(),'Partial ODB extraction failed'
    for name in ['result_raw.json','frames.npz','raw_history.npz']:shutil.copy2(N/name,E/'abaqus'/name)
    state=m.read(B/'status.json');state['jobs_submitted']+=1;state['cases']['D']['JAX']='running';m.write(B/'status.json',state)
    m.status('D','jax_running',['JAX D'])
    limit=min(3900,remaining());jr=m.run(m.wsl('run_jax.py','D',limit),E/'checks/JAX.log',limit,E/'checks/JAX_driver.json')
    m.status('D','analyzing_partial_reference',[])
    ar=m.run(m.wsl('analyze_partial_D.py','D',min(240,remaining(10))),E/'analysis/analysis.log',min(240,remaining(10)),E/'checks/analysis_driver.json')
    if ar['returncode']:raise RuntimeError('Partial-reference analysis failed; all raw paths preserved')
    original=m.read(archive/'analysis_costs.json')
    cost={**original,'Abaqus_original_failure_retained':True,'Abaqus_check_solve_seconds':sum(m.read(E/'checks'/f'abaqus_{x}_driver.json')['wall_seconds'] for x in ['datacheck','solve']),
        'JAX_driver_seconds':jr['wall_seconds'],'overall_minutes':(time.time()-start)/60,'within_100min':time.time()-start<=6000,
        'no_clock_reset':True,'no_new_Abaqus_attempt':True,'scope':'Original fourth dynamic task with failed/partial Abaqus reference'}
    m.write(E/'analysis/costs.json',cost)
    for p in (E/'analysis').glob('*'):
        if p.is_file():shutil.copy2(p,O/'D'/p.name)
    shutil.copy2(E/'README.md',O/'D/README.md');shutil.copy2(E/'input/protocol.json',O/'D/protocol.json')
    metrics=m.read(E/'analysis/metrics.json');state=m.read(B/'status.json')
    state['cases']['D'].update(stage='partial_reference_analyzed',Abaqus='failed_excessive_distortion',JAX='complete' if metrics['JAX_complete'] else 'partial_or_failed',
        result=True,full20_completed=False,three_scalar_targets_within_10pct=False,full20_acceptance='not_evaluable_reference_failed',overall_minutes=cost['overall_minutes'])
    state.update(stage='batch_finished',active_jobs=[],new_FEM_responses=2+int(metrics['JAX_complete']),next_action='Discuss P pass and D reference failure; no automatic resubmission')
    for p in [B/'status.json',O/'status.json']:m.write(p,state)
    for p in [E/'status.json',O/'D/status.json']:m.write(p,state['cases']['D'])
    summary=m.read(B/'summary.json');summary=[x for x in summary if x['kind']!='D'];summary.append({'kind':'D','metrics':metrics,'costs':cost})
    for p in [B/'summary.json',O/'summary.json']:m.write(p,summary)
    guard=m.read(B/'checks/source_guard_before.json');changed=[p for p,h in guard.items() if m.sha(R/p)!=h];assert not changed,changed
    m.write(E/'checks/continuation_integrity.json',{'original_sources_changed':changed,'original_source_count':len(guard),'Abaqus_INP_sha256':m.sha(N/'platen.inp'),'no_new_Abaqus_solve':True})
    print(json.dumps({'D_JAX_complete':metrics['JAX_complete'],'paired_full20':False,'partial_metrics':metrics.get('partial_diagnostic_windows'),'costs':cost},ensure_ascii=False),flush=True)

if __name__=='__main__':main()

"""One sequential diverse_04 Abaqus/JAX pair; reuse existing bounded launchers."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib.util,json,os,shutil,sys,time
W=Path(r'C:\Users\xuehu\Documents\Codex\2026-10-01\referenced-chatgpt-conversation-this-is-an')
R=Path(r'\\wsl.localhost\Ubuntu-24.04\home\xuehu\projects\tpms_jax')
B=R/'validation/platen_xy_diverse04_20261012_r54';O=W/'output/r54_platen_xy_diverse04_20261012'
E=B/'diverse_04';G=R/'validation/platen_xy_G_20261010_r51'
L='/home/xuehu/projects/tpms_jax';LB=L+'/validation/platen_xy_diverse04_20261012_r54'
PY=L+'/.pixi/envs/default/bin/python';ABQ=r'E:\ABAQUS\2026\Commands\abaqus.bat'
native=Path(r'E:\ABAQUS\2026temp\Abaqus_Work\tpms_jax_abaqus\r54_diverse04_midsurface_end_20pct_20261012')
spec=importlib.util.spec_from_file_location('r52_bounded_launch',R/'validation/platen_xy_PD_20261010_r52/tools/run_batch.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old);run=old.run
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
 tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def status(stage,active=None,**kw):
 s=read(B/'status.json');s.update(stage=stage,active_jobs=active or [],**kw);write(B/'status.json',s);write(O/'status.json',s)
def wsl(script,limit,*args):
 return ['wsl.exe','-d','Ubuntu-24.04','--exec','/usr/bin/timeout','--signal=TERM','--kill-after=10s',str(max(1,int(limit)))+'s',
  '/usr/bin/env','PYTHONUNBUFFERED=1','JAX_PLATFORMS=cuda,cpu','XLA_PYTHON_CLIENT_PREALLOCATE=false','OMP_NUM_THREADS=1',PY,LB+'/tools/'+script,*args]
def main():
 assert not (B/'checks/pair_started.json').exists() and not E.exists() and not native.exists()
 for sub in ['checks','input','abaqus','analysis']:(E/sub).mkdir(parents=True)
 start=time.time();clock={'started_unix':start,'started_UTC':datetime.now(timezone.utc).isoformat(),'overall_budget_seconds':6000}
 write(E/'checks/execution_clock.json',clock);write(B/'checks/pair_started.json',{**clock,'pid':os.getpid()})
 guard=read(B/'checks/source_guard_before.json');assert all(sha(R/p)==h for p,h in guard.items())
 def remaining(reserve=300):return 6000-(time.time()-start)-reserve
 def execute(cmd,log,limit,receipt):
  r=run(cmd,log,limit,receipt)
  if r['returncode']:print(json.dumps({'stage_error':log.name,'receipt':r}),flush=True)
  return r
 cost={};abq_complete=False;jax_complete=False
 status('preparing_inputs',[])
 try:
  lim=min(600,remaining());pr=execute(wsl('prepare.py',lim,'diverse_04'),E/'checks/preparation.log',lim,E/'checks/preparation_receipt.json')
  if pr['returncode']:raise RuntimeError('Input preparation failed; no dynamic jobs submitted')
  cost['preparation_seconds']=time.time()-start
  cfg=read(E/'input/protocol.json');write(O/'model_parameters.json',cfg)
  shutil.copy2(E/'abaqus/platen_20pct.inp',O/'platen_20pct.inp')
  native.mkdir();abqstart=time.time()
  for action in ['datacheck','solve']:
   status('abaqus_'+action,['Abaqus diverse_04 '+action],datachecks=1,new_dynamic_jobs=0 if action=='datacheck' else 1)
   cmd=[sys.executable,str(G/'tools/run_abaqus.py'),action,'--native-dir',str(native),'--output-dir',str(E/'abaqus'),
        '--input',str(E/'abaqus/platen_20pct.inp'),'--config',str(E/'input/protocol.json')]
   lim=min(1500-(time.time()-abqstart),remaining());r=execute(cmd,E/'abaqus'/('driver_'+action+'.log'),lim,E/'checks'/('abaqus_'+action+'_driver.json'))
   if r['returncode'] and action=='datacheck':raise RuntimeError('Abaqus datacheck failed; stop before dynamic submission')
   if action=='solve':abq_complete=r['returncode']==0
  cost['abaqus_check_solve_seconds']=time.time()-abqstart
  status('abaqus_extract',[])
  cmd=[sys.executable,str(G/'tools/run_abaqus.py'),'extract','--native-dir',str(native),'--output-dir',str(E/'abaqus'),'--config',str(E/'input/protocol.json')]
  ext=execute(cmd,E/'abaqus/driver_extract.log',min(150,remaining()),E/'checks/abaqus_extract_driver.json')
  if (native/'result.json').exists() and not (E/'abaqus/result.json').exists():
   shutil.copy2(native/'result.json',E/'abaqus/result.json')
  if (native/'result.json').exists():
   abq_complete=abq_complete and read(native/'result.json')['complete']
  if abq_complete and ext['returncode']==0:
   shutil.copy2(G/'experiments/midsurface_rim_20pct_20261010/tools/extract_rim20_frames.py',native/'extract_frames.py')
   cmd=['cmd','/d','/c',ABQ,'python',str(native/'extract_frames.py'),str(native)]
   fr=execute(cmd,E/'abaqus/frame_extract.log',min(150,remaining()),E/'checks/frame_extract_driver.json')
   if fr['returncode']==0 and (native/'frames.npz').exists():shutil.copy2(native/'frames.npz',E/'abaqus/frames.npz')
  cost['Abaqus_complete']=abq_complete
  # Reference failure does not cancel the already authorized independent JAX attempt.
  status('jax_running',['JAX diverse_04'],new_dynamic_jobs=2,Abaqus_complete=abq_complete)
  lim=min(3900,remaining());jr=execute(wsl('run_jax.py',lim,'diverse_04'),E/'checks/JAX.log',lim,E/'checks/JAX_driver.json')
  cost['JAX_driver_seconds']=jr['wall_seconds']
  if (E/'jax/result.json').exists():jax_complete=read(E/'jax/result.json')['complete']
  status('analyzing',[],Abaqus_complete=abq_complete,JAX_complete=jax_complete)
  if (E/'abaqus/result.json').exists() and (E/'jax/result.json').exists():
   if abq_complete and (E/'abaqus/frames.npz').exists():
    lim=min(240,remaining(10));execute(wsl('analyze.py',lim),E/'analysis/analysis.log',lim,E/'checks/analysis_driver.json')
   lim=min(240,remaining(10));execute(wsl('compare_peaks.py',lim),E/'analysis/peak_analysis.log',lim,E/'checks/peak_analysis_driver.json')
  else:cost['analysis_missing_response']='One solver did not leave extractable path; preserve inputs and logs'
 except Exception as exc:
  cost['driver_failure']={'type':type(exc).__name__,'message':str(exc)};write(E/'checks/driver_failure.json',cost['driver_failure']);print(str(exc),flush=True)
 finally:
  cost.update(overall_minutes=(time.time()-start)/60,within_100min=(time.time()-start)<=6000,
   budgets_minutes={'total':100,'JAX_build_compile_solve':65,'Abaqus_check_solve':25,'preparation_extraction_analysis':10})
  write(E/'analysis/costs.json',cost)
  for p in (E/'analysis').glob('*'):
   if p.is_file():shutil.copy2(p,O/p.name)
  for p in [E/'checks/input_audit.json',E/'checks/preparation_receipt.json']:
   if p.exists():shutil.copy2(p,O/p.name)
  changed=[p for p,h in guard.items() if sha(R/p)!=h]
  write(B/'checks/source_guard_after.json',{'changed':changed,'count':len(guard)});assert not changed,changed
  metrics=read(E/'analysis/peak_metrics.json') if (E/'analysis/peak_metrics.json').exists() else None
  status('complete_pair_analyzed' if abq_complete and jax_complete and metrics else 'partial_or_failed_pair_saved',[],
   complete_pair=abq_complete and jax_complete,Abaqus_complete=abq_complete,JAX_complete=jax_complete,
   overall_minutes=cost['overall_minutes'],source_guard_unchanged=True)
  print(json.dumps({'finished':True,'complete_pair':abq_complete and jax_complete,'minutes':cost['overall_minutes'],'metrics':metrics},ensure_ascii=False),flush=True)
if __name__=='__main__':main()

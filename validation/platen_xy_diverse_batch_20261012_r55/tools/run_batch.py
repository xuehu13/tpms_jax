"""One foreground coordinator, sequential independent pairs, automatic evidence summary."""
from pathlib import Path
from datetime import datetime,timezone
import importlib.util,json,os,subprocess,sys,time
B=Path(__file__).resolve().parents[1];R=B.parents[1]
W=Path(r'C:\Users\xuehu\Documents\Codex\2026-10-01\referenced-chatgpt-conversation-this-is-an');O=W/'output/r55_platen_xy_diverse_batch_20261012'
spec=importlib.util.spec_from_file_location('bounded',R/'validation/platen_xy_PD_20261010_r52/tools/run_batch.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):mod.write(p,x)
def main():
 assert not (B/'checks/batch_started.json').exists()
 cases=read(B/'protocol.json')['cases'];start=time.time()
 write(B/'checks/batch_started.json',{'pid':os.getpid(),'started_unix':start,'UTC':datetime.now(timezone.utc).isoformat(),'maximum_seconds':18000})
 for i,kind in enumerate(cases):
  s=read(B/'status.json');s.update(queue=cases[i+1:],current_case=kind,batch_finished=False);write(B/'status.json',s);write(O/'status.json',s)
  if time.time()-start>=18000-60:break
  # The inner coordinator owns the exact100min clock; 45s only permits its final receipts to flush.
  receipt=mod.run([sys.executable,str(B/'tools/run_pair.py'),kind],B/'checks'/f'{kind}_coordinator.log',min(6045,18000-(time.time()-start)),B/'checks'/f'{kind}_coordinator_receipt.json')
  print(json.dumps({'case_finished':kind,'returncode':receipt['returncode'],'wall_minutes':receipt['wall_seconds']/60}),flush=True)
  # Derive/publicize current results after each pair, without launching new solves.
  mod.run([sys.executable,str(B/'tools/summarize.py')],B/'checks'/f'{kind}_summary.log',120,B/'checks'/f'{kind}_summary_receipt.json')
 s=read(B/'status.json');s.update(stage='batch_attempts_finished',active_jobs=[],queue=[],batch_finished=True,batch_minutes=(time.time()-start)/60)
 write(B/'status.json',s);write(O/'status.json',s)
 mod.run([sys.executable,str(B/'tools/summarize.py'),'--final'],B/'checks/final_summary.log',180,B/'checks/final_summary_receipt.json')
 print(json.dumps({'batch_finished':True,'minutes':s['batch_minutes'],'status':str(B/'status.json')}),flush=True)
if __name__=='__main__':main()

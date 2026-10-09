"""Attach only to the existing r45 solver; no numerical job is launched here."""
from pathlib import Path
import json,os,signal,time
D=Path(__file__).resolve().parent
P=json.loads((D/'protocol.json').read_text());deadline=P['started_unix']+7200-120
receipt=json.loads((D/'monitor_before_attach/receipt.json').read_text());pid=receipt['pid']
receipt.update(attached_monitor_unix=time.time(),monitor_handoff='monitor_handoff.json')
def get(n):
 try:return json.loads((D/n).read_text())
 except (FileNotFoundError,json.JSONDecodeError):return None
def write(n,x):(D/n).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
sent=None
try:
 while True:
  final=get('peak_N64_receipt.json')
  if final:
   if sent is None:
    failure=get('peak_N64/failure.json')
    if failure and failure['type']=='TimeoutError':sent={'reason':'shared_two_hour_budget_internal_guard','last_accepted':failure['last_valid'],'actual_controller_stop':True}
   receipt.update(returncode=final['returncode']);break
  rows=get('peak_N64/accepted_path.json');revision=get('peak_followup_resource_revision.json') or {};delta=revision.get('new_postpeak_increment',.01);stop=None
  if rows and len(rows)>4:
   loading=[x for x in rows if .01<=x['compression'] and x['time']<=.001+1e-12]
   if loading:
    peak=max(loading,key=lambda x:-x['Fz_N']);tail=loading[-3:]
    if len(tail)==3 and rows[-1]['compression']>=peak['compression']+delta and all(-x['Fz_N']<=.9*(-peak['Fz_N']) for x in tail):stop={'reason':'planned_peak_plus_resource_window','observed_peak':peak,'last_accepted':rows[-1],'postpeak_delta_compression':rows[-1]['compression']-peak['compression'],'required_delta':delta}
   if stop is None and rows[-1]['compression']>=.2-1e-12:stop={'reason':'loading_ceiling_without_certified_peak','last_accepted':rows[-1]}
  # Internal controller already stops at deadline. The external watchdog
  # waits 45 seconds for normal checkpoint writes, within the 120s reserve.
  if stop is None and time.time()>deadline+45:stop={'reason':'shared_two_hour_external_watchdog','last_accepted':rows[-1] if rows else None}
  if stop and sent is None:
   stop['requested_unix']=time.time();sent=stop;write('peak_stop_request.json',stop);receipt['planned_stop']=stop;write('peak_monitor_receipt.json',receipt);os.kill(pid,signal.SIGINT)
  if sent and time.time()>sent.get('requested_unix',time.time())+50:
   os.kill(pid,signal.SIGTERM);receipt['forced_terminate_after_save_timeout']=True;break
  time.sleep(4)
finally:
 receipt.update(planned_stop=sent,finished_unix=time.time(),forward_wall_seconds=time.time()-receipt['forward_started_unix'],total_elapsed_seconds=time.time()-P['started_unix'],stopped_as_planned=sent is not None);write('peak_monitor_receipt.json',receipt);print(json.dumps(receipt),flush=True)

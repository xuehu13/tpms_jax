"""Stop only the selected path after the main-plan inertia stop condition."""
from pathlib import Path
import hashlib,json,os,signal,time
O=Path('/home/xuehu/projects/tpms_jax/validation/geometry_transfer_20261006_r15')
P=O/'T0p004_cpu_reference'
assert not (O/'operator_stop.json').exists() and not (P/'result.json').exists()
progress=json.loads((P/'progress.json').read_text());assert progress['KE_over_U']>.05 and progress['rejected_blocks']>=2
matches=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
    try:args=[x.decode() for x in p.read_bytes().split(b'\0') if x]
    except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    if 'scripts/thin_target_explicit.py' in args and '--output' in args and args[args.index('--output')+1]==str(P):
        matches.append(int(p.parent.name))
assert len(matches)==1,matches
note={'main_plan_step':2,'action':'SIGINT to selected JAX solver only','pid':matches[0],
 'reason':'Main plan requires stopping on actual inertia/completion concern. Rejected nonfinite blocks plus KE/U well above 5%; do not extend/retune merely to reach 20%.',
 'last_observed_progress':progress,'unix_time':time.time(),'not_a_budget_or_material_domain_failure_claim':True,
 'complete_20pct':False,'accepted_state_in_memory_not_checkpointed':True,
 'preserved_rejected_field_files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(P.glob('rejected_block_*.npz'))},
 'no_new_jobs_or_source_changes':True,'launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(O/'operator_stop.json').write_text(json.dumps(note,indent=2,allow_nan=False)+'\n')
os.kill(matches[0],signal.SIGINT)
print(json.dumps({'signal_sent_to':matches[0],'compression':progress['compression'],'KE_over_U':progress['KE_over_U']}))

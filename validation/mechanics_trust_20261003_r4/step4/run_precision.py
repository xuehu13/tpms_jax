"""Reuse the measured preflight monitor for the two required accuracy meshes."""
from pathlib import Path
import json,sys
HERE=Path(__file__).resolve().parent;P=Path('/home/xuehu/projects/tpms_jax');O=P/'validation/mechanics_trust_20261003_r4/step4'
decision=json.loads((O/'preflight_decision.json').read_text())
assert decision['status']=='ready_for_precision_lock'
records=json.loads((O/'execution.json').read_text())
assert len(records)==2 and all(r['status']=='ok' for r in records)
measured={r['N']:r for r in records}
b=(measured[32]['peak_sampled_host_RSS_GiB']-measured[16]['peak_sampled_host_RSS_GiB'])/(32**3-16**3)
offset=measured[16]['peak_sampled_host_RSS_GiB']-b*16**3
lock={'phase':'precision_resources_locked_after_measured_preflight','measured':records,
      'RSS_cubic_extrapolation_GiB':{str(n):offset+b*n**3 for n in (48,64)},
      'estimate_limit':'Two-point estimate only; actual guarded execution decides. Same locked per-state/path and memory limits remain.',
      'order':[48,64],'Abaqus_started_yet':False}
(O/'precision_launch.json').write_text(json.dumps(lock,indent=2)+'\n')
source=(HERE/'run_preflight.py').read_text().split("assert not (O/'execution.json').exists()")[0]
scope={'__file__':str(HERE/'run_preflight.py'),'__name__':'monitor_reuse'};exec(compile(source,'existing_preflight_monitor','exec'),scope)
scope['ledger']=records
(O/'run_precision.py').write_bytes(Path(__file__).read_bytes())
for n in (48,64):
    print('PRECISION_MESH_START '+str(n),flush=True)
    r=scope['monitor'](n)
    r['kind']='JAX_precision_path';scope['dump'](O/'execution.json',scope['ledger'])
    if r['status']!='ok':
        scope['dump'](O/'precision_decision.json',{'status':'stopped','failed_mesh':n,'record':r,
             'reason':'Required accuracy mesh did not complete within physical/resource budget','Abaqus_started':False})
        sys.exit(2)
scope['dump'](O/'precision_decision.json',{'status':'both_background_paths_complete','N':[48,64],'Abaqus_started':False,
    'next':'Check grid difference before half increment and independent references'})

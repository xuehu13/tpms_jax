from pathlib import Path
import json
R=Path('/home/xuehu/projects/tpms_jax');O=R/'validation/geometry_interface_20261003_r2'
c=json.loads((O/'step3/N32/summary.json').read_text());m=c['memory']
forward=[]
for f in list((O/'step1').glob('M*_N64.json'))+list((O/'step2').glob('*N64*.json')):
 r=json.loads(f.read_text());forward.append({'file':str(f),'rss_MiB':r['runtime']['peak_host_rss_MiB']})
base=max(r['rss_MiB'] for r in forward)
delta=max(m['adjoint_peak_MiB']-m['forward_peak_MiB'],m['adjoint_rss_MiB']-m['forward_rss_MiB'],0)
meminfo={line.split(':')[0]:int(line.split()[1])/1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.split(':')[0] in ['MemTotal','MemAvailable']}
estimate=base+delta*8
gate={'decision':'skip_N64' if estimate+1024>meminfo['MemAvailable'] else 'host_pass_GPU_gate_still_required','available_host_MiB':meminfo['MemAvailable'],'total_host_MiB':meminfo['MemTotal'],'known_max_N64_forward_peak_MiB':base,'N32_adjoint_added_host_MiB':delta,'scale_factor':8,'estimated_N64_forward_plus_adjoint_peak_MiB':estimate,'minimum_host_headroom_MiB':1024,'estimated_host_headroom_MiB':meminfo['MemAvailable']-estimate,'N32_device_peak_MiB':m['sampled_device_peak_MiB'],'required_device_headroom_MiB':512,'GPU_N64_projection':'Not certified; host gate fails first','estimate_is_exact_bound':False,'method':'Observed N64 peak + N32 adjoint peak/RSS increment times element-count ratio; conservative planning estimate, not an OOM proof','forward_evidence':forward,'N64_new_cost_forward_used':0,'N64_adjoint_used':0}
(O/'step3/N64_resource_gate.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps(gate,indent=2))

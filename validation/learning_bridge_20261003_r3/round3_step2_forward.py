from pathlib import Path
import json,subprocess,sys,time
R=Path('/home/xuehu/projects/tpms_jax');O=R/'validation/learning_bridge_20261003_r3';S=O/'step2'
assert json.loads((O/'step1/summary.json').read_text())['passed']
assert json.loads((O/'regression.json').read_text())['exit_code']==0
assert json.loads((O/'new_checks.json').read_text())['exit_code']==0
rows=[]
for a in json.loads((S/'anchors.json').read_text()):
 for N in [48,64]:
  ledger=json.loads((O/'execution.json').read_text());assert sum(x.get('seconds',0) for x in ledger)<7200
  with (S/f"{a['label']}_N{N}.console.txt").open('x') as stream:
   p=subprocess.run([sys.executable,str(O/'round3_width_forward.py'),a['label'],str(N)],cwd=R,stdout=stream,stderr=subprocess.STDOUT,timeout=1210)
  assert p.returncode==0
  row=json.loads((S/f"{a['label']}_N{N}/summary.json").read_text());assert row['passed']
  rows.append(row);print(json.dumps({k:row[k] for k in ['label','N','K','Vf','seconds']}),flush=True)
 pair=[r for r in rows if r['label']==a['label']];diff=abs(pair[0]['K']-pair[1]['K'])/pair[1]['K']
 assert diff<=.01,'Background grid criterion failed; stop before Abaqus'
baseline=1.9764840370021242
summary={'cases':[{k:r[k] for k in ['label','N','K','Vf','seconds']} for r in rows],'uniform_baseline_K':baseline,'background_passed':True,'Vf_target':.3505280533,'background_max_absolute_mesh_change':max(abs(rows[i]['K']-rows[i+1]['K']) for i in [0,2]),'physical_span_background_only':max([baseline]+[r['K'] for r in rows if r['N']==64])-min([baseline]+[r['K'] for r in rows if r['N']==64]),'binary_reference_state':'not_yet_computed','forward_count':4,'Abaqus_count':0}
(S/'forward_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)

from pathlib import Path
import shutil,json,hashlib
R=Path('/home/xuehu/projects/tpms_jax');D=R/'validation/critical_mode_20261007_r21'
W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
A=D/'mechanical_attempt02';A.mkdir(exist_ok=False)
assert not (D/'results/result.json').exists()
rows=json.loads((D/'results/partial_result.json').read_text())['rows']
for name in ['diagnose.py','run.py','launch_manifest.json','launch_receipt.json','diagnostic.log','process.json','environment.json','basis_checks.json','cost_stop.json']:
 shutil.move(str(D/name),str(A/name))
shutil.move(str(D/'results'),str(A/'results'));(D/'results').mkdir()
copied=[]
for row in rows:
 prefix=f"{row['state'][:-4]}_N{row['Ncoarse']}"
 for p in (A/'results').glob(prefix+'*'):
  shutil.copy2(p,D/'results'/p.name)
  copied.append({'source':str(p.relative_to(D)),'destination':str((D/'results'/p.name).relative_to(D)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(D/'reuse_rows.json').write_text(json.dumps(rows,indent=2))
note={'scientific_protocol_unchanged':True,'operator':'same full-domain material energy Hessian, same Gauss, states, subspaces, gauge, eigensolver settings',
 'implementation':'Nine JVP columns replace fused jacfwd; 9 BLAS contractions replace generic four-operand einsum. Exact expression checks required before acceptance.',
 'completed_rows_reused_without_new_eigen_extraction':[(r['state'],r['Ncoarse']) for r in rows],
 'copied_files':copied,'cumulative_25min_budget_retained':True,'no_new_forward_or_design_AD':True}
(D/'execution_update.json').write_text(json.dumps(note,indent=2))
for name in ['diagnose.py','run.py','repair_cost.py','stop_cost.py']:shutil.copy2(W/'work/critical_mode_20261007'/name,D/name)
print(json.dumps({'completed_rows_reused':len(rows),'files_reused':len(copied),'same_operator_cost_repair':True}))

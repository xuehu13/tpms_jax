"""Add opt-in stability diagnostics; preserve core force and advance AST."""
from pathlib import Path
import ast,json
R=Path('/home/xuehu/projects/tpms_jax');W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an/work/step_control_20261007')
p=R/'scripts/thin_target_explicit.py';s=p.read_text();old=s
anchor='\ndef save_state(path,state,dt,N):';assert s.count(anchor)==1
addition=(W/'bound_functions.py').read_text().split('\n',1)[1]
s=s.replace(anchor,'\n'+addition+anchor,1)
anchor="    if getattr(a,'replay_state',None) is not None:\n"
assert s.count(anchor)==1
s=s.replace(anchor,"    if getattr(a,'stability_states',None) is not None:\n        return saved_stability_probe(ex,a,cfg,N)\n"+anchor,1)
anchor="    p.add_argument('--replay-state'";index=s.index(anchor)
s=s[:index]+"    p.add_argument('--stability-states',type=Path,nargs='+',help='Frozen-state bound/cost only; no time advance')\n    p.add_argument('--stability-batch-cells',type=int,default=128,help='Conservative bound batch size only')\n"+s[index:]
anchor='    run(a)\n';assert s.count(anchor)==1
s=s.replace(anchor,"    if a.stability_batch_cells<1:p.error('--stability-batch-cells must be positive')\n    if a.stability_states is not None and (a.action!='target' or a.replay_state is not None or a.adaptive or a.diagnose_first_failure):\n        p.error('--stability-states is target-only, no replay/adaptive/first-failure option')\n"+anchor,1)
core=lambda x:ast.dump(next(n for n in ast.parse(x).body if isinstance(n,ast.ClassDef) and n.name=='ExplicitXYZ'),include_attributes=False)
assert core(s)==core(old);p.write_text(s)
print(json.dumps({'ExplicitXYZ_AST_unchanged':True,'only_optional_bound_diagnostic_added':True}))

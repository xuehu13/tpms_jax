from pathlib import Path
import ast,json
R=Path('/home/xuehu/projects/tpms_jax');W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an/work/step_control_20261007')
p=R/'scripts/thin_target_explicit.py';s=p.read_text()
anchor='\ndef saved_stability_probe(ex,a,cfg,N):';assert s.count(anchor)==1
s=s.replace(anchor,'\n'+(W/'control_function.py').read_text().split('\n',1)[1]+anchor,1)
anchor="    if getattr(a,'stability_states',None) is not None:\n";assert s.count(anchor)==1
s=s.replace(anchor,"    if getattr(a,'state_step_control',False):\n        return state_step_target(ex,a,cfg,N)\n"+anchor,1)
anchor="    p.add_argument('--stability-states'";i=s.index(anchor)
s=s[:i]+"    p.add_argument('--state-step-control',action='store_true',help='Opt-in conservative state-bound time control; same material/mass/block equations')\n    p.add_argument('--control-budget-seconds',type=float,default=1500,help='Body budget only for opt-in state control')\n"+s[i:]
anchor='    run(a)\n'
s=s.replace(anchor,"    if a.state_step_control and (a.action!='target' or a.stability_states is not None or a.replay_state is not None or a.adaptive or a.diagnose_first_failure):\n        p.error('--state-step-control is target-only and excludes other diagnostic/adaptive modes')\n    if not math.isfinite(a.control_budget_seconds) or a.control_budget_seconds<=0:\n        p.error('--control-budget-seconds must be finite and positive')\n"+anchor,1)
ast.parse(s);p.write_text(s);print(json.dumps({'one_block_integrator':True,'only_state_control_opt_in':True}))

"""Keep pre-closure reading documents; no scientific result is edited."""
from pathlib import Path
import json, shutil
W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
R=Path('/home/xuehu/projects/tpms_jax');O=R/'validation/step_control_20261007_r18'
names=['RESEARCH_BACKGROUND.md','RESEARCH_PLAN.md','PROJECT_OVERVIEW.md','FILE_MAP.md',
       'AGENTS.md','START_HERE.md','TPMS_RESEARCH_REVIEW.md']
wh=W/'history/before_step_control_closure_20261007'
rh=R/'docs/history/before_step_control_closure_20261007'
wh.mkdir(exist_ok=False);rh.mkdir(exist_ok=False)
for name in names:
    shutil.copy2(W/name,wh/name)
    source=R/'AGENTS.md' if name=='AGENTS.md' else R/'README.md' if name=='START_HERE.md' else R/'docs'/('RESEARCH_STATUS.md' if name=='PROJECT_OVERVIEW.md' else name)
    shutil.copy2(source,rh/name)
source_notes={'accessed_date':'2026-10-07','public_manual_version':'2025',
    'installed_solver_version':'2026','public_2026_theory_page_not_accessible_in_browse':True,
    'sources':[
        {'title':'Explicit dynamic analysis','url':'https://docs.software.vt.edu/abaqusv2025/English/SIMACAETHERefMap/simathe-c-expdynamic.htm',
         'relevance':'Central difference, current-frequency limits, varying dt and shell rotational bulk viscosity.'},
        {'title':'Energy balance','url':'https://docs.software.vt.edu/abaqusv2025/English/SIMACAEGSARefMap/simagsa-c-qsienergybal.htm',
         'relevance':'Energy balance and low kinetic energy are separate checks; project keeps its original gates.'},
        {'title':'Unstable static problem: reinforced plate under compressive loads','url':'https://docs.software.vt.edu/abaqusv2025/English/SIMACAEEXARefMap/simaexa-c-unstablestaticplate.htm',
         'relevance':'Localized release can create kinetic energy; does not prove the TPMS mode is physically correct.'},
        {'title':'Buckling of a column with general contact','url':'https://docs.software.vt.edu/abaqusv2025/English/SIMACAEBMKRefMap/simabmk-c-sscxsec.htm',
         'relevance':'Only indexed excerpt read; symmetric models can delay branching. No TPMS imperfection experiment performed.'},
        {'title':'ALAFF Gershgorin summary','url':'https://www.cs.utexas.edu/~flame/laff/alaff/chapter09-summary.html',
         'relevance':'Absolute row-sum eigenvalue upper bounds; element contribution proof is derived for the project.'}],
    'existing_shell_input':'validation/geometry_transfer_20261006_r15/abaqus/explicit_T0p040/thin_shell.inp',
    'input_observation':'No explicit bulk-viscosity or section-control override found; default-version mechanism is an inference, not a measured attribution of ALLVD.'}
(O/'source_notes.json').write_text(json.dumps(source_notes,indent=2,ensure_ascii=False)+'\n')
print('Pre-closure reading documents retained; science untouched.')

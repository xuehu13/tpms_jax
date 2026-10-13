"""A posteriori bound on original loading quality, without extending the path."""
from pathlib import Path
import json, numpy as np
R=Path('/home/xuehu/projects/tpms_jax');O=R/'validation/step_control_20261007_r18'
p=json.loads((O/'controlled_forward/accepted_path.json').read_text())
t=np.array([x['time'] for x in p]);a=np.array([x['compression'] for x in p]);k=np.array([x['KE_over_U'] for x in p])
w=np.r_[0,np.diff(t)];mask=(a>=.01)&(t<=.004+1e-12)
eligible=float(w[mask].sum());bad=float(w[mask&(k>.05)].sum())
remaining=max(0,.004-float(t[-1]));upper=1-bad/(eligible+remaining)
peak=max(p,key=lambda x:-x['Fz_N']);old=json.loads((R/'validation/mechanism_20261007_r17/original_diagnostic/accepted_path.json').read_text())
facts={'observed_eligible_loading_duration_s':eligible,'observed_high_KE_duration_s':bad,
    'maximum_remaining_loading_duration_s':remaining,
    'maximum_possible_final_low_KE_fraction_if_all_remaining_low_KE':upper,
    'original_95pct_quality_still_possible':bool(upper>=.95),'controlled_peak_observed':peak,
    'maximum_KE_over_U_loading_ge1pct':float(k[mask].max()),
    'scope':'Same original right-endpoint duration convention. This upper bound is not a completed trajectory or new forward. Triggered solver stop remains budget, independently of this quality conclusion.'}
(O/'analysis/quality_bound.json').write_text(json.dumps(facts,indent=2,allow_nan=False)+'\n')
print(json.dumps(facts,indent=2))

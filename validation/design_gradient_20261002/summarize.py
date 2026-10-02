"""Recheck saved small-grid derivative evidence and regenerate this report."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import jax_fem.solver as installed_solver
import jax_fem.problem as installed_problem

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source(name, recorded):
    if (ROOT/name).exists() and sha(ROOT/name) == recorded:
        return
    revisions = subprocess.check_output(['git','log','--all','--format=%H','--',name],cwd=ROOT,text=True).splitlines()
    for revision in revisions:
        content = subprocess.run(['git','show',revision+':'+name],cwd=ROOT,capture_output=True)
        if content.returncode == 0 and hashlib.sha256(content.stdout).hexdigest() == recorded:
            return
    raise ValueError('Untraceable calculation source: '+name)


plan = load(OUT/'plan.json')
a = plan['acceptance']
specs = [dict(plan['uniform_c_check'],name='uniform_N4')]
specs += [dict(N=n,theta=plan['theta'],directions=plan['directions'],name=f'periodic_N{n}') for n in plan['grids']]
installed = {name:sha(Path(module.__file__)) for name,module in [('jax_fem.solver',installed_solver),('jax_fem.problem',installed_problem)]}
cases, physical, case_summaries = [], [], []
for spec in specs:
    r = load(OUT/(spec['name']+'.json'))
    if r['status'] != 'ok' or r['case'] != spec:
        raise ValueError('Case not accepted or differs from predeclared plan')
    if r['installed_source_sha256'] != installed:
        raise ValueError('Installed source differs from computation')
    for name, recorded in r['source_sha256'].items():
        verify_source(name, recorded)
    jac = np.asarray(r['ad']['jacobian'])
    if not np.isfinite(jac).all() or not np.allclose(r['ad']['values'],r['baseline']['values'],atol=1e-10,rtol=1e-8):
        raise ValueError('AD values differ from accepted primal')
    if np.any(np.abs(jac[0]-r['ad']['energy_envelope_gradient']) > a['energy_envelope_atol']+a['energy_envelope_rtol']*np.abs(r['ad']['energy_envelope_gradient'])) or np.linalg.norm(jac[2]) < a['nonstationary_gradient_min_norm']:
        raise ValueError('Envelope or nonstationary sensitivity check failed')
    for key in ('Fz_top','U_internal'):
        ref = r['uniform_m4_reference'][key]
        if abs(r['uniform_forward'][key]-ref) > a['uniform_parity_atol']+a['uniform_parity_rtol']*abs(ref):
            raise ValueError('Uniform M4 parity failed')
    physical += [r['baseline'],r['uniform_forward'],r['uniform_m4_reference']]
    final_scaled, final_relative, all_scaled = [], [], []
    for d,declared in zip(r['directions'],spec['directions'],strict=True):
        direction = np.asarray(declared,dtype=float)
        direction /= np.linalg.norm(direction)
        if not np.allclose(direction,d['direction'],atol=1e-15,rtol=0):
            raise ValueError('Direction differs from plan')
        ad = jac @ direction
        if not np.allclose(ad,d['ad_directional'],atol=1e-14,rtol=1e-14):
            raise ValueError('Directional derivative differs from recorded Jacobian')
        per_step = []
        for step,h in zip(d['steps'],plan['central_difference_steps'],strict=True):
            if step['h'] != h:
                raise ValueError('Difference step differs from plan')
            for sign,key in ((1,'plus'),(-1,'minus')):
                if not np.allclose(step[key]['theta'],np.asarray(spec['theta'])+sign*h*direction,atol=1e-15,rtol=0):
                    raise ValueError('Perturbed design differs from plan')
                physical.append(step[key])
            fd = (np.asarray(step['plus']['values'])-np.asarray(step['minus']['values']))/(2*h)
            if not np.allclose(fd,step['fd'],atol=1e-14,rtol=1e-14):
                raise ValueError('Difference reconstruction failed')
            error = np.abs(fd-ad)
            scaled = error/(a['derivative_atol']+a['derivative_rtol']*np.abs(ad))
            per_step.append(scaled)
            if h in plan['central_difference_steps'][-2:]:
                final_scaled.extend(scaled)
                final_relative.extend(error/np.maximum(np.abs(ad),1e-12))
        all_scaled.append(per_step)
    if max(final_scaled) > 1:
        raise ValueError('Last two predeclared steps failed')
    expected_count = 2*len(spec['directions'])*len(plan['central_difference_steps'])
    if r['forward_perturbation_count'] != expected_count:
        raise ValueError('Perturbation count mismatch')
    case_summaries.append({'name':spec['name'],'N':spec['N'],'theta':spec['theta'],
                           'values':r['baseline']['values'],'jacobian':r['ad']['jacobian'],
                           'max_scaled_error_last_two':float(max(final_scaled)),
                           'max_relative_error_last_two':float(max(final_relative)),
                           'nonstationary_gradient_norm':r['nonstationary_gradient_norm'],
                           'perturbations':expected_count,'seconds':r['elapsed_seconds']})
    cases.append((spec,np.max(np.asarray(all_scaled),axis=0)))
for p in physical:
    if p['status'] != 'ok' or not all(p['checks'].values()) or not np.isfinite(p['Fz_top']):
        raise ValueError('Physical acceptance missing')
    for key,limit in (('red_res',1e-8),('balance',1e-8),('work_identity_err',1e-8),('reaction_consistency_err',1e-6)):
        if not np.isfinite(p[key]) or abs(p[key]) > limit:
            raise ValueError('Saved physical quantity failed '+key)
test_text = (OUT/'full_tests.txt').read_text()
match = re.search(r'(\d+) passed in ([\d.]+)s',test_text)
if not match or int(match[1]) != 125 or 'failed' in test_text.lower():
    raise ValueError('Full regression is not 125/125')
summary = {'case_count':len(cases),'perturbed_forward_solves':sum(r['perturbations'] for r in case_summaries),
           'cases':case_summaries,'full_tests_passed':int(match[1]),'full_tests_seconds':float(match[2]),
           'max_scaled_error_last_two':max(r['max_scaled_error_last_two'] for r in case_summaries),
           'max_relative_error_last_two':max(r['max_relative_error_last_two'] for r in case_summaries),
           'max_physical_abs':{k:max(abs(p[k]) for p in physical) for k in ('red_res','balance','reaction_consistency_err','work_identity_err')},
           'fixed_lateral_discrete_first_derivatives_verified':True,'continuum_gradient_converged':False,
           'free_lateral_gradient_verified':False,'volume_calibration_derivative_verified':False,
           'optimization_performed':False,'new_abaqus_jobs':0}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
fig,axes = plt.subplots(1,3,figsize=(13.5,4),layout='constrained')
for axis,name,index in zip(axes,('Axial apparent modulus','Projected volume','Displacement diagnostic'),range(3)):
    for spec,scaled in cases:
        axis.loglog(plan['central_difference_steps'],np.maximum(scaled[:,index],1e-12),'o-',label=spec['name'])
    axis.axhline(1.,color='gray',ls='--',label='Declared acceptance limit')
    axis.axvspan(plan['central_difference_steps'][-1],plan['central_difference_steps'][-2],color='gray',alpha=.08)
    axis.set(xlabel='Central-difference step h',ylabel='Max error / declared limit',title=name)
    axis.grid(alpha=.2,which='both')
axes[0].legend(fontsize=7)
fig.savefig(OUT/'gradient_check.png',dpi=180)
plt.close(fig)
source = dict(load(OUT/'periodic_N8.json')['source_sha256'])
source['tests/test_design_fem.py'] = sha(ROOT/'tests/test_design_fem.py')
manifest = {'parent_commit':plan['parent_commit'],'current_source_sha256':source,
            'installed_source_sha256':installed,
            'artifacts_sha256':{p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name not in ('source_manifest.json','README.md')}}
(OUT/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(summary,indent=2),flush=True)

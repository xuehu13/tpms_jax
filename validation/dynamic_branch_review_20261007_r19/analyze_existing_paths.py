"""Read frozen accepted histories; no FEM imports, time advance or design AD."""
from pathlib import Path
import hashlib, json, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = Path('/home/xuehu/projects/tpms_jax')
W = Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
D = R / 'validation/dynamic_branch_review_20261007_r19'
jfile = R / 'validation/step_control_20261007_r18/controlled_forward/accepted_path.json'
sfile = R / 'validation/geometry_transfer_20261006_r15/abaqus/explicit_T0p040/results/shell.json'
j = json.loads(jfile.read_text()); shell = json.loads(sfile.read_text()); s = shell['force_path']
jload = [p for p in j if p['compression'] >= 0.01]
sload = [p for p in s if 0.01 <= p['compression'] and p['time'] <= 0.040]
jpeak = max(jload, key=lambda p: -p['Fz_N'])
speak = max(sload, key=lambda p: -p['Fz_N'])
idx = j.index(jpeak)
work = float(np.trapezoid([-p['Fz_N'] for p in j[idx:]], [10*p['compression'] for p in j[idx:]]))
du = j[-1]['energy_N_mm'] - jpeak['energy_N_mm']
dk = j[-1]['KE_N_mm'] - jpeak['KE_N_mm']
observations = {
    'source_files': [{'path': str(p.relative_to(R)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in (jfile, sfile)],
    'not_new_simulation': True, 'observations_are_saved_endpoints_not_internal_steps': True,
    'JAX_history_complete_20pct': False,
    'JAX_observed_dynamic_peak': jpeak, 'shell_observed_dynamic_peak': speak,
    'peak_compression_difference_percentage_points': 100*(jpeak['compression']-speak['compression']),
    'peak_force_relative_shell_difference': (-jpeak['Fz_N'])/(-speak['Fz_N'])-1,
    'JAX_peak_to_interrupted_endpoint': {
        'start_compression': jpeak['compression'], 'end_compression': j[-1]['compression'],
        'delta_elastic_N_mm': du, 'delta_kinetic_N_mm': dk,
        'applied_work_trapezoid_N_mm': work, 'delta_energy_minus_work_N_mm': du+dk-work,
        'not_a_settled_hold_endpoint': True,
        'interpretation': 'Energy release plus continued loading is consistent with a dynamic snap; energy balance alone does not certify correct mode or spatial accuracy.'},
    'shell_original_checks': shell['checks'],
    'loading_times_s': {'JAX': 0.004, 'shell': 0.040},
    'new_dynamic_scope_does_not_change_old_checks': True}
(D / 'history_analysis.json').write_text(json.dumps(observations, indent=2), encoding='utf-8')

plt.rcParams.update({'font.size': 10})
fig, ax = plt.subplots(3, 1, figsize=(9, 10), sharex=True, constrained_layout=True)
ja = np.array([100*p['compression'] for p in jload]); sa = np.array([100*p['compression'] for p in sload])
ax[0].plot(ja, [-p['Fz_N'] for p in jload], label='JAX r18: load 0.004 s (partial)')
ax[0].plot(sa, [-p['Fz_N'] for p in sload], label='Abaqus shell: load 0.040 s')
for p, color in ((jpeak, 'C0'), (speak, 'C1')):
    ax[0].plot(100*p['compression'], -p['Fz_N'], 'o', color=color)
    ax[0].annotate(f"{100*p['compression']:.3f}%", (100*p['compression'], -p['Fz_N']), xytext=(6, 8), textcoords='offset points', color=color)
ax[0].set_ylabel('Compressive force (N)'); ax[0].legend(loc='upper left')
ax[1].plot(ja, [100*p['KE_over_U'] for p in jload], label='JAX: KE / stored energy U')
ax[1].plot(sa, [100*p['ALLKE']/p['ALLIE'] for p in sload], label='Shell: ALLKE / ALLIE (includes artificial energy)')
ax[1].axhline(5, color='grey', linestyle=':', label='Old quasistatic guide, not a dynamic failure line')
ax[1].set_ylabel('Kinetic / internal energy (%)'); ax[1].legend(loc='upper left', fontsize=8)
ax[2].plot(ja, [p['energy_N_mm'] for p in jload], label='JAX stored energy U')
ax[2].plot(ja, [p['KE_N_mm'] for p in jload], label='JAX kinetic energy KE')
ax[2].plot(ja, [p['energy_N_mm']+p['KE_N_mm'] for p in jload], label='JAX U + KE')
ax[2].set_ylabel('Energy (N mm)'); ax[2].set_xlabel('Macroscopic compression (%)'); ax[2].legend(loc='upper left')
for a in ax:
    a.grid(alpha=.25); a.set_xlim(1,20)
    a.axvline(100*j[-1]['compression'], color='C0', linewidth=.7, linestyle='--')
fig.suptitle('diverse_04, L = 10 mm, t = 0.50 mm, NH, XYZ periodic\nDifferent loading rates; saved observations only; JAX ends at 17.536%')
fig.savefig(D / 'response_energy.png', dpi=180)
Wfig = W / 'output/figures/DYNAMIC_BRANCH_response_energy.png'
shutil.copy2(D / 'response_energy.png', Wfig)
plt.close(fig)
print(json.dumps({k: observations[k] for k in ('peak_compression_difference_percentage_points','peak_force_relative_shell_difference','JAX_peak_to_interrupted_endpoint')}, indent=2))

"""r25 collection/attribution gate; read existing JAX data, do not solve JAX."""
from pathlib import Path
import hashlib,json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/xuehu/projects/tpms_jax')
D=R/'validation/shell_fill_diagnostic_20261007_r25'
P=Path('/mnt/e/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus/shell_fill_20261007_r25_nearzero')
OLD=P.parent/'shell_rate_20261007_r20_diverse04_explicit_T0p004'
W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def plot_response(original,near,j,path,passed):
    fig,ax=plt.subplots(figsize=(9,5),constrained_layout=True)
    for rows,color,label in [(original,'#244b70','Original shell, same 0.004s load'),
            (near,'#ad522a','Near-zero filler + embedding control'),
            (j,'#4f7956','Existing JAX r18 (partial, no new solve)')]:
        x=np.array([r['compression'] for r in rows]);y=-np.array([r['Fz_N'] for r in rows])
        sel=x<=.1754;ax.plot(100*x[sel],y[sel],color=color,label=label)
    if not passed:
        p=near[-1];ax.plot(100*p['compression'],-p['Fz_N'],'x',color='#ad522a',markersize=8)
        ax.annotate(f"Aborted at {100*p['compression']:.2f}%\nNo snap peak comparison",(100*p['compression'],-p['Fz_N']),
            xytext=(22,-50),textcoords='offset points',fontsize=9,color='#ad522a',arrowprops={'arrowstyle':'->','color':'#ad522a'})
    ax.set(xlabel='Macroscopic compression (%)',ylabel='Compression reaction (N)',xlim=(0,17.6))
    ax.grid(alpha=.22);ax.legend(fontsize=9);ax.axvline(17,color='#888888',lw=.8,ls=':')
    ax.set_title('diverse_04: ordinary near-zero filler control\nAttribution gate: '+('PASS' if passed else 'FAIL')+'; no 20% or gradient claim',fontsize=11)
    fig.savefig(path,dpi=170);plt.close(fig)
def main():
    assert not (D/'analysis').exists()
    A=D/'analysis';A.mkdir();B=D/'results';B.mkdir()
    receipt=json.loads((P/'solve_gauge_receipt.json').read_text());protocol=json.loads((D/'protocol.json').read_text())
    for name in ['solve_gauge_receipt.json','solve_gauge.log','nearzero_gauge.sta','nearzero_gauge.dat',
                 'nearzero_gauge.msg','result_gauge.json','result_gauge_verified.json','fatal_field_diagnostic.json','extraction_gauge.log','input_gauge.json','shell_fill_gauge.inp']:
        if (P/name).exists():shutil.copy2(P/name,B/name)
    frozen=json.loads((D/'freeze_before.json').read_text())
    preservation={p:sha(Path(p))==h for p,h in frozen.items()}
    write(D/'preservation.json',preservation);assert all(preservation.values())
    odb=P/'nearzero_gauge.odb'
    write(B/'retention.json',{'ODB_retained':str(odb),'ODB_sha256':sha(odb) if odb.exists() else None,
        'no_duplicate_ODB':True,'old_program_and_scientific_files_unchanged':all(preservation.values())})
    if not (P/'result_gauge.json').exists():
        decision={'nearzero_gate_pass':False,'state':'no_extractable_curve',
            'next_action':'inspect preserved solver/extraction failure; no finite-fill launch',
            'Abaqus_receipt':receipt,'physical_attribution':False}
        write(D/'decision.json',decision);print(json.dumps(decision,indent=2));return
    trial=json.loads((P/'result_gauge_verified.json').read_text());shell=json.loads((OLD/'shell.json').read_text())
    original=[r for r in shell['force_path'] if r['time']<=.004]
    near=trial['force_path'];limit=min(.17,near[-1]['compression'])
    def curve(rows):return np.array([r['compression'] for r in rows]),-np.array([r['Fz_N'] for r in rows])
    sx,sf=curve(original);nx,nf=curve(near)
    def peak(rows,end):
        a=[r for r in rows if .01<=r['compression']<=end+1e-12]
        return max(a,key=lambda r:-r['Fz_N']) if a else None
    sp=peak(original,.17);npk=peak(near,limit)
    op=[r for r in original if r['compression']<=.17]
    oscale=max(abs(r['ALLWK']) for r in op)
    oq={'comparison_range':[0.,.17],
        'energy_drift_relative_work':max(abs(r['ETOTAL']-op[0]['ETOTAL']) for r in op)/oscale,
        'artificial_energy_relative_work':max(abs(r['ALLAE']) for r in op)/oscale,
        'original20pct_checks_preserved':shell['checks']}
    metrics={}
    if limit>=.01:
        grid=np.linspace(.01,limit,1001);delta=np.interp(grid,nx,nf)-np.interp(grid,sx,sf)
        rms=float(np.sqrt(np.mean(delta*delta)));matched=-sp['Fz_N'];fixed=2.25891
        metrics={'available_common_range':[.01,float(limit)],'predeclared_range_complete':bool(limit>=.17-1e-7),
            'curve_RMS_N':rms,'curve_RMS_original_fixed_denominator':rms/fixed,
            'curve_RMS_matched_fast_peak_denominator':rms/matched,
            'observed_peak_shift_percentage_points':100*(npk['compression']-sp['compression']) if trial['target_reached'] else None,
            'observed_peak_force_relative_difference':npk['Fz_N']/sp['Fz_N']-1 if trial['target_reached'] else None,
            'trial_observed_peak_at_last_history_point':abs(npk['compression']-limit)<1e-6,
            'observed_peak_not_always_a_bracketed_snap':True,
            'RF_at_10pct_original_N':float(np.interp(.1,sx,sf)),
            'RF_at_10pct_trial_N':float(np.interp(.1,nx,nf)) if limit>=.1 else None,
            'dynamic_peaks_not_certified_critical_strains':True,'partial_peak_not_comparable_if_target_incomplete':not trial['target_reached']}
    g=protocol['predeclared_gate']
    fatal=json.loads((P/'fatal_field_diagnostic.json').read_text()) if (P/'fatal_field_diagnostic.json').exists() else None
    checks={'solver_completed':bool(receipt['analysis_completed_successfully']),
        'extraction_success':bool(receipt.get('extraction_success',False)),
        'target17_reached':trial['target_reached'],'finite':trial['finite_curve'],
        'peak_shift_le_0p5pp':bool(metrics and trial['target_reached'] and abs(metrics['observed_peak_shift_percentage_points'])<=g['observed_peak_shift_max_percentage_points']),
        'peak_force_difference_le5pct':bool(metrics and trial['target_reached'] and abs(metrics['observed_peak_force_relative_difference'])<=.05),
        'curve_fixed_RMS_le5pct':bool(metrics and trial['target_reached'] and metrics['curve_RMS_original_fixed_denominator']<=.05),
        'curve_matched_RMS_le5pct':bool(metrics and trial['target_reached'] and metrics['curve_RMS_matched_fast_peak_denominator']<=.05),
        'energy_drift_le1pct':trial['global_energy_drift_relative_work']<=.01,
        'artificial_energy_le5pct':trial['artificial_energy_relative_work']<=.05,
        'sampled_PBC_le1e6_mm':trial['max_sampled_PBC_error_mm']<=1e-6,
        'sampled_UR_PBC_le1e6_rad':trial['max_sampled_UR_PBC_error_rad']<=1e-6,
        'sampled_positive_host_volume':trial['min_sampled_host_EVOL_mm3'] is not None and trial['min_sampled_host_EVOL_mm3']>0,
        'fatal_diagnostic_no_inverted_host':fatal is None or fatal['center_J_from_failed_field']>0}
    passed=all(checks.values())
    group_energies={}
    for name,hist in trial['element_group_histories'].items():
        group_energies[name]={}
        for key,rows in hist.items():
            ar=np.asarray(rows,float)
            if ar.size:
                group_energies[name][key]={'at_trial_observed_peak_N_mm':float(np.interp(npk['time'],ar[:,0],ar[:,1])) if npk else None,
                    'at_last_history_N_mm':float(ar[-1,1])}
    comparison={'original_fast_shell_peak':sp,'nearzero_trial_peak_observed':npk,'metrics':metrics,
        'checks':checks,'nearzero_attribution_gate_pass':passed,'last_compression':trial['last_compression'],
        'added_mass_fraction':protocol['estimated_mass']['added_over_shell'],
        'global_energy_drift_relative_work':trial['global_energy_drift_relative_work'],
        'artificial_energy_relative_work':trial['artificial_energy_relative_work'],
        'wall_seconds':receipt['wall_seconds'],'element_group_history_names':list(trial['element_group_histories']),
        'original_shell_quality_same_17pct_window':oq,
        'fatal_field_diagnostic':fatal,
        'group_energy_at_peak_and_last':group_energies,
        'not_JAX20pct_validation':True,'no_JAX_forward_or_AD':True}
    op_available=[r for r in original if r['compression']<=limit]
    avalscale=max(abs(r['ALLWK']) for r in op_available)
    comparison['original_shell_quality_available_window']={
        'compression_range':[0.,float(limit)],
        'energy_drift_relative_work':max(abs(r['ETOTAL']-op_available[0]['ETOTAL']) for r in op_available)/avalscale,
        'artificial_energy_relative_work':max(abs(r['ALLAE']) for r in op_available)/avalscale}
    write(A/'comparison.json',comparison)
    j=json.loads((R/'validation/step_control_20261007_r18/controlled_forward/accepted_path.json').read_text())
    jx,jf=curve(j)
    plot_response(original,near,j,A/'response.png',passed)
    decision={'nearzero_gate_pass':passed,'checks':checks,
        'response_gate_pass':all(checks[k] for k in ['target17_reached','peak_shift_le_0p5pp','peak_force_difference_le5pct','curve_fixed_RMS_le5pct','curve_matched_RMS_le5pct']),
        'numerical_screen_pass':all(checks[k] for k in ['solver_completed','finite','energy_drift_le1pct','artificial_energy_le5pct','sampled_PBC_le1e6_mm','sampled_UR_PBC_le1e6_rad','sampled_positive_host_volume','fatal_diagnostic_no_inverted_host']),
        'next_action':'finite-fill trial may be separately assessed using identical grid/mass/coupling' if passed else 'do not submit finite-fill attribution; evaluate the coupling/numerical screen failure first',
        'soft_void_is_proven_unique_cause':False,'nearzero_control_does_not_certify_JAX':True,
        'no_finite_fill_submitted':True,'no_new_JAX_or_design_AD':True}
    write(D/'decision.json',decision)
    print(json.dumps(comparison,indent=2))
if __name__=='__main__':main()

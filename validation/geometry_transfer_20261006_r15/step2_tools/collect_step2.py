"""Retain the bounded attempt; do not manufacture a completed JAX path."""
from pathlib import Path
import hashlib,json,shutil
R=Path('/home/xuehu/projects/tpms_jax');O=R/'validation/geometry_transfer_20261006_r15'
W=Path('/mnt/c/Users/xuehu/Documents/Codex/2026-10-01/referenced-chatgpt-conversation-this-is-an')
D=W/'work/geometry_forward_20261006';P=O/'T0p004_cpu_reference'
A=Path('/mnt/e/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus/geometry_transfer_20261006_r15_diverse04_explicit_T0p040')
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    assert not p.exists(),p
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
receipt=read(O/'T0p004_cpu_reference_receipt.json');stop=read(O/'operator_stop.json')
assert receipt['returncode']==-2 and not receipt['result_exists'] and not receipt['failure_exists']
assert not (P/'field.npz').exists() and not (P/'last_valid_field.npz').exists()
cfg=read(P/'input.json');input_cfg=read(O/'input.json');rej=read(P/'rejected_blocks.json')
assert cfg['reference_geometry_on_cpu'] and cfg['devices'][0]=='cuda:0'
for k in ['case_id','N','cell_size_mm','thickness_mm','E_MPa','nu','eta','interface_10_90_mm','material_model']:
    assert cfg[k]==input_cfg[k],k
assert cfg['Gauss_field_sha256']==sha(O/'gauss_field.npz') and cfg['load_time_seconds']==.004
assert cfg['element_type']=='HEX27' and cfg['Gauss_points_per_cell']==27
assert cfg['mass_lumping']=='HRZ_positive_diagonal_scaling'
assert cfg['virtual_NH_weight_bounds']==[.001,.01] and cfg['virtual_continuation_max_J']==.1
assert cfg['experiment_sha256']==sha(R/'scripts/thin_target_explicit.py')
assert len(rej)==3 and all('Nonfinite state or material energy/force'==r['reason'] for r in rej)
assert read(A/'launch_receipt.json')['analysis_completed_successfully'] and read(A/'launch_receipt.json')['returncode']==0
assert read(A/'extraction_corrected_receipt.json')['returncode']==0
shell=read(A/'shell.json');scfg=read(A/'input.json')
assert shell['checks']['complete'] and shell['checks']['target_compression'] and shell['checks']['finite']
assert sha(A/'thin_shell.inp')==scfg['shell_inp_sha256']
assert sha(A/'input.json')==sha(O/'abaqus/explicit_T0p040/input.json')
assert sha(A/'extract_thin_explicit.py')==sha(R/'scripts/extract_thin_explicit.py')
com=(A/'thin_shell.com').read_text();assert "'cpus':4" in com and "'double_precision':BOTH" in com
dest=O/'abaqus/explicit_T0p040/results';dest.mkdir(exist_ok=False)
native_records={}
for p in sorted(A.iterdir()):
    if not p.is_file():continue
    native_records[p.name]={'native_path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
    if p.suffix in {'.json','.npz','.sta','.dat','.msg','.log'}:shutil.copy2(p,dest/p.name)
write(dest/'retention.json',{'native_directory':str(A),'files':native_records,'ODB_retained_native':True,
 'job_count':1,'read_only_extraction_retry_count':1,'input_INP_extractor_unchanged':True})
rows=[]
for line in (O/'T0p004_cpu_reference.log').read_text().splitlines():
    if line.startswith('{"time":'):
        row=json.loads(line);assert row['invalid_material_points']==0 and row['J_finite'];rows.append(row)
rows += [r['last_valid'] for r in rej]
rows=sorted({r['time']:r for r in rows}.values(),key=lambda r:r['time'])
write(O/'accepted_logged_observations.json',{
 'scope':'Only printed accepted observations and rejected-block last_valid records, not the full accepted path or a final-state checkpoint',
 'source_log_sha256':sha(O/'T0p004_cpu_reference.log'),'source_rejections_sha256':sha(P/'rejected_blocks.json'),
 'no_interpolated_or_invented_rows':True,'rows':rows})
summary={'main_plan_step':2,'status':'attempted_not_accepted_at_20pct',
 'main_plan_step1_complete':True,'step2_successful_completion':False,
 'next_only':'step3 bounded diagnosis of existing records; no new forward/AD/training',
 'research_question':'Can the frozen candidate migrate to one different midsurface without retuning?',
 'JAX':{'status':'stopped_for_observed_quality_concern','last_observed_progress':stop['last_observed_progress'],
 'first_nonfinite_rejected_block_previous_compression':rej[0]['last_valid']['compression'],
 'first_rejection_last_valid_KE_over_U':rej[0]['last_valid']['KE_over_U'],'rejected_blocks':len(rej),
 'wall_seconds':receipt['wall_seconds'],'first_preflight_wall_seconds':read(O/'T0p004_receipt.json')['wall_seconds'],
 'maximum_KE_over_U_in_sparse_logged_rows':max(r['KE_over_U'] for r in rows),
 'minimum_required_positive_J_in_sparse_logged_rows':min(r['required_positive_J_min'] for r in rows),
 'sparse_logged_observations':len(rows),'full_accepted_path_retained':False,'accepted_final_field_retained':False,
 'rejected_full_fields_retained':len(list(P.glob('rejected_block_*.npz'))),
 'rejected_fields_are_valid_accepted_states':False,'material_domain_failure_claimed':False,
 'global_material_domain_certified':False,'preprocessing_compile_cost_and_peak_RSS_not_persisted':True},
 'shell':{'checks':shell['checks'],'hold_mean_Fz_N':shell['hold_mean_Fz_N'],
 'macro_input_work_N_mm':shell['macro_work_from_RF_U_N_mm'],
 'energy_drift_relative_work':shell['global_energy_drift_relative_work'],
 'artificial_energy_relative_max_work':shell['artificial_energy_relative_max_work'],
 'terminal_KE_over_IE':shell['KE_fraction_of_IE_terminal'],
 'loading_time_fraction_KE_below_5pct':shell['loading_time_fraction_KE_below_5pct'],
 'terminal_energies_N_mm':shell['energies_N_mm'],'wall_seconds':read(A/'launch_receipt.json')['wall_seconds'],
 'corrected_extraction_seconds':read(A/'extraction_corrected_receipt.json')['seconds'],
 'strict_reference_quality_pass':all(shell['checks'].values())},
 'new_JAX_time_paths_started':1,'new_completed_JAX_paths':0,'new_Abaqus_jobs':1,
 'new_full_AD_jobs':0,'new_training_jobs':0,'extra_slow_jobs':0,
 'JAX_preflight_failure_preserved':'Exact CPU cache/GPU reference-point assertion; no time integration. Existing CPU-reference/GPU advancement then passed unchanged check.',
 'shell_extraction_failure_preserved':'Missing input.json argument; wrapper returned 0 but no JSON. Corrected read-only extraction, kept original logs.',
 'fixed_physical_and_numerical_constants':True,'source_changed':False,'not_a_budget_stop':True,
 'dense_27_125_final_accepted_field_check_completed':False,
 'paired_20pct_error_computed':False,'new_geometry_accuracy_accepted':False,'gradient20_certified':False,
 'known_limit':'KeyboardInterrupt is not caught by the existing Exception handler; no accepted q/v checkpoint or full accepted path persisted. Rejected fields and sparse observations are not a valid final state.',
 'input_hashes':{'JAX_log':sha(O/'T0p004_cpu_reference.log'),'operator_stop':sha(O/'operator_stop.json'),
 'shell_result':sha(A/'shell.json'),'shell_field':sha(A/'shell_field.npz')}}
write(O/'step2_summary.json',summary)
tools=O/'step2_tools';tools.mkdir(exist_ok=True)
for name in ['run_jax.py','run_jax_cpu_reference.py','run_shell.py','extract_shell.py','stop_jax_for_quality.py','collect_step2.py']:
    target=tools/name
    if target.exists():assert sha(target)==sha(D/name)
    else:shutil.copy2(D/name,target)
attempt=tools/'run_shell_initial.py';shutil.copy2(D/'source_attempt/run_shell_initial.py',attempt)
assert sha(attempt)==read(A/'launch_manifest.json')['launcher_sha256']
print(json.dumps({k:summary[k] for k in ['status','JAX','shell','known_limit','next_only']},ensure_ascii=False,indent=2))

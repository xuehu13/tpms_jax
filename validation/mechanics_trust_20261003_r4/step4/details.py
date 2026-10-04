from pathlib import Path
import json,hashlib
P=Path('/home/xuehu/projects/tpms_jax');O=P/'validation/mechanics_trust_20261003_r4/step4'
def load(f):return json.loads(f.read_text())
left=load(O/'N16_d005.json');right=load(O/'N32_d005.json')
keyed=lambda rows:{round(r['compression'],9):r for r in rows}
l=keyed(left['rows']);r=keyed(right['rows']);scaleF=max(abs(v['Fz_top']) for v in r.values());scaleU=max(v['energy'] for v in r.values())
coarse={'force_path_difference_over_N32_max':max(abs(l[k]['Fz_top']-r[k]['Fz_top']) for k in r)/scaleF,
        'energy_path_difference_over_N32_max':max(abs(l[k]['energy']-r[k]['energy']) for k in r)/scaleU,
        'interpretation':'Coarse-mesh indication only; cannot substitute required N48/N64 check or independent solid reference'}
oldfile=P/'validation/near_term_20261003/step1/N48_fixed.json';old=load(oldfile)
assert old['beta']==40 and old['emin_ratio']==1e-4 and old['model']['c']==.541062
small=keyed(load(O/'N48_d005.progress.json'))[.0001];oldK=-old['Fz_top']/.01;newK=-small['Fz_top']/.0001
summary=load(O/'summary.json');n48=summary['cases'][-1]
profile=(O/'N48_d005.log').read_text();stopmarker=profile.rsplit('START_STATE ',1)[-1]
details={'coarse_grid':coarse,'small_strain_N48':{'compression':.0001,'old_linear_stiffness':oldK,'new_finite_strain_secant':newK,
      'relative_difference':abs(newK-oldK)/oldK,'pass_locked_1percent':abs(newK-oldK)/oldK<=.01,
      'old_record':str(oldfile),'old_record_sha256':hashlib.sha256(oldfile.read_bytes()).hexdigest()},
    'N48_resource_boundary':{'last_completed_compression':.0001,'attempted_compression':.005,'state_limit_seconds':300,
      'total_attempt_wall_seconds':n48['record']['wall_seconds'],'completed_small_point_seconds':small['solve_seconds'],
      'peak_sampled_host_RSS_GiB':n48['record']['peak_sampled_host_RSS_GiB'],
      'completed_point_reported_host_peak_GiB':small['host_RSS_peak_MiB']/1024,
      'peak_device_total_used_GiB':n48['record']['peak_device_total_used_GiB'],
      'first_correction_linear_seconds':210.376,'last_logged_residual_l2':6.12e-6,
      'memory_limit_triggered':False,'negative_detF_or_solver_divergence_reported':False,
      'interpretation':'Stopped during second linear correction by external time guard. Incomplete point is not an accepted equilibrium.'},
    'performance_limit':'One configured host/device and PETSc GMRES/GAMG. This is not an intrinsic impossibility proof; preconditioner setup versus Krylov iteration not separately timed.',
    'FEM_solves_added_by_this_analysis':0}
assert details['small_strain_N48']['pass_locked_1percent']
(O/'details.json').write_text(json.dumps(details,indent=2)+'\n');(O/'details.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(details,indent=2))

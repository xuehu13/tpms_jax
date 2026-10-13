"""Read-only paired local JAX/Abaqus solid comparison; no dynamic conclusions."""
from pathlib import Path
import csv, hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R = Path('/home/xuehu/projects/tpms_jax')
D = R / 'validation/platen_xy_G_20261010_r51'
B = D / 'diagnostics/end_section_benchmark_20261010'
A = B / 'analysis/jax_comparison';A.mkdir(exist_ok=True)
ref = json.loads((B / 'abaqus/cap/result.json').read_text())
rim = json.loads((B / 'abaqus/rim/result.json').read_text())
cfg = json.loads((B / 'input/jax_protocol.json').read_text())
grid = np.linspace(0, .02, 201)
ref_x = np.array([r['compression'] for r in ref['path']]);ref_f = np.array([r['force_N'] for r in ref['path']])
ref_force = np.interp(grid, ref_x, ref_f);den = ref_force.max()
ref_work = float(np.trapezoid(ref_f, ref_x * 10))
abq_field = np.load(B / 'abaqus/cap/field.npz')
old_mesh = np.load(B / 'input/solid_mesh.npz')
indices = old_mesh['indices'];mapping = (indices[:,0] * 3 + indices[:,1]) * 81 + indices[:,2]
metrics = {'reference':{'case':'Abaqus local C3D20R full cap static hard contact','force_denominator_N':float(den),
    'endpoint_force_N':ref['path'][-1]['force_N'],'RF_U_work_N_mm':ref_work,'frames':len(ref['path'])},
    'scope':'Local conforming fully solid static validation, not G, no background occupancy/HRZ/dynamic or design-gradient certification.',
    'cases':{}}
curves = {'Abaqus full cap':ref_force}
for c in ('nodal','gauss'):
    result = json.loads((B / f'jax/{c}/result.json').read_text());assert result['complete']
    rows = result['path'];x = np.array([r['compression'] for r in rows]);f = np.array([r['force_N'] for r in rows])
    assert np.all(np.diff(x) > 0)
    force = np.interp(grid, x, f);curves['JAX '+c] = force
    last = rows[-1]
    data = np.load(B / f'jax/{c}/field.npz')
    np.testing.assert_allclose(data['xyz_mm'][mapping], abq_field['xyz_mm'], atol=5e-7, rtol=0.)
    diff = data['u_mm'][mapping] - abq_field['u_mm']
    rms = float(np.sqrt(np.mean(np.sum(diff**2,axis=1))))
    ref_rms = float(np.sqrt(np.mean(np.sum(abq_field['u_mm']**2,axis=1))))
    work = float(np.trapezoid(f, x * 10))
    metrics['cases'][c] = {
        'endpoint_force_N':last['force_N'],'endpoint_relative_bias':last['force_N']/ref['path'][-1]['force_N']-1,
        'loading_RMS_over_fixed_Abaqus_cap_max':float(np.sqrt(np.mean((force-ref_force)**2))/den),
        'maximum_force_difference_over_fixed_Abaqus_cap_max':float(np.max(abs(force-ref_force))/den),
        'RF_U_work_N_mm':work,'work_relative_bias':work/ref_work-1,
        'endpoint_rotation_degree':float(np.degrees(last['top_cross_section_rotation_rad'])),
        'endpoint_force_lever_mm':last['force_lever_from_current_mid_mm'],
        'endpoint_tip_U1_mm':last['tip_mid_U1_mm'],'endpoint_tip_U3_mm':last['tip_mid_U3_mm'],
        'physical_node_displacement_RMS_difference_mm':rms,'physical_node_displacement_RMS_relative_difference':rms/ref_rms,
        'maximum_reaction_imbalance_over_endpoint_force':max(abs(r['bottom_force_N']-r['force_N']) for r in rows)/last['force_N'],
        'maximum_free_residual_l2_N':max(r['free_residual_l2_N'] for r in rows),
        'minimum_saved_actual_J':min(r['J_min'] for r in rows),
        'maximum_enabled_quadrature_penetration_mm':max(r['max_enabled_quadrature_penetration_mm'] for r in rows),
        'maximum_physical_top_node_penetration_mm':max(r['max_physical_top_node_penetration_mm'] for r in rows),
        'endpoint_internal_plus_contact_energy_N_mm':last['internal_energy_N_mm']+last['contact_energy_N_mm'],
        'endpoint_RF_U_work_minus_energy_over_work':(work-last['internal_energy_N_mm']-last['contact_energy_N_mm'])/work,
        'accepted_static_increments':len(rows)-1,'maximum_Newton_corrections_per_increment':max(r['Newton_corrections'] for r in rows),
        'wall_seconds':result['wall_seconds'],
    }
    m = metrics['cases'][c]
    m['provisional_local_forward_gate_passed'] = abs(m['endpoint_relative_bias']) <= .05 and m['loading_RMS_over_fixed_Abaqus_cap_max'] <= .05 and m['minimum_saved_actual_J']>0 and m['maximum_free_residual_l2_N']<=2e-9
    with (A / f'{c}_path.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
metrics['between_JAX_cases']={'gauss_over_nodal_minus_one_endpoint':metrics['cases']['gauss']['endpoint_force_N']/metrics['cases']['nodal']['endpoint_force_N']-1,
    'changed_factor':'Only contact sampling and reference area weights; shared HEX27 volume, material, base BC, kn=320 and 40 static increments.'}
protected=json.loads((B/'input/jax_protected_before.json').read_text())
bad=[n for n,h in protected.items() if not (R/n).exists() or hashlib.sha256((R/n).read_bytes()).hexdigest()!=h]
assert not bad,bad
write=lambda p,v:p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
write(A/'metrics.json',metrics);write(A/'integrity.json',{'protected_files':len(protected),'mismatches':bad})
with (A/'comparison.csv').open('w',newline='') as stream:
    writer=csv.writer(stream);writer.writerow(['compression','Abaqus_cap_force_N','JAX_nodal_force_N','JAX_gauss_force_N']);writer.writerows(zip(grid,*curves.values()))

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
colors={'Abaqus full cap':'#333333','JAX nodal':'#1e659b','JAX gauss':'#b45c30'}
fig,axes=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
for name,f in curves.items():axes[0].plot(grid*100,f*1000,color=colors[name],lw=2,label=name)
axes[0].plot([r['compression']*100 for r in rim['path']],[r['force_N']*1000 for r in rim['path']],color='#999999',ls=':',label='Abaqus centerline (different model)')
axes[0].set(xlabel='Compression / height (%)',ylabel='Platen force (mN)',title='Same finite-width physical cut face, local solid');axes[0].legend(frameon=False,fontsize=8)
for name in ['JAX nodal','JAX gauss']:axes[1].plot(grid*100,100*(curves[name]-ref_force)/den,color=colors[name],lw=2,label=name)
axes[1].axhline(0,color='#555555',lw=1)
axes[1].set(xlabel='Compression / height (%)',ylabel='Force difference / fixed Abaqus full-cap max (%)',title='Difference uses one fixed denominator');axes[1].legend(frameon=False)
for ax in axes:ax.grid(alpha=.2)
fig.savefig(A/'local_JAX_Abaqus_force.png',dpi=190);plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
for c in ['nodal','gauss']:
    s=json.loads((B/f'jax/{c}/result.json').read_text());x=np.array([r['compression'] for r in s['path']])*100
    axes[0].plot(x,[np.degrees(r['top_cross_section_rotation_rad']) for r in s['path']],color=colors['JAX '+c],label='JAX '+c)
    axes[1].plot(x,[r['max_physical_top_node_penetration_mm']*1000 for r in s['path']],color=colors['JAX '+c],label='All top nodes: '+c)
    axes[1].plot(x,[r['max_enabled_quadrature_penetration_mm']*1000 for r in s['path']],color=colors['JAX '+c],ls='--',label='Enabled contact points: '+c)
axes[0].plot(ref_x*100,[np.degrees(r['top_cross_section_rotation_rad']) for r in ref['path']],color=colors['Abaqus full cap'],ls='--',label='Abaqus full cap')
axes[0].set(xlabel='Compression / height (%)',ylabel='Top section rotation (degree)',title='End rotation also agrees')
axes[1].set(xlabel='Compression / height (%)',ylabel='Penetration into platen (micrometre)',title='Gauss sampling can miss the outer edge')
for ax in axes:ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=8)
fig.savefig(A/'local_JAX_Abaqus_rotation_penetration.png',dpi=190);plt.close(fig)
print(json.dumps({'cases':metrics['cases'],'protected_files':len(protected),'mismatches':bad},indent=2))

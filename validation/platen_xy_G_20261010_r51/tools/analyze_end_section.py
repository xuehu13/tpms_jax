"""Analyze the two local same-solid Abaqus contact-support tests; no G solves."""
from pathlib import Path
import csv, hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = Path('/home/xuehu/projects/tpms_jax')
D = R / 'validation/platen_xy_G_20261010_r51'
B = D / 'diagnostics/end_section_benchmark_20261010'
A = B / 'analysis'
results = {c: json.loads((B / f'abaqus/{c}/result.json').read_text()) for c in ('cap', 'rim')}
protocol = json.loads((B / 'input/protocol.json').read_text())
grid = np.linspace(*protocol['comparison_grid_compression'])
curves = {}
metrics = {}
for c, result in results.items():
    assert result['complete']
    rows = result['path']
    x = np.array([v['compression'] for v in rows])
    f = np.array([v['force_N'] for v in rows])
    assert np.all(np.diff(x) > 0)
    curves[c] = np.interp(grid, x, f)
    last = rows[-1]
    metrics[c] = {
        'endpoint_force_N': last['force_N'],
        'external_work_path_trapezoid_N_mm': float(np.trapezoid(f, x * 10)),
        'endpoint_ALLWK_N_mm': result['energy']['ALLWK'][-1][1],
        'endpoint_ALLIE_N_mm': result['energy']['ALLIE'][-1][1],
        'IE_minus_WK_over_WK': result['energy']['ALLIE'][-1][1] / result['energy']['ALLWK'][-1][1] - 1,
        'endpoint_rotation_degrees': float(np.degrees(last['top_cross_section_rotation_rad'])),
        'endpoint_force_lever_mm': last['contact_force_lever_from_current_mid_mm'],
        'endpoint_tip_lateral_mm': last['tip_mid_U1_mm'],
        'endpoint_local_normal_moment_N_mm': last['force_N'] * last['contact_force_lever_from_current_mid_mm'],
        'maximum_contact_node_penetration_mm': result['maximum_contact_node_penetration_mm'],
        'maximum_physical_cap_penetration_mm': result['maximum_physical_cap_penetration_mm'],
        'maximum_reaction_imbalance_over_endpoint_force': result['maximum_reaction_imbalance_N'] / last['force_N'],
        'maximum_contact_force_discrepancy_over_endpoint_force': max(abs(v['contact_normal_force_N'] - v['force_N']) for v in rows) / last['force_N'],
        'endpoint_force_centroid_crosscheck_mm': abs(last['contact_force_x_centroid_mm'] - last['base_reaction_inferred_force_x_mm']),
        'saved_frames': len(rows),
        'solve_wall_seconds': json.loads((B / f'abaqus/{c}/solve_receipt.json').read_text())['wall_seconds'],
        'datacheck_wall_seconds': json.loads((B / f'abaqus/{c}/datacheck_receipt.json').read_text())['wall_seconds'],
    }
    with (A / f'{c}_path.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

metrics['comparison'] = {
    'cap_over_rim_minus_one_endpoint': metrics['cap']['endpoint_force_N'] / metrics['rim']['endpoint_force_N'] - 1,
    'rim_over_cap_minus_one_endpoint': metrics['rim']['endpoint_force_N'] / metrics['cap']['endpoint_force_N'] - 1,
    'RMS_difference_over_cap_maximum': float(np.sqrt(np.mean((curves['cap'] - curves['rim']) ** 2)) / np.max(curves['cap'])),
    'cap_over_rim_minus_one_external_work': metrics['cap']['endpoint_ALLWK_N_mm'] / metrics['rim']['endpoint_ALLWK_N_mm'] - 1,
    'grid_points': len(grid),
    'geometry_and_material_same': True,
    'is_G_test': False,
    'is_JAX_validation': False,
    'magnitude_transfer_to_G_allowed': False,
    'rim_contact_is_physical_full_cap': False,
}
common_force = metrics['rim']['endpoint_force_N']
cap_rows = results['cap']['path']
metrics['comparison']['equal_force_N'] = common_force
metrics['comparison']['cap_compression_at_equal_force'] = float(np.interp(common_force, [v['force_N'] for v in cap_rows], [v['compression'] for v in cap_rows]))
metrics['comparison']['cap_rotation_degrees_at_equal_force'] = float(np.degrees(np.interp(common_force, [v['force_N'] for v in cap_rows], [v['top_cross_section_rotation_rad'] for v in cap_rows])))
metrics['working_gate'] = {
    'support_difference_above_5pct': abs(metrics['comparison']['cap_over_rim_minus_one_endpoint']) > .05,
    'both_static_balance_below_1e_minus3': all(v['maximum_reaction_imbalance_over_endpoint_force'] < 1e-3 for v in [metrics['cap'], metrics['rim']]),
    'both_selected_contact_node_penetration_below_1e_minus4_mm': all(v['maximum_contact_node_penetration_mm'] < 1e-4 for v in [metrics['cap'], metrics['rim']]),
    'rim_has_unconstrained_cap_penetration': metrics['rim']['maximum_physical_cap_penetration_mm'] > 1e-4,
}
metrics['working_gate']['causal_mechanism_demonstrated_in_local_solid'] = all(metrics['working_gate'][n] for n in ['support_difference_above_5pct', 'both_static_balance_below_1e_minus3', 'both_selected_contact_node_penetration_below_1e_minus4_mm'])
(A / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False) + '\n')
with (A / 'comparison.csv').open('w', newline='') as stream:
    writer = csv.writer(stream); writer.writerow(['compression', 'cap_force_N', 'rim_force_N', 'difference_N'])
    writer.writerows(zip(grid, curves['cap'], curves['rim'], curves['cap'] - curves['rim']))

# Verify that all non-contact input records are identical, not only metadata.
def remove_contact_surface(p):
    keep = []; active = False
    for line in p.read_text().splitlines():
        if line.startswith('*'):
            active = line.upper().startswith('*SURFACE,TYPE=NODE,NAME=END_CONTACT')
        if not active: keep.append(line)
    return keep
same_input = remove_contact_surface(B / 'abaqus/cap.inp') == remove_contact_surface(B / 'abaqus/rim.inp')
assert same_input
before = json.loads((B / 'input/frozen_before.json').read_text())
mismatches = [n for n, h in before.items() if not (R / n).exists() or hashlib.sha256((R / n).read_bytes()).hexdigest() != h]
assert not mismatches, mismatches
(A / 'integrity.json').write_text(json.dumps({'protected_files': len(before), 'mismatches': mismatches, 'non_contact_INP_records_identical': same_input}, indent=2) + '\n')

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
colors = {'cap': '#1f5e9b', 'rim': '#b75b30'}
names = {'cap': 'Full cut-face contact', 'rim': 'Centerline-only contact'}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
for c in results:
    axes[0].plot(100 * grid, 1000 * curves[c], color=colors[c], label=names[c], lw=2)
    rows = results[c]['path']
    axes[1].plot([v['compression'] * 100 for v in rows], [np.degrees(v['top_cross_section_rotation_rad']) for v in rows], color=colors[c], label=names[c], lw=2)
axes[0].set(xlabel='Compression / height (%)', ylabel='Platen force (mN)', title='Same Abaqus solid; contact support changes only')
axes[0].legend(frameon=False)
bias = metrics['comparison']['cap_over_rim_minus_one_endpoint'] * 100
axes[0].text(.04, .84, f'Full cut face: +{bias:.1f}% force at 2%', transform=axes[0].transAxes)
axes[1].set(xlabel='Compression / height (%)', ylabel='Top section rotation (degree)', title='Both can rotate; finite width creates a moment')
for ax in axes: ax.grid(alpha=.22)
fig.savefig(A / 'same_solid_force_rotation.png', dpi=190); plt.close(fig)

mesh = np.load(B / 'input/solid_mesh.npz');idx = mesh['indices'];xyz = mesh['xyz_mm']
fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), constrained_layout=True)
for c in results:
    data = np.load(B / f'abaqus/{c}/field.npz');u = data['u_mm']
    # labels are unchanged and in increasing order, while npz node indices are zero based.
    for edge in (0, 8):
        ids = np.flatnonzero((idx[:, 0] == edge) & (idx[:, 1] == 0));ids = ids[np.argsort(idx[ids, 2])]
        pts = xyz[ids] + u[ids]
        axes[0].plot(pts[:, 0], pts[:, 2], color=colors[c], lw=1.5, label=names[c] if edge == 0 else None)
    top = np.flatnonzero((idx[:, 2] == 80) & (idx[:, 1] == 0));top = top[np.argsort(idx[top, 0])]
    points = xyz[top] + u[top]; mid = results[c]['path'][-1]
    # Relative to the current section midpoint; platen remains at z=9.8 mm.
    xmid = 6 + mid['tip_mid_U1_mm']
    ax = axes[1 if c == 'cap' else 2]
    ax.axhline(0, color='#333333', lw=3, label='Rigid platen')
    ax.plot(points[:, 0] - xmid, points[:, 2] - 9.8, '-o', ms=4, color=colors[c], label='Deformed solid cut face')
    ids_contact = np.array(json.loads((B / f'input/{c}.json').read_text())['contact_nodes']) - 1
    p = xyz[ids_contact] + u[ids_contact]
    # Show only y=0 nodes in this x-z section.
    on_y0 = idx[ids_contact, 1] == 0
    ax.scatter(p[on_y0, 0] - xmid, p[on_y0, 2] - 9.8, facecolors='none', edgecolors='#161616', s=60, label='Enabled contact nodes', zorder=5)
    lever = mid['contact_force_lever_from_current_mid_mm']
    ax.annotate('', xy=(lever, -.003), xytext=(lever, .038), arrowprops={'arrowstyle': '-|>', 'color': colors[c], 'lw': 2})
    ax.set(xlabel='x relative to section midpoint (mm)', ylabel='z relative to platen (mm)', title=names[c], xlim=(-.3, .3), ylim=(-.085, .055))
    ax.text(.03, .04, f'Force lever: {lever:.3f} mm\nMax solid penetration: {metrics[c]["maximum_physical_cap_penetration_mm"]:.3f} mm', transform=ax.transAxes, fontsize=9)
    ax.grid(alpha=.2)
axes[0].axhline(9.8, color='#333333', lw=2)
axes[0].set(xlabel='x (mm)', ylabel='z (mm)', title='Local inclined wall, actual deformation')
axes[0].set_aspect('equal');handles, _ = axes[0].get_legend_handles_labels();axes[0].legend(handles, ['Full cut face', 'Centerline'], frameon=True, framealpha=.95, edgecolor='none', fontsize=8, loc='upper left')
axes[1].legend(frameon=False, fontsize=7, loc='upper right')
fig.savefig(A / 'same_solid_contact_geometry.png', dpi=190); plt.close(fig)
print(json.dumps({'comparison': metrics['comparison'], 'gate': metrics['working_gate'], 'integrity': {'files': len(before), 'mismatches': mismatches}}, indent=2))

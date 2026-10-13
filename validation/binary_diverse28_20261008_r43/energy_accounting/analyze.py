from pathlib import Path
import json,hashlib
R=Path('/home/xuehu/projects/tpms_jax');D=Path(__file__).resolve().parent
old=R/'validation/initial_bias_reassessment_20261008_r42/analyze.py'
s=old.read_text()
changes=[
("D = R/'validation/initial_bias_reassessment_20261008_r42'", "D = R/'validation/binary_diverse28_20261008_r43/energy_accounting'"),
('validation/geometry_transfer_20261006_r15/gauss_field.npz','validation/large_compression_20261005_r6/quadratic_candidate/gauss_field.npz'),
('validation/local_quadrature_20261008_r33/selected_cells.npy','validation/binary_diverse28_20261008_r43/selected_cells.npy'),
('assert len(sel) == 1941','assert len(sel) == 3896'),
('initial_tangent_20261008_r30','initial_diverse28_20261008_r41'),
('local_reequilibrium_20261008_r34','binary_diverse28_20261008_r43/smooth_mixed12'),
('binary_occupancy_20261008_r40/original27','binary_diverse28_20261008_r43/original27'),
('binary_occupancy_20261008_r40/mixed12','binary_diverse28_20261008_r43/binary_mixed12'),
("surface = PeriodicSurfaceDistance(c['surface_vertices'], c['surface_triangles'])", "geometry_cache = np.load(R/'validation/thin_target_20261004_r5/gauss_field.npz')\n        surface = PeriodicSurfaceDistance(geometry_cache['surface_vertices'], geometry_cache['surface_triangles'])"),
('diverse_04: same-rule','diverse_28: same-rule'),
('ax.set_ylim(-5.2, 4.3)','ax.set_ylim(-6,6)')]
for before,after in changes:
 assert s.count(before)==1,(before,s.count(before))
 s=s.replace(before,after)
(D/'source_adaptation.json').write_text(json.dumps({'source_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),'compiled_source_sha256':hashlib.sha256(s.encode()).hexdigest(),'production_changed':False,'new_equilibrium':False},indent=2)+'\n')
exec(compile(s,str(D/'adapted_analysis.py'),'exec'),{'__file__':str(D/'adapted_analysis.py'),'__name__':'__main__'})

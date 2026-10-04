"""Field inspection without a FEM solve; preserves the recorded M64 input."""
from pathlib import Path
import hashlib
import json
import sys
ROOT = Path('/home/xuehu/projects/tpms_jax')
sys.path.insert(0, str(ROOT))
import numpy as np
import jax.numpy as jnp
from voxel_field import periodic_trilinear
from geometry import density

OUT = ROOT / 'validation/near_term_20261003/step2'
row = json.loads((OUT / 'M64_N64_fixed.json').read_text())
path = OUT / 'M64_N64_fixed.input.npy'
assert hashlib.sha256(path.read_bytes()).hexdigest() == row['input_representation']['input_sha256']
values = jnp.asarray(np.load(path))
N = 64
gauss_offsets = .5*(1+jnp.array([-1., 1.])/jnp.sqrt(3.))
axis = ((jnp.arange(N)[:,None]+gauss_offsets)/N).ravel()
points = jnp.stack(jnp.meshgrid(axis, axis, axis, indexing='ij'), axis=-1)
analytic = np.asarray(density(points, .541062, 40.))
mapped = np.asarray(periodic_trilinear(values, points))
probe = jnp.array([[0., .3, 1.], [-.01, .9, 1.2], [.31, .43, .72]])
periodic_error = float(jnp.max(jnp.abs(periodic_trilinear(values, probe)-periodic_trilinear(values, probe+jnp.array([1., -1., 2.])))))
assert mapped.min() >= 0 and mapped.max() <= 1
assert periodic_error <= 1e-12
assert abs(float(np.mean(np.abs(mapped-analytic)))-row['input_representation']['weighted_L1_field_error']) <= 1e-12
assert abs(float(mapped.mean())-row['vf_int']) <= 1e-12
audit = {'FEM_solves': 0, 'input_sha256': row['input_representation']['input_sha256'],
         'input_min': float(values.min()), 'input_max': float(values.max()),
         'mapped_Gauss_min': float(mapped.min()), 'mapped_Gauss_max': float(mapped.max()),
         'periodic_translation_error': periodic_error,
         'Gauss_field_matches_saved_weighted_L1_and_volume': True,
         'Gauss_definition': 'Same 2x2x2 HEX8 points, reordered as tensor grid; uniform JxW',
         'transition_definition': '.05 < phi < .95; fractions are cell-volume fractions',
         'analytic_transition_volume_fraction': float(np.mean((analytic > .05) & (analytic < .95))),
         'mapped_transition_volume_fraction': float(np.mean((mapped > .05) & (mapped < .95))),
         'interpretation_limit': 'Changed transition region is observed; its isolated mechanical contribution has not been quantified'}
(OUT / 'field_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
print(json.dumps(audit), flush=True)

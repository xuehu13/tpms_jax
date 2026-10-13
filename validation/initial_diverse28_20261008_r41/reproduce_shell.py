"""Recreate r41's frozen shell deck, or verify its existing bytes."""
from pathlib import Path
import hashlib,json
R=Path('/home/xuehu/projects/tpms_jax');D=Path(__file__).resolve().parent/'abaqus'
base=R/'validation/large_compression_20261005_r6/abaqus/explicit_T0p040'
original=(base/'thin_shell.inp').read_text(encoding='utf-8')
body=original.split('*Amplitude, name=MACRO')[0]
old='*Hyperelastic, Neo Hooke\n1.923076923076923, 0.24'
assert body.count(old)==1
body=body.replace(old,'*Elastic\n10., 0.3')
tail=(R/'validation/initial_tangent_20261008_r30/abaqus/shell.inp').read_text().split('*Step, name=COMPRESSION')[1]
deck=body+'*Step, name=COMPRESSION'+tail
assert deck.count('*Equation')==2196 and '*Dynamic' not in deck
expected=hashlib.sha256(deck.encode()).hexdigest()
metadata=json.loads((D/'shell_input.json').read_text())
assert expected==metadata['shell_inp_sha256']
p=D/'shell.inp'
if p.exists():
    assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
    print('Existing frozen deck verified; no rewrite:',expected)
else:
    p.write_text(deck,encoding='utf-8')
    print('Recreated from canonical geometry/equations:',expected)

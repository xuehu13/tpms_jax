from pathlib import Path
import json,hashlib,subprocess
P=Path('/home/xuehu/projects/tpms_jax');O=P/'validation/mechanics_trust_20261003_r4/step2'
A=Path('/mnt/e/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus/mechanics_trust_20261004_r4')
def load(f): return json.loads(f.read_text(encoding='utf-8-sig'))
for n in (48,64):
    row=load(O/f'primitive/N{n}.json')
    print('BACKGROUND',n,json.dumps(row,ensure_ascii=False))
for g in (32,48):
    folder=A/f'primitive/G{g}';expected=load(next(folder.glob('*.expected.json')));case=expected['case']
    print('EXPECTED',g,json.dumps(expected))
    for suffix in ('.acceptance.json','.diagnostics.json','.runtime.json'):
        f=folder/'work'/(case+suffix)
        if f.exists(): print(suffix,g,json.dumps(load(f)))
    print('WORK_FILES',g,[f.name for f in (folder/'work').iterdir()])
print('OLD_G32_THIN',json.dumps(load(next((A/'thin_gyroid/G32').glob('*.expected.json')))))
print('PRESERVATION_KEYS',list(load(O/'preservation_before.json')))
print('GIT_HEAD',subprocess.check_output(['git','-C',str(P),'rev-parse','HEAD'],text=True).strip())
print('NEW_LOCKS',[str(f) for f in A.rglob('*.lck')])
(O/'collect.py').write_bytes(Path(__file__).read_bytes())

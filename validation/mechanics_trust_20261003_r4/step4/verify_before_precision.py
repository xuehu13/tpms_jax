from pathlib import Path
import json,hashlib,subprocess
P=Path('/home/xuehu/projects/tpms_jax');O=P/'validation/mechanics_trust_20261003_r4/step4'
def sha(f):
    h=hashlib.sha256()
    with f.open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
old=json.loads((O/'preservation_before.json').read_text());allowed={str(P/'hyperelastic_fem.py')}
differences={f:{'before':v,'current':sha(Path(f))} for f,v in old['sha256'].items() if sha(Path(f))!=v}
assert set(differences)==allowed
src=json.loads((O/'source_at_run.json').read_text());assert all(sha(P/n)==v for n,v in src.items())
assert subprocess.check_output(['git','-C',str(P),'rev-parse','HEAD'],text=True).strip()==old['HEAD']
evidence={'protected_files':len(old['sha256']),'only_numerical_change':'hyperelastic_fem.py density-energy extension',
          'prior_stage_and_other_sources_unchanged':True,'differences':differences,'source_at_run_unchanged':True,'tests':148}
(O/'pre_precision_integrity.json').write_text(json.dumps(evidence,indent=2)+'\n');(O/'verify_before_precision.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(evidence,indent=2))

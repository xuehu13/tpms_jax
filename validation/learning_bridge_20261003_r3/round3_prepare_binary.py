from pathlib import Path
import json,time,sys,subprocess,shutil,hashlib
R=Path('/home/xuehu/projects/tpms_jax');O=R/'validation/learning_bridge_20261003_r3';S=O/'step2'
assert json.loads((S/'forward_summary.json').read_text())['background_passed']
label=sys.argv[1];G=int(sys.argv[2]);q=next(a['theta'] for a in json.loads((S/'anchors.json').read_text()) if a['label']==label)
assert label in ['ax035','az035'] and G in [32,48]
out=Path('/mnt/e/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus/learning_bridge_20261003_r3')/label/f'G{G}'
assert not out.exists(),out
start=time.perf_counter()
log=S/f'{label}_G{G}_prepare.console.txt'
with log.open('x') as f:
 p=subprocess.run([sys.executable,str(R/'scripts/prepare_abaqus_binary.py'),'--output',str(out),'--n',str(G),'--width-parameters',*map(str,q)],cwd=R,stdout=f,stderr=subprocess.STDOUT,timeout=1800)
elapsed=time.perf_counter()-start
ledger_path=O/'abaqus_execution.json';ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else []
row={'kind':'prepare','case':f'{label}_G{G}','seconds':elapsed,'exit_code':p.returncode,'theta':q,'directory':str(out)};ledger.append(row);ledger_path.write_text(json.dumps(ledger,indent=2)+'\n')
assert p.returncode==0,log
scripts=out/'scripts';scripts.mkdir()
for name in ['extract_abaqus_binary.py','extract_uniform_baseline.py','run_abaqus_binary.ps1']:
 shutil.copy2(R/'scripts'/name,scripts/name)
expected=next(out.glob('*.expected.json'));m=json.loads(expected.read_text());assert m['c']==q
record={'theta':q,'G':G,'directory':str(out),'expected_file':str(expected),'source_sha256':{n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in ['binary_gyroid.py','scripts/prepare_abaqus_binary.py','scripts/extract_abaqus_binary.py','scripts/run_abaqus_binary.ps1']},'geometry':m['geometry'],'prepare_seconds':elapsed,'case':m['case']}
(S/f'{label}_G{G}_prepared.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)

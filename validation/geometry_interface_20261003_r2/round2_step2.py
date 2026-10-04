from pathlib import Path
import json,sys,subprocess,time,hashlib,os,shutil
R=Path('/home/xuehu/projects/tpms_jax');O=R/'validation/geometry_interface_20261003_r2';S=O/'step2';S.mkdir(exist_ok=False)
choice=json.loads((O/'step1/summary.json').read_text())['selected_M']
def run(cmd,log,kind,case,timeout=1200):
 start=time.perf_counter()
 with log.open('w') as f:result=subprocess.run(cmd,cwd=R,env={**os.environ,'XLA_PYTHON_CLIENT_PREALLOCATE':'false'},stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
 elapsed=time.perf_counter()-start
 ledger=json.loads((O/'execution.json').read_text());ledger.append({'stage':2,'case':case,'kind':kind,'command':cmd,'seconds':elapsed,'exit_code':result.returncode});(O/'execution.json').write_text(json.dumps(ledger,indent=2)+'\n')
 assert result.returncode==0,log
 print(json.dumps({'completed':case,'kind':kind,'seconds':elapsed}),flush=True)
for label,c in [('c048',.48),('c060',.60)]:
 for N,M in [(48,None),(64,None)]+([(64,choice)] if choice else []):
  name=f'{label}_N{N}'+(f'_M{M}' if M else '_analytic')
  cmd=[sys.executable,str(R/'scripts/capture_binary_projection_reference.py'),'--N',str(N),'--beta','40','--emin-ratio','1e-4','--lateral','fixed','--solver','petsc','--c',str(c),'--out',str(S/(name+'.json'))]
  if M:cmd+=['--voxel-M',str(M),'--field-kind','implicit']
  run(cmd,S/(name+'.console.txt'),'large_forward',name)
 # Independent conforming models; package directory carries the c label.
 for G in (32,48):
  package=Path('/mnt/e/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus/geometry_interface_20261003_r2')/label/f'G{G}'
  assert not package.exists(),package
  cmd=[sys.executable,str(R/'scripts/prepare_abaqus_binary.py'),'--output',str(package),'--n',str(G),'--lateral','fixed','--c',str(c)]
  run(cmd,S/f'{label}_G{G}.prepare.console.txt','geometry_preparation',f'{label}_G{G}',1800)
  (package/'scripts').mkdir()
  for n in ['run_abaqus_binary.ps1','extract_abaqus_binary.py','extract_uniform_baseline.py']:shutil.copy2(R/'scripts'/n,package/'scripts'/n)
  manifest=next(package.glob('*.expected.json'));shutil.copy2(manifest,S/f'{label}_G{G}.expected.json')
  (S/f'{label}_G{G}.package.json').write_text(json.dumps({'package_wsl':str(package),'package_windows':str(package).replace('/mnt/e/','E:/'),'c':c,'G':G,'case':json.loads(manifest.read_text())['case']},indent=2)+'\n')
  print('READY_ABAQUS '+str(package),flush=True)

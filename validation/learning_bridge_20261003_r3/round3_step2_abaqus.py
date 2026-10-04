from pathlib import Path
import subprocess,json,sys,time
R=Path('/home/xuehu/projects/tpms_jax');O=R/'validation/learning_bridge_20261003_r3';S=O/'step2'
assert json.loads((S/'forward_summary.json').read_text())['background_passed']
ps='/mnt/c/Users/xuehu/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/powershell/pwsh.exe'
for label in ['ax035','az035']:
 for G in [32,48]:
  record=S/f'{label}_G{G}_prepared.json'
  if not record.exists():
   result=subprocess.run([sys.executable,str(O/'round3_prepare_binary.py'),label,str(G)],cwd=R,timeout=1805)
   assert result.returncode==0
  prepared=json.loads(record.read_text());root=Path(prepared['directory']);case=prepared['case']
  acceptance=root/'work'/f'{case}.acceptance.json'
  if acceptance.exists():
   report=json.loads(acceptance.read_text(encoding='utf-8-sig'));assert report['status']=='ok'
   (S/f'{label}_G{G}_acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
   (S/f'{label}_G{G}_diagnostics.json').write_text((root/'work'/f'{case}.diagnostics.json').read_text(encoding='utf-8-sig'))
   print('Reuse completed reference '+label+f'_G{G}',flush=True);continue
  win='E:\\ABAQUS\\2026temp\\Abaqus_Work\\tpms_jax_abaqus\\learning_bridge_20261003_r3\\'+label+'\\G'+str(G)
  ledger=json.loads((O/'abaqus_execution.json').read_text());used=sum(r['seconds'] for r in ledger)
  assert used<5400
  limit=min(1800-prepared['prepare_seconds'],5400-used);assert limit>0
  extract_only=(root/'work'/f'{case}.odb').exists()
  start=time.perf_counter();command=[ps,'-NoProfile','-ExecutionPolicy','Bypass','-File',win+'\\scripts\\run_abaqus_binary.ps1','-PackageDirectory',win,'-Cases',case]
  if extract_only:command.append('-ExtractOnly')
  console=S/f'{label}_G{G}_run.console.txt'
  if console.exists():console=S/f'{label}_G{G}_retry_extract.console.txt'
  try:
   with console.open('x') as f:
    proc=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=limit)
   exitcode=proc.returncode
  except subprocess.TimeoutExpired:
   # Terminate only the jobs created by this case, in their exact work directory.
   for name in [case,case+'_check']:
    text="Set-Location -LiteralPath '"+win+"\\work'; & 'E:\\ABAQUS\\2026\\Commands\\abaqus.bat' terminate job="+name
    subprocess.run([ps,'-NoProfile','-Command',text],capture_output=True,timeout=60)
   exitcode=124
  elapsed=time.perf_counter()-start
  acceptance=root/'work'/f'{case}.acceptance.json'
  report=json.loads(acceptance.read_text(encoding='utf-8-sig')) if acceptance.exists() else None
  row={'kind':'extract_only' if extract_only else 'analysis_and_datacheck_extract','case':f'{label}_G{G}','seconds':elapsed,'exit_code':exitcode,'Abaqus_analysis_count':0 if extract_only else int((root/'work'/f'{case}.sta').exists()),'datacheck_count':0 if extract_only else int((root/'work'/f'{case}_check.dat').exists()),'status':report.get('status') if report else None}
  ledger=json.loads((O/'abaqus_execution.json').read_text());ledger.append(row);(O/'abaqus_execution.json').write_text(json.dumps(ledger,indent=2)+'\n')
  print(json.dumps(row),flush=True)
  assert exitcode==0 and report and report['status']=='ok',S/f'{label}_G{G}_run.console.txt'
  (S/f'{label}_G{G}_acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
  diagnostics=root/'work'/f'{case}.diagnostics.json'
  (S/f'{label}_G{G}_diagnostics.json').write_text(diagnostics.read_text(encoding='utf-8-sig'))
print('Four binary references completed; do not start learning before physical-span decision.',flush=True)

"""Sequential prelocked paths; reuse the existing job diagnostic function."""
from pathlib import Path
import os,json,subprocess,sys,time,hashlib,shutil
P=Path('/home/xuehu/projects/tpms_jax');O=P/'validation/mechanics_trust_20261003_r4/step3'
HERE=Path(__file__).resolve().parent
A=Path('/mnt/e/ABAQUS/2026temp/Abaqus_Work/tpms_jax_abaqus/mechanics_trust_20261004_r4/step3_uniform')
plan=json.loads((O/'plan.json').read_text());ledger=[]
assert not (O/'execution.json').exists()
def dump(f,v):f.write_text(json.dumps(v,indent=2)+'\n')
def win(f):return str(f).replace('/mnt/e/','E:/').replace('/','\\')
env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','XLA_PYTHON_CLIENT_PREALLOCATE':'false','JAX_PLATFORMS':'cpu'}
python=str(P/'.pixi/envs/default/bin/python')
# NumPy-version-independent trapezoidal extraction; no physical definition changed.
extractor=P/'scripts/extract_finite_strain.py'
(O/'extractor_before_numpy_compat.py').write_bytes(extractor.read_bytes())
extractor.write_bytes((HERE/'newfiles/scripts/extract_finite_strain.py').read_bytes())
dump(O/'source_at_run.json',{n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in json.loads((O/'new_source_files.json').read_text())})
(O/'run.py').write_bytes(Path(__file__).read_bytes())
def execute(cmd,log,kind,label,timeout):
    start=time.monotonic()
    with log.open('x') as stream:
        try:r=subprocess.run(cmd,cwd=P,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout);status=r.returncode
        except subprocess.TimeoutExpired:status='timeout'
    record={'kind':kind,'label':label,'command':cmd,'exit_code':status,'wall_seconds':time.monotonic()-start,'log':str(log)}
    ledger.append(record);dump(O/'execution.json',ledger);print(json.dumps(record),flush=True)
    if status!=0:raise RuntimeError('Stage stopped after '+label+'; log retained')
for path in plan['JAX_paths']:
    n=path['N'];inc=path['increment'];label=f'N{n}_d{round(1000*inc):03d}'
    execute([python,str(P/'scripts/finite_strain_uniform.py'),'capture','--N',str(n),'--increment',str(inc),'--out',str(O/(label+'.json'))],
            O/(label+'.log'),'JAX_path',label,180)
runner_source=(P/'scripts/run_abaqus_binary.ps1').read_text()
functions=runner_source[runner_source.index('function Assert-Logs'):runner_source.index('Push-Location -LiteralPath $jobRoot')]
for path in plan['Abaqus_paths']:
    n=path['N'];inc=path['increment'];label=f'N{n}_d{round(1000*inc):03d}';package=A/label
    execute([python,str(P/'scripts/finite_strain_uniform.py'),'prepare','--N',str(n),'--increment',str(inc),'--out',str(package)],
            O/(label+'.prepare.log'),'Abaqus_input_preparation',label,60)
    expected_path=next(package.glob('*.expected.json'));ex=json.loads(expected_path.read_text());case=ex['case']
    scripts=package/'scripts';scripts.mkdir()
    for name in ('extract_finite_strain.py','extract_uniform_baseline.py'):shutil.copy2(P/'scripts'/name,scripts/name)
    header='''param([Parameter(Mandatory=$true)][string]$PackageDirectory,[Parameter(Mandatory=$true)][string]$Case)
$ErrorActionPreference='Stop'
$AbaqusCommand='E:\\ABAQUS\\2026\\Commands\\abaqus.bat'
$root=(Resolve-Path -LiteralPath $PackageDirectory).Path
$jobRoot=Join-Path $root 'work'
New-Item -ItemType Directory -Path $jobRoot -ErrorAction Stop | Out-Null
'''
    body='''Push-Location -LiteralPath $jobRoot
try {
 if($Case -notmatch '^uniform_finite_N4_d(005|010)$'){throw 'Unexpected locked case name'}
 $inputPath=Join-Path $root ($Case+'.inp');$expectedPath=Join-Path $root ($Case+'.expected.json')
 $expected=Get-Content -LiteralPath $expectedPath -Raw | ConvertFrom-Json
 if((Get-FileHash -LiteralPath $inputPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected.input_sha256){throw 'Input hash differs from lock'}
 $check=$Case+'_check'
 & $AbaqusCommand "job=$check" "input=$inputPath" datacheck interactive cpus=2 memory=2gb 2>&1 | Tee-Object -FilePath ($check+'.console.txt')
 if($LASTEXITCODE -ne 0){throw 'Datacheck command failed'}
 Assert-Logs $check $true
 & $AbaqusCommand "job=$Case" "input=$inputPath" interactive cpus=2 memory=2gb 2>&1 | Tee-Object -FilePath ($Case+'.console.txt')
 if($LASTEXITCODE -ne 0){throw 'Analysis command failed'}
 Assert-Logs $Case $false
 & $AbaqusCommand python (Join-Path $root 'scripts\\extract_finite_strain.py') ($Case+'.odb') --expected $expectedPath --out ($Case+'.acceptance.json') 2>&1 | Tee-Object -FilePath ($Case+'.extract.txt')
 if($LASTEXITCODE -ne 0){throw 'Finite-strain physical extraction/check failed'}
 $report=Get-Content -LiteralPath ($Case+'.acceptance.json') -Raw | ConvertFrom-Json
 if($report.status -ne 'ok'){throw 'Physical acceptance failed'}
} finally { Pop-Location }
'''
    script=scripts/'run_locked_uniform.ps1';script.write_text(header+functions+body,encoding='utf-8')
    dump(package/'runner_source.json',{'diagnostics_reused_from':'scripts/run_abaqus_binary.ps1',
         'original_source_sha256':hashlib.sha256((P/'scripts/run_abaqus_binary.ps1').read_bytes()).hexdigest(),
         'actual_runner_sha256':hashlib.sha256(script.read_bytes()).hexdigest()})
    command=['/mnt/c/Users/xuehu/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/powershell/pwsh.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',win(script),'-PackageDirectory',win(package),'-Case',case]
    execute(command,O/(label+'.abaqus.log'),'Abaqus_analysis_and_datacheck',label,600)
dump(O/'paths_complete.json',{'stage':'round4_step3_paths_complete','JAX_paths':3,'Abaqus_analyses':2,'datachecks':2,'steps4_started':False})
print('LOCKED_UNIFORM_PATHS_COMPLETE',flush=True)

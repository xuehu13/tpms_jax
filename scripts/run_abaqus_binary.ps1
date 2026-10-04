param(
 [Parameter(Mandatory=$true)][string]$PackageDirectory,
 [string]$AbaqusCommand='E:\ABAQUS\2026\Commands\abaqus.bat',
 [Parameter(Mandatory=$true)][string[]]$Cases,
 [int]$Cpus=2,
 [string]$Memory='8gb',
 [switch]$ExtractOnly,
 [switch]$QualityDiagnostics
)
$ErrorActionPreference='Stop'
$root=(Resolve-Path -LiteralPath $PackageDirectory).Path
$jobRoot=Join-Path $root 'work'
New-Item -ItemType Directory -Path $jobRoot -Force | Out-Null
function Assert-Logs([string]$Name,[bool]$Datacheck) {
 $files=@(Join-Path $jobRoot ($Name+'.dat'))
 $msg=Join-Path $jobRoot ($Name+'.msg')
 if(Test-Path -LiteralPath $msg){$files += $msg}
 $errors=@(Select-String -LiteralPath $files -Pattern '\*\*\*ERROR')
 $warnings=@(Select-String -LiteralPath $files -Pattern '\*\*\*WARNING')
 $other=@($warnings | Where-Object {$_.Line -notmatch '^\s*\*\*\*WARNING:\s+\d+ elements are distorted\.' -and $_.Line -notmatch '^\s*\*\*\*WARNING: THE MEMORY LIMIT IS INSUFFICENT TO RUN THE PURE THREAD-BASED'})
 if($errors.Count -or $other.Count){throw "$Name has unresolved solver diagnostics"}
 if($Datacheck){
  if(-not(Select-String -LiteralPath $files[0] -Pattern 'ANALYSIS DATACHECK COMPLETE')){throw 'Datacheck incomplete'}
 }else{
  if(-not(Select-String -LiteralPath (Join-Path $jobRoot ($Name+'.sta')) -Pattern 'THE ANALYSIS HAS COMPLETED SUCCESSFULLY')){throw 'Analysis incomplete'}
 }
 $count=0
 foreach($warning in $warnings){if($warning.Line -match '(\d+) elements are distorted'){$count=[int]$Matches[1]}}
 $fallback=@($warnings | Where-Object {$_.Line -match 'THE MEMORY LIMIT IS INSUFFICENT'}).Count -gt 0
 if($fallback -and -not(Select-String -LiteralPath $msg -Pattern 'TO THE HYBRID SOLVER FOR OUT-OF-CORE SOLUTION')){throw 'Memory warning lacks documented solver fallback'}
 $diagnostics=@{case=$Name;errors=$errors.Count;warning_messages=$warnings.Count;distorted_elements=$count;quality_warning=($count -gt 0);out_of_core_solver=$fallback;convergence_claim=$false}
 $diagnostics | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $jobRoot ($Name+'.diagnostics.json')) -Encoding utf8
 Write-Output "${Name}: errors=0; distorted-element count=$count (retained for quality/convergence assessment)"
}
Push-Location -LiteralPath $jobRoot
try{
 foreach($case in $Cases){
  if($case -notmatch '^binary_(uniform_C3D10|(gyroid|primitive)_G[0-9]+_R[01]_C3D10)_(fixed|relaxed_free)$'){throw 'Invalid case name'}
  $inputPath=Join-Path $root ($case+'.inp');$expectedPath=Join-Path $root ($case+'.expected.json')
  $expected=Get-Content -LiteralPath $expectedPath -Raw | ConvertFrom-Json
  if((Get-FileHash -LiteralPath $inputPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected.input_sha256){throw 'Input hash mismatch'}
  if($ExtractOnly){
   if(-not(Test-Path -LiteralPath ($case+'.odb'))){throw 'Completed ODB missing'}
  }else{
   if(Test-Path -LiteralPath ($case+'.odb')){throw 'ODB already exists; use a new job directory'}
   $check=$case+'_check'
   if(Test-Path -LiteralPath ($check+'.dat')){throw 'Datacheck already exists'}
   & $AbaqusCommand "job=$check" "input=$inputPath" datacheck interactive "cpus=$Cpus" "memory=$Memory" 2>&1 | Tee-Object -FilePath ($check+'.console.txt')
   Assert-Logs $check $true
   & $AbaqusCommand "job=$case" "input=$inputPath" interactive "cpus=$Cpus" "memory=$Memory" 2>&1 | Tee-Object -FilePath ($case+'.console.txt')
  }
  Assert-Logs $case $false
  & $AbaqusCommand python (Join-Path $root 'scripts\extract_abaqus_binary.py') ($case+'.odb') --expected $expectedPath 2>&1 | Tee-Object -FilePath ($case+'.extract.txt')
  $report=Get-Content -LiteralPath ($case+'.acceptance.json') -Raw | ConvertFrom-Json
  if($report.status -ne 'ok'){throw 'Physical consistency acceptance failed'}
  if($QualityDiagnostics){
   & $AbaqusCommand python (Join-Path $root 'scripts\abaqus_mesh_quality.py') ($case+'.odb') --expected $expectedPath --out ($case+'.quality.json') --nodal-out ($case+'.nodal.npz') 2>&1 | Tee-Object -FilePath ($case+'.quality.console.txt')
   if($LASTEXITCODE -ne 0){throw 'Quality extraction failed'}
  }
  Write-Output "$case passed physical consistency; convergence assessed separately"
 }
}finally{Pop-Location}

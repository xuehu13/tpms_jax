param([Parameter(Mandatory=$true)][string]$PackageDirectory)
$ErrorActionPreference='Stop'
$manifest=Get-ChildItem -LiteralPath $PackageDirectory -Filter '*.expected.json'
if(@($manifest).Count -ne 1){throw 'Expected one model per package'}
$case=(Get-Content -LiteralPath $manifest.FullName -Raw|ConvertFrom-Json).case
$start=Get-Date
$argsList=@('-NoProfile','-ExecutionPolicy','Bypass','-File',"$PackageDirectory\scripts\run_abaqus_binary.ps1",'-PackageDirectory',$PackageDirectory,'-Cases',$case,'-Cpus','2','-Memory','8gb')
$engine=(Get-Process -Id $PID).Path
$p=Start-Process $engine -WindowStyle Hidden -ArgumentList $argsList -PassThru -RedirectStandardOutput "$PackageDirectory\execution.stdout.txt" -RedirectStandardError "$PackageDirectory\execution.stderr.txt"
if(-not $p.WaitForExit(1800000)){
 & 'E:\ABAQUS\2026\Commands\abaqus.bat' "job=$case" terminate
 Stop-Process -Id $p.Id -ErrorAction SilentlyContinue
 @{case=$case;seconds=((Get-Date)-$start).TotalSeconds;timeout=$true}|ConvertTo-Json|Set-Content -LiteralPath "$PackageDirectory\execution.json" -Encoding utf8
 throw '30 minute job budget reached; named job terminated'
}
$p.Refresh()
@{case=$case;seconds=((Get-Date)-$start).TotalSeconds;exit_code=$p.ExitCode;timeout=$false}|ConvertTo-Json|Set-Content -LiteralPath "$PackageDirectory\execution.json" -Encoding utf8
if($p.ExitCode -ne 0){throw "Abaqus verification failed: $PackageDirectory"}
Write-Output "$PackageDirectory completed and extracted"

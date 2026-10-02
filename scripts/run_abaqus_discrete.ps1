param(
    [Parameter(Mandatory=$true)][string]$PackageDirectory,
    [string]$AbaqusCommand = 'E:\ABAQUS\2026\Commands\abaqus.bat',
    [Parameter(Mandatory=$true)][string[]]$Cases
)
$ErrorActionPreference = 'Stop'
$packageRoot = (Resolve-Path -LiteralPath $PackageDirectory).Path
$jobRoot = Join-Path $packageRoot 'work'
New-Item -ItemType Directory -Path $jobRoot -Force | Out-Null
function Assert-AbaqusLogs([string]$JobName, [bool]$Datacheck) {
    $datPath = Join-Path $jobRoot ($JobName+'.dat')
    $files = @($datPath)
    $msgPath = Join-Path $jobRoot ($JobName+'.msg')
    if (Test-Path -LiteralPath $msgPath) { $files += $msgPath }
    $errors = @(Select-String -LiteralPath $files -Pattern '\*\*\*ERROR')
    $warnings = @(Select-String -LiteralPath $files -Pattern '\*\*\*WARNING')
    $unexpected = @($warnings | Where-Object { $_.Line -notmatch 'THE \*ELEMENT OUTPUT OPTION IS NOT SUPPORTED FOR USER ELEMENTS' })
    if ($errors.Count -gt 0 -or $unexpected.Count -gt 0) { throw "$JobName contains unresolved analysis messages" }
    if ($Datacheck) {
        if (-not (Select-String -LiteralPath $datPath -Pattern 'ANALYSIS DATACHECK COMPLETE')) { throw "$JobName datacheck incomplete" }
    } else {
        $staPath = Join-Path $jobRoot ($JobName+'.sta')
        if (-not (Select-String -LiteralPath $staPath -Pattern 'THE ANALYSIS HAS COMPLETED SUCCESSFULLY')) { throw "$JobName incomplete" }
    }
    Write-Output "$JobName verified: errors=0; documented user-element output warnings=$($warnings.Count)"
}
Push-Location -LiteralPath $jobRoot
$previousOutputDiagnostics = $env:ABA_OUTPUT_DIAGNOSTICS
$env:ABA_OUTPUT_DIAGNOSTICS = 'unlimited'
try {
    foreach ($case in $Cases) {
        if ($case -notmatch '^discrete_gyroid_N[0-9]+_(fixed|relaxed_free)$') { throw 'Invalid case name' }
        $inputPath = Join-Path $packageRoot ($case+'.inp')
        $expectedPath = Join-Path $packageRoot ($case+'.expected.json')
        $expected = Get-Content -LiteralPath $expectedPath -Raw | ConvertFrom-Json
        $inputHash = (Get-FileHash -LiteralPath $inputPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($inputHash -ne $expected.input_sha256) { throw 'Prepared input hash mismatch' }
        if (Test-Path -LiteralPath (Join-Path $jobRoot ($case+'.odb'))) { throw "$case already has an ODB; use a new package directory" }
        $check = $case+'_check'
        if (Test-Path -LiteralPath (Join-Path $jobRoot ($check+'.dat'))) { throw "$check already exists; use a new package directory" }
        & $AbaqusCommand "job=$check" "input=$inputPath" datacheck interactive cpus=1 2>&1 |
            Tee-Object -FilePath (Join-Path $jobRoot ($check+'.console.txt'))
        Assert-AbaqusLogs $check $true
        & $AbaqusCommand "job=$case" "input=$inputPath" interactive cpus=1 2>&1 |
            Tee-Object -FilePath (Join-Path $jobRoot ($case+'.console.txt'))
        Assert-AbaqusLogs $case $false
        $extractPath = Join-Path $packageRoot 'scripts\extract_abaqus_discrete.py'
        & $AbaqusCommand python $extractPath ($case+'.odb') --expected $expectedPath 2>&1 |
            Tee-Object -FilePath (Join-Path $jobRoot ($case+'.extract.txt'))
        $reportPath = Join-Path $jobRoot ($case+'.acceptance.json')
        if (-not (Test-Path -LiteralPath $reportPath)) { throw 'Extraction did not create an acceptance report' }
        $report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
        if ($report.status -ne 'ok') { throw "$case failed numerical acceptance" }
        Write-Output "$case passed all declared acceptance checks"
    }
} finally {
    $env:ABA_OUTPUT_DIAGNOSTICS = $previousOutputDiagnostics
    Pop-Location
}

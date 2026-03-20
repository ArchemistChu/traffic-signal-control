param(
    [Parameter(Mandatory = $true)]
    [string]$ModelPath,

    [Parameter(Mandatory = $true)]
    [string]$EvalJsonPath,

    [string[]]$AdditionalPaths = @(
        "run_vancouver_to_cologne_transfer.ps1",
        "push_transfer_artifacts.ps1"
    ),

    [string]$CommitMessage = "Add Vancouver-to-Cologne transfer model and evaluation outputs.",
    [string]$Remote = "origin",
    [string]$Branch = "main"
)

$ErrorActionPreference = "Stop"

$allPaths = @($ModelPath, $EvalJsonPath) + $AdditionalPaths
$existingPaths = @()
foreach ($path in $allPaths) {
    if (-not [string]::IsNullOrWhiteSpace($path) -and (Test-Path $path)) {
        $existingPaths += $path
    }
}

$existingPaths = $existingPaths | Select-Object -Unique

if (-not $existingPaths -or $existingPaths.Count -eq 0) {
    throw "No existing files were found to stage."
}

Write-Host "== Staging files ==" -ForegroundColor Cyan
$existingPaths | ForEach-Object { Write-Host ("  {0}" -f $_) }
& git add -- @existingPaths
if ($LASTEXITCODE -ne 0) {
    throw "git add failed with exit code $LASTEXITCODE."
}

$staged = git diff --cached --name-only
if (-not $staged) {
    Write-Host "No staged changes found. Nothing to commit." -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "== Creating commit ==" -ForegroundColor Cyan
& git commit -m $CommitMessage
if ($LASTEXITCODE -ne 0) {
    throw "git commit failed with exit code $LASTEXITCODE."
}

Write-Host ""
Write-Host "== Pushing to remote ==" -ForegroundColor Cyan
& git push $Remote $Branch
if ($LASTEXITCODE -ne 0) {
    throw "git push failed with exit code $LASTEXITCODE."
}

Write-Host ""
Write-Host "Push completed." -ForegroundColor Green

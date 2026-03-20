param(
    [int]$TrainEpisodes = 300,
    [int]$EvalEpisodes = 120,
    [int]$TrainDuration = 1200,
    [int]$EvalDuration = 1200,
    [int]$DecisionInterval = 5,
    [double]$ControlledLightsRatio = 0.5,
    [double]$RegionalRewardWeight = 0.01,
    [double]$RegionGridSize = 500.0,
    [int]$Seed = 42,
    [int]$SaveEvery = 25,
    [int]$LanesPerTl = 8,
    [string]$ModelOut = "models/marl_vancouver_shared_dqn_regionaware_v9_safe_transfer.pt",
    [string]$EvalOut = "",
    [switch]$UseEmissionsOutput,
    [switch]$PushToRepo,
    [string]$CommitMessage = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($EvalOut)) {
    $evalSuffix = if ($UseEmissionsOutput) { "_seeded_emissions" } else { "_seeded" }
    $EvalOut = "output/eval_cologne_from_vancouver_v9_safe_transfer_ep{0}{1}.json" -f $TrainEpisodes, $evalSuffix
}

$trainArgs = @(
    "train_marl_los_angeles.py",
    "--dataset", "vancouver",
    "--episodes", $TrainEpisodes,
    "--duration", $TrainDuration,
    "--decision-interval", $DecisionInterval,
    "--controlled-lights-ratio", $ControlledLightsRatio,
    "--regional-reward-weight", $RegionalRewardWeight,
    "--region-grid-size", $RegionGridSize,
    "--seed", $Seed,
    "--save-every", $SaveEvery,
    "--lanes-per-tl", $LanesPerTl,
    "--out", $ModelOut
)

$evalArgs = @(
    "evaluate_marl_osm.py",
    "--dataset", "cologne",
    "--model", $ModelOut,
    "--episodes", $EvalEpisodes,
    "--duration", $EvalDuration,
    "--decision-interval", $DecisionInterval,
    "--controlled-lights-ratio", $ControlledLightsRatio,
    "--lanes-per-tl", $LanesPerTl,
    "--seed", $Seed,
    "--run-id", 1,
    "--out", $EvalOut
)

if ($UseEmissionsOutput) {
    $evalArgs += "--sumo-emissions-output"
}

Write-Host "== Training on Vancouver ==" -ForegroundColor Cyan
Write-Host ("Model output: {0}" -f $ModelOut)
& python @trainArgs
if ($LASTEXITCODE -ne 0) {
    throw "Training failed with exit code $LASTEXITCODE."
}

Write-Host ""
Write-Host "== Evaluating on Cologne ==" -ForegroundColor Cyan
Write-Host ("Evaluation output: {0}" -f $EvalOut)
& python @evalArgs
if ($LASTEXITCODE -ne 0) {
    throw "Evaluation failed with exit code $LASTEXITCODE."
}

Write-Host ""
Write-Host "Done." -ForegroundColor Green
Write-Host ("Model: {0}" -f $ModelOut)
Write-Host ("Evaluation JSON: {0}" -f $EvalOut)

if ($PushToRepo) {
    Write-Host ""
    Write-Host "== Committing and pushing artifacts ==" -ForegroundColor Cyan
    $effectiveCommitMessage = $CommitMessage
    if ([string]::IsNullOrWhiteSpace($effectiveCommitMessage)) {
        $effectiveCommitMessage = "Add Vancouver-to-Cologne transfer model and evaluation outputs."
    }

    $pushArgs = @(
        "-ExecutionPolicy", "Bypass",
        "-File", ".\push_transfer_artifacts.ps1",
        "-ModelPath", $ModelOut,
        "-EvalJsonPath", $EvalOut,
        "-CommitMessage", $effectiveCommitMessage
    )

    & powershell @pushArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Push step failed with exit code $LASTEXITCODE."
    }
} else {
    Write-Host ""
    Write-Host "Push helper example:" -ForegroundColor Yellow
    Write-Host ("powershell -ExecutionPolicy Bypass -File "".\push_transfer_artifacts.ps1"" -ModelPath ""{0}"" -EvalJsonPath ""{1}""" -f $ModelOut, $EvalOut)
}

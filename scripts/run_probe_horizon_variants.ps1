param(
    [int[]]$Steps = @(2, 3, 5)
)

$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
$python = "D:\software\minicoonda\envs\fd\python.exe"
$dataRoot = "C:\Users\admin\Desktop\need_do\new_paper_four\FedPhoenix-main\data"
$partition = Join-Path $project "data\cifar10_100_noniidCase5_beta0.3.json"
$output = Join-Path $project "results\probe_horizon_functional_recovery_2of64"
New-Item -ItemType Directory -Force -Path $output | Out-Null

$processes = @()
foreach ($stepCount in $Steps) {
    if ($stepCount -notin @(2, 3, 5)) {
        throw "Unsupported formal Probe horizon: $stepCount"
    }
    $name = "m2_steps${stepCount}_seed1_2of64_200r"
    $stdout = Join-Path $output "${name}.log"
    $stderr = Join-Path $output "${name}.err.log"
    $arguments = @(
        (Join-Path $project "main_fedrad.py"),
        "--algorithm", "ours",
        "--rounds", "200",
        "--seed", "1",
        "--device", "cuda:0",
        "--num-workers", "0",
        "--eval-every", "1",
        "--no-download",
        "--data-root", $dataRoot,
        "--partition-path", $partition,
        "--output-root", $output,
        "--reset-ratio", "0.03125",
        "--functional-probe-replicates", "2",
        "--probe-steps", [string]$stepCount,
        "--run-name", $name
    )
    $processes += Start-Process -FilePath $python -ArgumentList $arguments `
        -WorkingDirectory $project -WindowStyle Hidden `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
}

for ($index = 0; $index -lt $processes.Count; $index++) {
    [pscustomobject]@{ pid = $processes[$index].Id; steps = $Steps[$index] }
}

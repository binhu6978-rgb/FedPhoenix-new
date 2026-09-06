param(
    [int[]]$Replicates = @(2, 3, 5)
)

$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
$python = "D:\software\minicoonda\envs\fd\python.exe"
$dataRoot = "C:\Users\admin\Desktop\need_do\new_paper_four\FedPhoenix-main\data"
$partition = Join-Path $project "data\cifar10_100_noniidCase5_beta0.3.json"
$output = Join-Path $project "results\multiprobe_functional_recovery_2of64"

New-Item -ItemType Directory -Force -Path $output | Out-Null

foreach ($replicateCount in $Replicates) {
    if ($replicateCount -lt 2) {
        throw "Formal Multi-Probe candidates must use at least two replicates"
    }
    $name = "multiprobe_m${replicateCount}_seed1_2of64_200r"
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
        "--functional-probe-replicates", [string]$replicateCount,
        "--run-name", $name
    )

    & $python @arguments 1> $stdout 2> $stderr
    if ($LASTEXITCODE -ne 0) {
        throw "Multi-Probe M=${replicateCount} failed with exit code $LASTEXITCODE"
    }
}

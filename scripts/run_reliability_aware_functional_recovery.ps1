param(
    [string[]]$Modes = @("half_se_lcb", "one_se_lcb")
)

$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
$python = "D:\software\minicoonda\envs\fd\python.exe"
$dataRoot = "C:\Users\admin\Desktop\need_do\new_paper_four\FedPhoenix-main\data"
$partition = Join-Path $project "data\cifar10_100_noniidCase5_beta0.3.json"
$output = Join-Path $project "results\reliability_aware_functional_recovery_2of64"

New-Item -ItemType Directory -Force -Path $output | Out-Null

foreach ($mode in $Modes) {
    if ($mode -notin @("half_se_lcb", "one_se_lcb")) {
        throw "Unsupported formal reliability mode: $mode"
    }
    $name = "m2_${mode}_seed1_2of64_200r"
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
        "--functional-reliability-mode", $mode,
        "--run-name", $name
    )

    # Windows PowerShell can promote a native process' harmless stderr warning
    # to a terminating NativeCommandError when ErrorActionPreference is Stop.
    # Let Python finish, then use its actual process exit code as authority.
    $ErrorActionPreference = "Continue"
    & $python @arguments 1> $stdout 2> $stderr
    $pythonExitCode = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    if ($pythonExitCode -ne 0) {
        throw "Reliability-aware mode $mode failed with exit code $pythonExitCode"
    }
}

param(
    [int]$WaitForPid = 0
)

$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
$python = "D:\software\minicoonda\envs\fd\python.exe"
$dataRoot = "C:\Users\admin\Desktop\need_do\new_paper_four\FedPhoenix-main\data"
$partition = Join-Path $project "data\cifar10_100_noniidCase5_beta0.3.json"
$output = Join-Path $project "results\m2_5step_1200"
New-Item -ItemType Directory -Force -Path $output | Out-Null

if ($WaitForPid -gt 0 -and (Get-Process -Id $WaitForPid -ErrorAction SilentlyContinue)) {
    Write-Output "WAIT pid=$WaitForPid"
    Wait-Process -Id $WaitForPid
}

$trainers = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -like "python*" -and $_.CommandLine -like "*main_fedrad.py*"
})
if ($trainers.Count -gt 0) {
    throw "Another main_fedrad.py trainer is running; refusing non-serial launch."
}

$name = "m2_steps5_seed1_2of64_1200r_serial"
$stdout = Join-Path $output "${name}.log"
$stderr = Join-Path $output "${name}.err.log"
$arguments = @(
    (Join-Path $project "main_fedrad.py"),
    "--algorithm", "ours",
    "--rounds", "1200",
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
    "--functional-reliability-mode", "mean",
    "--functional-assignment-mode", "hungarian",
    "--probe-steps", "5",
    "--probe-recovery-measurement", "terminal",
    "--probe-query-mode", "legacy",
    "--probe-support-coverage", "fixed",
    "--probe-query-batches", "1",
    "--run-name", $name
)

$process = Start-Process -FilePath $python -ArgumentList $arguments `
    -WorkingDirectory $project -WindowStyle Hidden `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
Write-Output "START pid=$($process.Id) rounds=1200 method=m2_5step_terminal"
$process.WaitForExit()
Write-Output "DONE pid=$($process.Id) exit=$($process.ExitCode)"
if ($process.ExitCode -ne 0) {
    exit $process.ExitCode
}

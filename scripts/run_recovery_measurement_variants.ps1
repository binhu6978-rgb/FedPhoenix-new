param(
    [int]$MaxParallel = 2
)

$ErrorActionPreference = "Stop"
if ($MaxParallel -ne 2) {
    throw "This controlled run is fixed to two concurrent trainers."
}

$project = Split-Path -Parent $PSScriptRoot
$python = "D:\software\minicoonda\envs\fd\python.exe"
$dataRoot = "C:\Users\admin\Desktop\need_do\new_paper_four\FedPhoenix-main\data"
$partition = Join-Path $project "data\cifar10_100_noniidCase5_beta0.3.json"
$output = Join-Path $project "results\recovery_measurement_functional_recovery_2of64"
New-Item -ItemType Directory -Force -Path $output | Out-Null

$pending = [System.Collections.Queue]::new()
@(
    [pscustomobject]@{ Name = "m2_steps6_terminal"; Steps = 6; Mode = "terminal" },
    [pscustomobject]@{ Name = "m2_steps8_terminal"; Steps = 8; Mode = "terminal" },
    [pscustomobject]@{ Name = "m2_steps5_trajectory_mean"; Steps = 5; Mode = "trajectory_mean" },
    [pscustomobject]@{ Name = "m2_steps5_endpoints_mean"; Steps = 5; Mode = "endpoints_mean" }
) | ForEach-Object { $pending.Enqueue($_) }

$active = [System.Collections.ArrayList]::new()
$failures = [System.Collections.ArrayList]::new()
while ($pending.Count -gt 0 -or $active.Count -gt 0) {
    while ($pending.Count -gt 0 -and $active.Count -lt $MaxParallel) {
        $variant = $pending.Dequeue()
        $runName = "$($variant.Name)_seed1_2of64_200r"
        $stdout = Join-Path $output "${runName}.log"
        $stderr = Join-Path $output "${runName}.err.log"
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
            "--probe-steps", [string]$variant.Steps,
            "--probe-recovery-measurement", $variant.Mode,
            "--run-name", $runName
        )
        $process = Start-Process -FilePath $python -ArgumentList $arguments `
            -WorkingDirectory $project -WindowStyle Hidden `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
        [void]$active.Add(
            [pscustomobject]@{ Variant = $variant; Process = $process }
        )
        Write-Output "START $($variant.Name) pid=$($process.Id)"
    }

    for ($index = $active.Count - 1; $index -ge 0; $index--) {
        $entry = $active[$index]
        $entry.Process.Refresh()
        if ($entry.Process.HasExited) {
            # Start-Process can leave ExitCode unset until the process handle has
            # been synchronized, even though HasExited is already true.
            $entry.Process.WaitForExit()
            $exitCode = $entry.Process.ExitCode
            Write-Output "DONE $($entry.Variant.Name) exit=$exitCode"
            if ($exitCode -ne 0) {
                [void]$failures.Add($entry.Variant.Name)
            }
            $active.RemoveAt($index)
        }
    }
    if ($pending.Count -gt 0 -or $active.Count -gt 0) {
        Start-Sleep -Seconds 15
    }
}

if ($failures.Count -gt 0) {
    throw "Failed variants: $($failures -join ', ')"
}
& $python (Join-Path $project "scripts\analyze_recovery_measurement_results.py")
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

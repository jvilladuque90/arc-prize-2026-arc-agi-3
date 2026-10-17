# Arm a Windows scheduled task that submits a kernel to the hidden test set as soon as the daily quota opens.
#
# Why a task and not a terminal process: a process started from a coding-assistant session dies when the session ends
# (it happened on 2026-09-09 and a submission was lost). The task survives closing the terminal.
# Cost: ZERO GPU quota - submissions run on Kaggle's infrastructure.
# The Python script does NOT compute the reset instant: it retries until the quota opens, so the start time only has to be
# "before the reset".
#
# Usage (single night):
#   powershell -File scripts/submit/arm_submit_task.ps1 -Kernel juliancamilovilla/arc-agi3-franzen-m2 `
#              -Message "..." -StartUtc "2026-10-18T00:02:00Z"
# Usage (every night, alternating the kernels listed in scripts/submit/daily_alternate.py):
#   powershell -File scripts/submit/arm_submit_task.ps1 -Alternate -Daily -StartUtc "2026-10-18T00:02:00Z"
# Switch it off with:  Disable-ScheduledTask -TaskName ARC-AGI3-SubmitOneShot
param(
  [string]$Kernel,
  [string]$Message,
  [Parameter(Mandatory=$true)][string]$StartUtc,
  [string]$TaskName = "ARC-AGI3-SubmitOneShot",
  [int]$MaxMinutes = 240,
  # -Daily: repeat the firings EVERY night (the script is idempotent per UTC day, so repeating costs nothing).
  [switch]$Daily,
  # -Alternate: the task calls scripts/submit/daily_alternate.py, which picks the kernel by the parity of the UTC date.
  [switch]$Alternate
)

$ErrorActionPreference = "Stop"
if (-not $Alternate -and ((-not $Kernel) -or (-not $Message))) { throw "-Kernel and -Message are required unless -Alternate is used" }
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)   # scripts/submit -> repo root
# The interpreter: the user-level install, which has `kaggle`. On 2026-09-11 the action pointed at the .venv python
# and the task left no log line at all.
$py = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"
if (-not (Test-Path $py)) { $py = (Get-Command python).Source }
# Probe for PRESENCE without importing: `import kaggle` authenticates on import and fails without environment variables.
& $py -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('kaggle') else 1)"
if ($LASTEXITCODE -ne 0) { throw "the interpreter $py does not have the kaggle package" }
$script = Join-Path $root "scripts\submit\submit_at_reset.py"
if ($Alternate) { $script = Join-Path $root "scripts\submit\daily_alternate.py" }

# The start time is given in UTC and converted to the LOCAL time Task Scheduler understands.
$startLocal = ([DateTime]::Parse($StartUtc)).ToLocalTime()

# The message travels through a file: passing it inline on the task command line breaks with quotes and accents.
$msgFile = Join-Path $root ".work\submit_message.txt"
New-Item -ItemType Directory -Force -Path (Split-Path $msgFile) | Out-Null
if ($Message) { Set-Content -Path $msgFile -Value $Message -Encoding utf8 }

if ($Alternate) {
    $inner = "import subprocess,sys,os; " +
             "os.environ['SUBMIT_MAX_MINUTES']='$MaxMinutes'; " +
             "sys.exit(subprocess.run([sys.executable, r'$script'], cwd=r'$root').returncode)"
} else {
    $inner = "import pathlib,subprocess,sys,os; " +
             "os.environ['SUBMIT_MAX_MINUTES']='$MaxMinutes'; " +
             "m=pathlib.Path(r'$msgFile').read_text(encoding='utf-8').strip(); " +
             "sys.exit(subprocess.run([sys.executable, r'$script', '$Kernel', m], cwd=r'$root').returncode)"
}
$action = New-ScheduledTaskAction -Execute $py -Argument "-c `"$inner`"" -WorkingDirectory $root

# FIVE firings per night, not one: if the first is lost (machine asleep, session closed) a later one recovers the quota.
# The last two are late on purpose: a rerun takes ~9 h and a platform ERROR is only visible afterwards; the script
# ignores submissions that ended in ERROR/CANCELED and retries.
$offsets = @(0, 0.75, 3, 10.5, 14)
$triggers = @()
foreach ($h in $offsets) {
    if ($Daily) { $triggers += New-ScheduledTaskTrigger -Daily -At $startLocal.AddHours($h) }
    else        { $triggers += New-ScheduledTaskTrigger -Once  -At $startLocal.AddHours($h) }
}
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew `
              -ExecutionTimeLimit (New-TimeSpan -Minutes ($MaxMinutes + 30)) `
              -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
              -RestartCount 2 -RestartInterval (New-TimeSpan -Minutes 10)

Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $triggers `
    -Settings $settings -Description "Submission to the hidden test set when the daily quota opens" | Out-Null

$t = Get-ScheduledTask -TaskName $TaskName
"task      : $TaskName  [$($t.State)]"
"kernel    : $(if ($Alternate) { 'alternating (scripts/submit/daily_alternate.py)' } else { $Kernel })"
"python    : $py"
"firings   : $StartUtc + 0 / 45 min / 3 h / 10.5 h / 14 h" + $(if ($Daily) { "  [EVERY NIGHT]" } else { "  [single night]" })
"retries   : up to $MaxMinutes min or until the quota opens"
"log       : $(Join-Path $root '.work\logs\daily_submit.log')"

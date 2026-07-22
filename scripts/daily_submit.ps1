# Submission diaria automatica a arc-prize-2026-arc-agi-3.
# La ejecuta una tarea programada de Windows a las 8pm (ver scripts/register_daily_task.ps1).
# Envia la ULTIMA version del mejor kernel (por defecto arc-agi3-llm, el hibrido) para
# consumir el cupo diario (~1/dia; resetea 00:00 UTC). Lee credenciales de .env (no en git).
param(
  [string]$Kernel = "juliancamilovilla/arc-agi3-llm",
  [string]$Comp   = "arc-prize-2026-arc-agi-3",
  [string]$File   = "submission.parquet"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$log  = Join-Path $root "daily_submit.log"

function Log($m) { "$([DateTime]::UtcNow.ToString('s'))Z  $m" | Add-Content -Encoding utf8 $log }

# Cargar .env (KAGGLE_API_TOKEN, kaggle_username)
$envPath = Join-Path $root ".env"
if (-not (Test-Path $envPath)) { Log "ERROR: no hay .env"; exit 1 }
Get-Content $envPath | ForEach-Object {
  $l = $_.Trim()
  if ($l -and -not $l.StartsWith("#") -and $l.Contains("=")) {
    $k,$v = $l.Split("=",2)
    Set-Item -Path "Env:$($k.Trim())" -Value $v.Trim()
  }
}
if ($env:kaggle_username) { $env:KAGGLE_USERNAME = $env:kaggle_username }

$msg = "auto-daily hybrid $([DateTime]::UtcNow.ToString('yyyy-MM-dd'))"
Log "submitting latest version of $Kernel ..."
# Sin -v: la CLI toma la ultima version del kernel (la mas reciente que corrio Save & Run).
$out = & kaggle competitions submit $Comp -k $Kernel -f $File -m $msg 2>&1
Log ($out -join " | ")
if ($out -match "successfully") { Log "OK" } else { Log "FALLO (posible cupo diario ya usado)" }

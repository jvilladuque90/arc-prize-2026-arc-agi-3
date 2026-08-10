# Submission diaria automatica a arc-prize-2026-arc-agi-3.
# La ejecuta una tarea programada de Windows a las 8pm (ver scripts/register_daily_task.ps1).
# Envia la ULTIMA version del mejor kernel (default: arc-agi3-explorer054, la replica 0.54)
# para consumir el cupo diario (~1/dia; resetea 00:00 UTC). Lee credenciales de .env.
#
# 2026-08-10: robustecido — la version anterior murio silenciosa ~2 semanas:
#   (a) $ErrorActionPreference=Stop abortaba en la llamada a kaggle sin loguear;
#   (b) 'kaggle' puede no estar en PATH en el contexto de la tarea programada.
param(
  [string]$Kernel = "juliancamilovilla/arc-agi3-explorer054",
  [string]$Comp   = "arc-prize-2026-arc-agi-3",
  [string]$File   = "submission.parquet"
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
$log  = Join-Path $root "daily_submit.log"

function Log($m) { "$([DateTime]::UtcNow.ToString('s'))Z  $m" | Add-Content -Encoding utf8 $log }

try {
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

  # Resolver el ejecutable kaggle aunque la tarea no tenga el PATH del usuario
  $kaggle = (Get-Command kaggle -ErrorAction SilentlyContinue).Source
  if (-not $kaggle) {
    $cands = @("$env:APPDATA\Python\Python312\Scripts\kaggle.exe",
               "$env:LOCALAPPDATA\Programs\Python\Python312\Scripts\kaggle.exe",
               (Join-Path $root ".venv\Scripts\kaggle.exe"))
    $kaggle = $cands | Where-Object { Test-Path $_ } | Select-Object -First 1
  }
  if (-not $kaggle) { Log "ERROR: kaggle.exe no encontrado en PATH ni candidatos"; exit 1 }
  Log "kaggle = $kaggle"

  $msg = "auto-daily $([DateTime]::UtcNow.ToString('yyyy-MM-dd')) latest $Kernel"
  Log "submitting latest version of $Kernel ..."
  $out = & $kaggle competitions submit $Comp -k $Kernel -f $File -m $msg 2>&1
  Log ("result: " + (($out | Out-String).Trim() -replace "`r?`n", " | "))
  if (($out | Out-String) -match "successfully") { Log "OK" } else { Log "FALLO (posible cupo diario ya usado u otro error, ver arriba)" }
}
catch {
  Log ("EXCEPTION: " + $_.Exception.Message)
}

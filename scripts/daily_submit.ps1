# Submission diaria automatica a arc-prize-2026-arc-agi-3.
# La ejecuta una tarea programada de Windows a las 8pm (ver scripts/register_daily_task.ps1).
# Envia la ULTIMA version del mejor kernel (default: arc-agi3-duck) para consumir el
# cupo diario (~1/dia; resetea 00:00 UTC). Lee credenciales de .env.
#
# 2026-08-10: robustecido — la version anterior murio silenciosa ~2 semanas:
#   (a) $ErrorActionPreference=Stop abortaba en la llamada a kaggle sin loguear;
#   (b) 'kaggle' puede no estar en PATH en el contexto de la tarea programada.
# 2026-08-11: CAUSA RAIZ del fallo persistente: las code competitions exigen
#   -v <version del kernel> ademas de -f. Sin -v Kaggle responde "Code competition
#   submissions require both the output file name and the version number" SIEMPRE
#   (no es señal de cupo agotado — ese diagnostico previo era incorrecto).
#   La version se lee de kernel_versions.json, que scripts/push_kernels.py
#   actualiza en cada push.
param(
  [string]$Kernel = "juliancamilovilla/arc-agi3-duck",
  [string]$Comp   = "arc-prize-2026-arc-agi-3",
  [string]$File   = "submission.parquet",
  [switch]$DryRun
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

  # Version del kernel (obligatoria en code competitions), de kernel_versions.json
  $vfile = Join-Path $root "kernel_versions.json"
  if (-not (Test-Path $vfile)) { Log "ERROR: no hay kernel_versions.json (correr push_kernels.py)"; exit 1 }
  $versions = Get-Content $vfile -Raw | ConvertFrom-Json
  $ver = $versions.$Kernel
  if (-not $ver) { Log "ERROR: $Kernel no esta en kernel_versions.json"; exit 1 }
  Log "kernel version = $ver (de kernel_versions.json)"

  $msg = "auto-daily $([DateTime]::UtcNow.ToString('yyyy-MM-dd')) $Kernel v$ver"
  if ($DryRun) { Log "DRY-RUN: & $kaggle competitions submit $Comp -k $Kernel -f $File -v $ver -m '$msg'"; exit 0 }
  Log "submitting $Kernel v$ver ..."
  $out = & $kaggle competitions submit $Comp -k $Kernel -f $File -v $ver -m $msg 2>&1
  Log ("result: " + (($out | Out-String).Trim() -replace "`r?`n", " | "))
  # Verificacion robusta: el CLI puede no imprimir nada en exito (visto 2026-08-13,
  # submission creada con result vacio). La verdad esta en la lista de submissions.
  Start-Sleep -Seconds 10
  $rows = & $kaggle competitions submissions -c $Comp --csv 2>$null | Where-Object { $_ -match "," }
  $newest = $rows | Select-Object -Skip 1 -First 1
  Log ("newest submission: " + $newest)
  if ($newest -match [regex]::Escape($msg)) { Log "OK (verificado en la lista)" }
  elseif (($out | Out-String) -match "Submission limit exceeded|maximum number") { Log "CUPO DIARIO YA USADO (no es error del script)" }
  else { Log "FALLO: la submission de hoy no aparece en la lista (ver arriba)" }
}
catch {
  Log ("EXCEPTION: " + $_.Exception.Message)
}

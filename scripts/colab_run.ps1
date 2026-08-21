# Lanza un script en Colab T4 limpiando antes las sesiones colgadas.
#
# POR QUE: cada `colab run` deja la sesion asignada al terminar (con una etiqueta
# aleatoria tipo 'c4b', 'poda', 'tier1'), y la siguiente invocacion falla con
# TooManyAssignmentsError. Ademas las sesiones sin parar queman compute units.
# Tres corridas seguidas se perdieron por esto antes de automatizarlo.
#
# Uso: powershell -ExecutionPolicy Bypass -File scripts\colab_run.ps1 scripts\colab_micro_eval.py [timeout]
param(
  [Parameter(Mandatory=$true)][string]$Script,
  [int]$TimeoutSec = 7000
)
$ErrorActionPreference = "Continue"

$sesiones = & colab --auth=adc sessions 2>&1
foreach ($linea in $sesiones) {
  if ($linea -match '^\[([^\]]+)\]') {
    $id = $Matches[1]
    Write-Host "[limpieza] parando sesion colgada '$id'"
    & colab --auth=adc stop -s $id 2>&1 | Out-Null
  }
}

Write-Host "[colab_run] lanzando $Script (timeout ${TimeoutSec}s)"
& colab --auth=adc run --gpu T4 --timeout $TimeoutSec $Script

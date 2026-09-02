# Rota la ADC de Colab a la cuenta N y lanza un script, con salvaguardas.
#
# REGLAS (autorizadas por Julian 2026-09-01, "adelante con las cuentas usalas"):
#  - Los backups adc_backup_cuentaN.json NUNCA se modifican; solo se copian.
#  - Antes de usar una cuenta se listan sus sesiones. En cuentas 1-3 (compartidas
#    con arc-agi-2) una sesion activa se respeta y se pasa a otra cuenta: puede
#    ser trabajo del otro proyecto. En 4-5 (nuestras) se limpian como colab_run.
#  - NUNCA rotar mientras uno de NUESTROS lanzamientos siga corriendo en otra
#    cuenta: la ADC es global y el refresco de token puede romper la sesion en
#    vuelo. El estado se lleva en scratchpad\colab_accounts_state.json.
#
# Uso: powershell -File scripts\colab_cuenta.ps1 -Cuenta 5 -Script scripts\colab_slots_bench.py [-TimeoutSec 7000]
param(
  [Parameter(Mandatory=$true)][int]$Cuenta,
  [Parameter(Mandatory=$true)][string]$Script,
  [int]$TimeoutSec = 7000
)
$ErrorActionPreference = "Continue"
$gcloud = Join-Path $env:APPDATA "gcloud"
$backup = Join-Path $gcloud "adc_backup_cuenta$Cuenta.json"
$activa = Join-Path $gcloud "application_default_credentials.json"
$estado = Join-Path $PSScriptRoot "..\_colab_state.json"

if (-not (Test-Path $backup)) { Write-Host "[cuenta$Cuenta] sin backup, abortando"; exit 1 }

# ¿hay un lanzamiento nuestro vivo en otra cuenta?
if (Test-Path $estado) {
  $st = Get-Content $estado -Raw | ConvertFrom-Json
  if ($st.corriendo -and $st.cuenta -ne $Cuenta) {
    Write-Host "[guard] hay un lanzamiento nuestro en cuenta $($st.cuenta) ($($st.script)); no roto la ADC"
    exit 2
  }
}

Copy-Item $backup $activa -Force
Write-Host "[cuenta$Cuenta] ADC instalada"

$sesiones = & colab --auth=adc sessions 2>&1
$ocupada = $false
foreach ($linea in $sesiones) {
  if ($linea -match '^\[([^\]]+)\]') {
    $id = $Matches[1]
    if ($Cuenta -ge 4) {
      Write-Host "[cuenta$Cuenta] parando sesion nuestra colgada '$id'"
      & colab --auth=adc stop -s $id 2>&1 | Out-Null
    } else {
      Write-Host "[cuenta$Cuenta] sesion activa '$id' (posible arc-agi-2): la respeto"
      $ocupada = $true
    }
  }
}
if ($ocupada) { Write-Host "[cuenta$Cuenta] ocupada; prueba otra cuenta"; exit 2 }

@{ corriendo = $true; cuenta = $Cuenta; script = $Script
   lanzado = (Get-Date -Format s) } | ConvertTo-Json | Out-File $estado -Encoding utf8

Write-Host "[cuenta$Cuenta] lanzando $Script (timeout ${TimeoutSec}s)"
& colab --auth=adc run --gpu T4 --timeout $TimeoutSec $Script
$rc = $LASTEXITCODE

@{ corriendo = $false; cuenta = $Cuenta; script = $Script
   terminado = (Get-Date -Format s); exit = $rc } | ConvertTo-Json | Out-File $estado -Encoding utf8
Write-Host "[cuenta$Cuenta] terminado (exit $rc)"
exit $rc

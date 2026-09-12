# Arma una tarea programada de Windows de UN SOLO DISPARO que envia un kernel al
# set oculto en cuanto el cupo diario abra.
#
# Por que una tarea y no un proceso de la terminal: un proceso de la sesion de
# Claude Code muere cuando la sesion termina (paso el 2026-09-09 y el envio de
# v23 se perdio). La tarea sobrevive al cierre de la terminal.
#
# Coste: CERO cuota de G4 — los envios corren en la infraestructura de Kaggle
# (regla de STRATEGY 11: ninguna tarea programada gasta GPU).
#
# El script de Python NO calcula el instante del reset: reintenta hasta que el
# cupo abra, asi que la hora de arranque solo tiene que ser "antes del reset".
#
# Uso:
#   powershell -File scripts/arm_submit_task.ps1 -Kernel juliancamilovilla/arc-agi3-duck-anim `
#              -Message "v23 ..." -StartUtc "2026-09-09T23:40:00Z"
param(
  [Parameter(Mandatory=$true)][string]$Kernel,
  [Parameter(Mandatory=$true)][string]$Message,
  [Parameter(Mandatory=$true)][string]$StartUtc,
  [string]$TaskName = "ARC-AGI3-SubmitOneShot",
  [int]$MaxMinutes = 240
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
# El interprete: el de la instalacion de usuario, que tiene `kaggle`. La noche del
# 2026-09-11 la accion quedo apuntando al python del .venv (asi lo resolvio
# Get-Command en aquella sesion) y la tarea no dejo ni una linea de log.
$py = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"
if (-not (Test-Path $py)) { $py = (Get-Command python).Source }
# Sondear PRESENCIA sin importar: `import kaggle` autentica al importarse y sin
# variables de entorno falla aunque el paquete este instalado.
& $py -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('kaggle') else 1)"
if ($LASTEXITCODE -ne 0) { throw "el interprete $py no tiene el paquete kaggle" }
$script = Join-Path $root "scripts\submit_at_reset.py"

# La hora de arranque se da en UTC y se convierte a la hora LOCAL de Windows,
# que es la que entiende el Task Scheduler. Evita el enredo de husos.
$startLocal = ([DateTime]::Parse($StartUtc)).ToLocalTime()

$msgFile = Join-Path $root ".submit_message.txt"
Set-Content -Path $msgFile -Value $Message -Encoding utf8

# El mensaje viaja por archivo: pasarlo inline por la linea de comandos de la
# tarea rompe con comillas y acentos.
$inner = "import pathlib,subprocess,sys,os; " +
         "os.environ['SUBMIT_MAX_MINUTES']='$MaxMinutes'; " +
         "m=pathlib.Path(r'$msgFile').read_text(encoding='utf-8').strip(); " +
         "sys.exit(subprocess.run([sys.executable, r'$script', '$Kernel', m], cwd=r'$root').returncode)"

$action  = New-ScheduledTaskAction -Execute $py -Argument "-c `"$inner`"" -WorkingDirectory $root
# TRES disparos por noche, no uno: si el primero se pierde (maquina dormida,
# sesion cerrada), el segundo o el tercero recuperan el cupo. El script es
# idempotente (si ya hay envio hoy UTC, no reenvia), asi que los extra no cuestan.
$triggers = @(
    (New-ScheduledTaskTrigger -Once -At $startLocal),
    (New-ScheduledTaskTrigger -Once -At $startLocal.AddMinutes(45)),
    (New-ScheduledTaskTrigger -Once -At $startLocal.AddHours(3))
)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew `
              -ExecutionTimeLimit (New-TimeSpan -Minutes ($MaxMinutes + 30)) `
              -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
              -RestartCount 2 -RestartInterval (New-TimeSpan -Minutes 10)

Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $triggers `
    -Settings $settings -Description "Envio unico de $Kernel al abrir el cupo diario" | Out-Null

$t = Get-ScheduledTask -TaskName $TaskName
"tarea      : $TaskName  [$($t.State)]"
"kernel     : $Kernel"
"python     : $py"
"disparos   : $startLocal, $($startLocal.AddMinutes(45)), $($startLocal.AddHours(3)) (local)  =  $StartUtc + 45m + 3h"
"reintenta  : hasta $MaxMinutes min o hasta que el cupo abra"
"log        : $(Join-Path $root 'daily_submit.log')"

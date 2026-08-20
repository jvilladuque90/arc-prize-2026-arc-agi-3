# Registra la tarea que despliega duck+effects en cuanto resetee la cuota semanal de G4.
#
# POR QUE UNA TAREA Y NO UN PROCESO EN SEGUNDO PLANO: el reset es el viernes 8pm y un
# bucle atado a una sesion de terminal no sobrevive tanto. La tarea reintenta cada 2 h
# aunque se cierre todo.
#
# El script llamado lleva centinela (.effects_deployed): tras validar el despliegue,
# los intentos siguientes salen de inmediato sin republicar. Sin eso la tarea gastaria
# cuota de G4 cada 2 h indefinidamente.
#
# Ejecutar una vez:  powershell -ExecutionPolicy Bypass -File scripts\register_effects_deploy_task.ps1
$ErrorActionPreference = "Stop"
$root   = Split-Path -Parent $PSScriptRoot
$script = Join-Path $root "scripts\push_effects_when_quota.py"
$log    = Join-Path $root "effects_deploy.log"
$name   = "ARC-AGI3-DeployEffects"

$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) { $py = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" }
if (-not (Test-Path $py)) { throw "no encuentro python.exe" }

$cmd = "& '$py' '$script' --once *>> '$log'"
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
  -Argument "-NoProfile -ExecutionPolicy Bypass -Command `"$cmd`"" -WorkingDirectory $root

# Cada 2 h durante 5 dias: cubre de sobra el reset del viernes 8pm.
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(5) `
  -RepetitionInterval (New-TimeSpan -Hours 2) -RepetitionDuration (New-TimeSpan -Days 5)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 3)

Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger -Settings $settings `
  -Description "Despliega duck+effects (carga del seam C) al resetear la cuota de G4" -Force | Out-Null

Write-Host "Tarea '$name' registrada: cada 2h durante 5 dias, log en $log"
Write-Host "Ver:     Get-ScheduledTask -TaskName $name"
Write-Host "Quitar:  Unregister-ScheduledTask -TaskName $name -Confirm:`$false"
Write-Host "Parar el despliegue sin quitar la tarea: crear el fichero .effects_deployed"

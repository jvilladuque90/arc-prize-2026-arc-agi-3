# Registra (o actualiza) la tarea programada de Windows que hace la submission diaria
# a las 8:00 pm (hora local) llamando a scripts/daily_submit.ps1.
# Ejecutar una vez:  powershell -ExecutionPolicy Bypass -File scripts\register_daily_task.ps1
$ErrorActionPreference = "Stop"
$root   = Split-Path -Parent $PSScriptRoot
$script = Join-Path $root "scripts\daily_submit.ps1"
$name   = "ARC-AGI3-DailySubmit"

$action  = New-ScheduledTaskAction -Execute "powershell.exe" `
  -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$script`""
$trigger = New-ScheduledTaskTrigger -Daily -At 8:00PM
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries

Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger -Settings $settings `
  -Description "Submission diaria a ARC-AGI-3 (arc-agi3-llm hibrido)" -Force | Out-Null

Write-Host "Tarea '$name' registrada: diaria 8:00pm hora local."
Write-Host "Ver:      Get-ScheduledTask -TaskName $name"
Write-Host "Probar:   Start-ScheduledTask -TaskName $name   (revisa daily_submit.log)"
Write-Host "Quitar:   Unregister-ScheduledTask -TaskName $name -Confirm:`$false"

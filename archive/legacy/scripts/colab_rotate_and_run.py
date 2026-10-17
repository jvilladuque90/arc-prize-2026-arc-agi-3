"""Orquestador local: espera sesión ajena → rota ADC a cuenta 4/5 → corre el proxy.

Regla de partición (2026-08-13): AG3 usa SOLO las cuentas 4 y 5 de Colab
(adc_backup_cuenta4/5.json). El ADC activo es global de la máquina: no rotar
mientras haya una sesión de AG2 corriendo (romperia el refresh de su cliente).

Flujo:
  1. Poll `colab sessions` bajo el ADC actual hasta que no haya sesiones (o timeout).
  2. Copia adc_backup_cuenta4.json → application_default_credentials.json.
  3. Sonda barata de T4 (colab_hello.py). "Service Unavailable" → rota a cuenta 5.
  4. Lanza scripts/colab_taaf_proxy.py en la T4 (timeout 2 h).

Uso: python scripts/colab_rotate_and_run.py   (lee KAGGLE_API_TOKEN de .env)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GCLOUD = Path(os.environ["APPDATA"]) / "gcloud"
ADC = GCLOUD / "application_default_credentials.json"
WAIT_FOREIGN_MAX_S = 6 * 3600
POLL_S = 900


def log(m: str) -> None:
    print(f"[rotate {time.strftime('%H:%M')}] {m}", flush=True)


def sessions_output() -> str:
    r = subprocess.run(["colab", "--auth=adc", "sessions"],
                       capture_output=True, text=True)
    return (r.stdout or "") + (r.stderr or "")


def run_colab(args: list[str], timeout_s: int) -> subprocess.CompletedProcess:
    return subprocess.run(["colab", "--auth=adc", *args],
                          capture_output=True, text=True, timeout=timeout_s + 300)


def main() -> int:
    for line in (ROOT / ".env").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()
    tok = os.environ["KAGGLE_API_TOKEN"]

    # 1. esperar a que no haya sesiones ajenas bajo el ADC actual (de AG2)
    deadline = time.time() + WAIT_FOREIGN_MAX_S
    while time.time() < deadline:
        out = sessions_output()
        if "No active sessions" in out:
            log("sin sesiones ajenas; procedo a rotar ADC")
            break
        log(f"sesión ajena activa aún: {out.strip().splitlines()[0][:100]}")
        time.sleep(POLL_S)
    else:
        log("timeout esperando a AG2; NO roto el ADC")
        return 1

    # 2-3. rotar a cuenta 4; si su T4 está agotada, cuenta 5
    for cuenta in ("4", "5"):
        backup = GCLOUD / f"adc_backup_cuenta{cuenta}.json"
        if not backup.exists():
            log(f"no existe {backup}"); continue
        shutil.copyfile(backup, ADC)
        log(f"ADC ← cuenta {cuenta}; sonda T4 ...")
        try:
            probe = run_colab(["run", "--gpu", "T4", "--timeout", "600",
                               "scripts/colab_hello.py"], 900)
        except subprocess.TimeoutExpired:
            log("sonda colgada; pruebo siguiente cuenta"); continue
        allout = (probe.stdout or "") + (probe.stderr or "")
        if probe.returncode == 0 and "gpu:" in allout.lower():
            log(f"cuenta {cuenta} tiene T4 disponible")
            # 4. lanzar el proxy (stdout en vivo hacia nuestro log)
            r = subprocess.run(["colab", "--auth=adc", "run", "--gpu", "T4",
                                "--timeout", "7200",
                                "scripts/colab_taaf_proxy.py", tok])
            log(f"proxy terminó rc={r.returncode}")
            return r.returncode
        log(f"cuenta {cuenta} sin T4 ahora: {allout.strip().splitlines()[-1][:150] if allout.strip() else 'sin salida'}")

    log("ninguna cuenta (4/5) tiene T4 disponible ahora")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

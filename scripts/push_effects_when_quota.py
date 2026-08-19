"""Publica duck+effects en cuanto resetee la cuota semanal de G4 (reintento cada 2 h).

QUE DESPLIEGA. La carga del seam C en formato palabras (docs/DESIGN.md §8.11-8.12):
la tabla de efectos MEDIDA del historial, inyectada como texto en el prompt. Cero
turnos, cero llamadas al sandbox. Medido en el banco micro con Qwen3-4B sobre 109
problemas de planificacion: 44.0% sin tabla -> 66.1% con tabla vectorial -> 86.2%
con el formato en palabras.

PROTECCION DE SECUENCIA (la razon de que este script exista). push_kernels.py
escribe kernel_versions.json al publicar, pero la validacion Save&Run tarda ~1 h.
Si el trigger diario de medianoche dispara en esa ventana, enviaria una version
sin validar y se perderia el envio del dia. Por eso aqui se revierte el fichero a
la ULTIMA VERSION VALIDADA justo despues del push, y solo se avanza cuando el
estado del kernel es COMPLETE.

A diferencia de push_v4_when_quota.py, la version de repliegue no esta cableada:
se lee del fichero ANTES de publicar, asi que el script sirve para cualquier
despliegue futuro sin editarlo.

Uso: python scripts/push_effects_when_quota.py [--horas 2] [--intentos 36]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VFILE = ROOT / "kernel_versions.json"
KERNEL = "juliancamilovilla/arc-agi3-duck"


def log(m: str) -> None:
    print(f"[push-effects {time.strftime('%m-%d %H:%M')}] {m}", flush=True)


def get_version() -> int | None:
    return json.loads(VFILE.read_text(encoding="utf-8")).get(KERNEL)


def set_version(v: int) -> None:
    d = json.loads(VFILE.read_text(encoding="utf-8"))
    d[KERNEL] = v
    VFILE.write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
    log(f"kernel_versions.json: duck -> v{v}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horas", type=float, default=2.0)
    ap.add_argument("--intentos", type=int, default=36)
    args = ap.parse_args()

    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, val = line.split("=", 1)
            os.environ[k.strip()] = val.strip()
    os.environ["KAGGLE_USERNAME"] = os.environ.get("kaggle_username", "juliancamilovilla")

    validada = get_version()          # version que el trigger diario esta enviando
    log(f"version validada actual: v{validada} (repliegue si el push no valida)")

    for intento in range(1, args.intentos + 1):
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "push_kernels.py"),
                            "duck", "--gpu"], capture_output=True, text=True)
        out = (r.stdout or "") + (r.stderr or "")

        if "quota" in out.lower():
            log(f"intento {intento}/{args.intentos}: cuota aun agotada; "
                f"reintento en {args.horas} h")
            time.sleep(args.horas * 3600)
            continue
        if r.returncode != 0 or "successfully pushed" not in out:
            log(f"push fallo por otra razon, ABORTO: {out.strip()[-400:]}")
            return 1

        m = re.search(r"[Kk]ernel version (\d+)", out)
        nueva = int(m.group(1)) if m else None
        log(f"push OK (v{nueva})")
        if validada is not None:
            set_version(validada)     # el trigger sigue enviando lo ya probado
            log(f"trigger apuntado a v{validada} hasta que la nueva valide")

        for _ in range(40):           # ~2 h de espera de validacion
            s = subprocess.run(["kaggle", "kernels", "status", KERNEL],
                               capture_output=True, text=True, shell=True)
            st = ((s.stdout or "") + (s.stderr or "")).strip()
            log(st.splitlines()[-1][:120] if st else "sin status")
            if "COMPLETE" in st:
                if nueva:
                    set_version(nueva)
                log("VALIDADO: el trigger diario ya envia la version con --effects")
                return 0
            if "ERROR" in st or "CANCEL" in st:
                log(f"la validacion FALLO; el trigger se queda en v{validada}")
                return 1
            time.sleep(180)
        log(f"timeout esperando validacion; el trigger se queda en v{validada}")
        return 1

    log("horizonte agotado sin reset de cuota")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

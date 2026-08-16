"""Publica duck v4 apenas la cuota semanal de G4 resetee (reintento cada 4 h).

Protección de secuencia: push_kernels.py escribe kernel_versions.json al publicar,
pero la validación Save&Run tarda ~1 h; si el trigger de las 8pm dispara en esa
ventana, el submit de una versión sin validar falla y se pierde la automatización
del día. Por eso este script revierte duck→2 justo tras el push y solo lo sube a
la versión nueva cuando el estado del kernel es COMPLETE.
"""

from __future__ import annotations

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
    print(f"[push-v4 {time.strftime('%H:%M')}] {m}", flush=True)


def set_version(v: int) -> None:
    d = json.loads(VFILE.read_text(encoding="utf-8"))
    d[KERNEL] = v
    VFILE.write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
    log(f"kernel_versions.json: duck -> v{v}")


def main() -> int:
    for line in (ROOT / ".env").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, val = line.split("=", 1)
            os.environ[k.strip()] = val.strip()
    os.environ["KAGGLE_USERNAME"] = os.environ.get("kaggle_username", "juliancamilovilla")

    for attempt in range(1, 19):  # 18 × 4 h = 72 h de horizonte
        log(f"intento de push {attempt}/18")
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "push_kernels.py"),
                            "duck", "--gpu"], capture_output=True, text=True)
        out = (r.stdout or "") + (r.stderr or "")
        if r.returncode == 0 and "successfully pushed" in out:
            m = re.search(r"[Kk]ernel version (\d+)", out)
            new_v = int(m.group(1)) if m else None
            # Revertir a la version ANTERIOR (la ya validada) mientras la nueva
            # corre su Save&Run: si el trigger de las 8pm dispara en esa ventana,
            # envia lo ultimo probado y no una version sin validar.
            prev = json.loads(VFILE.read_text(encoding="utf-8")).get(KERNEL)
            prev = (new_v - 1) if (new_v and prev == new_v) else prev
            log(f"push OK (v{new_v}); revierto a v{prev} hasta que valide")
            set_version(int(prev))
            # esperar validación
            for _ in range(40):
                s = subprocess.run(["kaggle", "kernels", "status", KERNEL],
                                   capture_output=True, text=True)
                st = (s.stdout or "") + (s.stderr or "")
                log(st.strip().splitlines()[-1][:120] if st.strip() else "sin status")
                if "COMPLETE" in st:
                    if new_v:
                        set_version(new_v)
                    log("validación COMPLETE; trigger apuntado a la versión nueva")
                    return 0
                if "ERROR" in st or "CANCEL" in st:
                    log("validación FALLÓ; trigger queda en v2")
                    return 1
                time.sleep(180)
            log("timeout esperando validación; trigger queda en v2")
            return 1
        if "quota" in out.lower():
            log("cuota aún agotada; reintento en 4 h")
        else:
            log(f"push falló por otra razón: {out.strip()[-300:]}")
        time.sleep(4 * 3600)
    log("horizonte agotado sin reset de cuota")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

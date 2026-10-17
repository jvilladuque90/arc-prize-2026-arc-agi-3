"""Lanza el A/B de banking (duckctx + duckctx2) — un solo disparo tras el reset.

Coste: ~5.2h de la cuota semanal COMPARTIDA (agi3 + agi2 + biohub facturan del
mismo pool de 30h; los envios diarios NO facturan — verificado 2026-08-28: toda
la semana salieron con la cuota a cero). Los dos brazos son identicos a v6 salvo
el flag banking; TAAF_OFFLINE_SOFT_MIN=120 = regimen emparejado por juego.

Uso: python scripts/push_banking_ab.py
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
SENTINEL = ROOT / ".banking_ab_launched"


def log(m: str) -> None:
    print(f"[banking-ab {time.strftime('%m-%d %H:%M')}] {m}", flush=True)


def main() -> int:
    if SENTINEL.exists():
        log("ya lanzado; nada que hacer")
        return 0
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()
    os.environ["KAGGLE_USERNAME"] = os.environ.get("kaggle_username", "juliancamilovilla")

    ok = 0
    for target in ("duckctx", "duckctx2"):
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "push_kernels.py"),
                            target, "--gpu"], capture_output=True, text=True)
        out = (r.stdout or "") + (r.stderr or "")
        if "quota" in out.lower():
            log(f"{target}: cuota agotada — no se lanza (reintentar tras el reset)")
            return 1
        if r.returncode == 0:
            ok += 1
            log(f"{target}: push OK, Save&Run en marcha")
        else:
            log(f"{target}: FALLO: {out.strip()[-200:]}")
    if ok == 2:
        SENTINEL.write_text(f"lanzado {time.strftime('%Y-%m-%d %H:%M')}", encoding="utf-8")
        log("ambos brazos corriendo (~2.6h). Comparar con scripts/compare_runs.py al terminar")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Smoke LOCAL (CPU, cero cuota) del parche de pensamiento apagado sobre el montaje REAL.

Ejecuta la celda EXACTA del notebook contra el harness del fork (byte-identico al del
bundle NVFP4) y comprueba: el global queda en False; el sitio que arma la peticion lo
lee POR NOMBRE del modulo (no una copia capturada), asi que el cambio llega a cada
llamada; la consolidacion sigue en el notebook; y el yield NO cambia (una variable).
"""

from __future__ import annotations

import inspect
import json
import os
import pathlib
import re
import sys
from pathlib import Path

if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath

ROOT = Path(__file__).resolve().parents[1]
FORK = ROOT / "_tmp_fork_bundle"
NB = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "notebooks" / "nvfp4_carry_nothink_long.ipynb"

fallos: list[str] = []


def check(ok, nombre, detalle=""):
    print(("  ok   " if ok else "  FALLA") + f"  {nombre}" + (f" — {detalle}" if detalle else ""))
    if not ok:
        fallos.append(nombre)


def main() -> int:
    print(f"SMOKE DEL PENSAMIENTO APAGADO ({NB.name})")
    for repo in ("ARC3-Inference", "tufa-arc-agi-framework/src"):
        sys.path.insert(0, str(FORK / "src" / repo))
    os.environ["LOCAL_ANALYZER_ENABLE_THINKING"] = "true"   # como en el kernel
    os.environ.setdefault("LOCAL_ANALYZER_YIELD_SECONDS", "60")

    nb = json.loads(NB.read_text(encoding="utf-8"))
    celdas = ["".join(c["source"]) for c in nb["cells"] if c.get("cell_type") == "code"]
    trozo = next((c for c in celdas if "NOTHINK_PATCH" in c), None)
    check(trozo is not None, "la celda del parche esta en el notebook")
    if trozo is None:
        return 1
    check(any("LEVEL_CARRY injected" in c for c in celdas), "conserva la consolidacion (palanca vigente)")
    check(not any("YIELD_PATCH" in c for c in celdas), "NO lleva el parche de yield (una sola variable)")

    import inference.agent.tool_agent as ta
    check(ta._LOCAL_ANALYZER_ENABLE_THINKING is True, "antes del parche el global es True (como en el kernel)")

    src = inspect.getsource(ta)
    usos = [l.strip() for l in src.splitlines() if "thinking=bool(_LOCAL_ANALYZER_ENABLE_THINKING)" in l]
    check(len(usos) >= 1, "la peticion lee el global POR NOMBRE en cada llamada", usos[0] if usos else "no encontrado")
    check(not re.search(r"self\._enable_thinking\s*=\s*_LOCAL_ANALYZER_ENABLE_THINKING", src),
          "no hay copia capturada en __init__ que ignore el parche")

    exec(compile(trozo, "<celda-nothink>", "exec"), {})
    check(ta._LOCAL_ANALYZER_ENABLE_THINKING is False, "el global queda en False")
    check(os.environ.get("LOCAL_ANALYZER_ENABLE_THINKING") == "false", "el entorno tambien queda en false")
    check(ta._LOCAL_ANALYZER_YIELD_SECONDS == 60.0, "el yield sigue en 60 (no se toca)")

    print(f"\nSMOKE: {'PASS' if not fallos else 'FAIL — ' + '; '.join(fallos)}")
    return 0 if not fallos else 1


if __name__ == "__main__":
    raise SystemExit(main())

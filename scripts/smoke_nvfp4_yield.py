"""Smoke LOCAL (CPU, cero cuota) del parche de presupuesto de turno sobre el montaje REAL.

Ejecuta la celda EXACTA del notebook contra el harness del fork (byte-identico al del
bundle NVFP4), construye un ToolAgent por la via del solver y comprueba que
self._yield_seconds == 180.0 (y que antes del parche era el valor del entorno).
"""

from __future__ import annotations

import json
import os
import pathlib
import pickle
import sys
from pathlib import Path

if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath

ROOT = Path(__file__).resolve().parents[1]
FORK = ROOT / "_tmp_fork_bundle"
NB = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "notebooks" / "nvfp4_carry_yield_long.ipynb"

fallos: list[str] = []


def check(ok, nombre, detalle=""):
    print(("  ok   " if ok else "  FALLA") + f"  {nombre}" + (f" — {detalle}" if detalle else ""))
    if not ok:
        fallos.append(nombre)


def main() -> int:
    print(f"SMOKE DEL PRESUPUESTO DE TURNO ({NB.name})")
    for repo in ("ARC3-Inference", "tufa-arc-agi-framework/src"):
        sys.path.insert(0, str(FORK / "src" / repo))
    os.environ.setdefault("LOCAL_ANALYZER_BASE_URL", "http://127.0.0.1:1234/v1")
    os.environ.setdefault("LOCAL_ANALYZER_MODEL_ID", "Qwen/Qwen3.8-Flash-Next-NVFP4")
    # Reproducir el entorno del bundle: yield 60 como en el kernel.
    os.environ["LOCAL_ANALYZER_YIELD_SECONDS"] = "60"

    nb = json.loads(NB.read_text(encoding="utf-8"))
    celdas = ["".join(c["source"]) for c in nb["cells"] if c.get("cell_type") == "code"]
    trozo = next((c for c in celdas if "YIELD_PATCH" in c), None)
    check(trozo is not None, "la celda del parche esta en el notebook")
    if trozo is None:
        return 1
    check(any("LEVEL_CARRY injected" in c for c in celdas),
          "el notebook conserva la celda de consolidacion (la palanca vigente)")

    import inference.agent.tool_agent as ta
    check(ta._LOCAL_ANALYZER_YIELD_SECONDS == 60.0,
          "antes del parche el global vale 60 (como en el kernel)", str(ta._LOCAL_ANALYZER_YIELD_SECONDS))

    with open(FORK / "benchmark_initial.pkl", "rb") as fh:
        bm = pickle.load(fh)
    game = next(iter(getattr(bm, "games", []) or []), None)
    antes = bm.solver._make_analyzer(game, 0)
    check(getattr(antes, "_yield_seconds", None) == 60.0,
          "un agente construido ANTES del parche lleva 60 s", str(getattr(antes, "_yield_seconds", None)))

    exec(compile(trozo, "<celda-yield>", "exec"), {})
    check(ta._LOCAL_ANALYZER_YIELD_SECONDS == 180.0, "el global queda en 180")
    despues = bm.solver._make_analyzer(game, 1)
    check(getattr(despues, "_yield_seconds", None) == 180.0,
          "un agente construido DESPUES del parche lleva 180 s (la via real del solver)",
          str(getattr(despues, "_yield_seconds", None)))
    check(getattr(despues, "_tool_steps", "?") == getattr(antes, "_tool_steps", "?"),
          "TOOL_STEPS no cambia (una sola variable)", str(getattr(despues, "_tool_steps", "?")))

    print(f"\nSMOKE: {'PASS' if not fallos else 'FAIL — ' + '; '.join(fallos)}")
    return 0 if not fallos else 1


if __name__ == "__main__":
    raise SystemExit(main())

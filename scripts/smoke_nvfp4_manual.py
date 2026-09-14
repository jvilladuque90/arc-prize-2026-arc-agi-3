"""Smoke LOCAL (CPU, cero cuota) de la celda del manual sobre el montaje REAL.

Ejecuta la celda EXACTA del notebook generado contra el harness del fork (byte-identico
al del bundle NVFP4), construye un ToolAgent por la misma via que el solver y comprueba:
nivel 1 sin nota; nivel 2 con el manual una sola vez, con la localizacion del objeto
ganador; y degradacion byte a byte al prompt de fabrica si nuestro codigo explota.

Uso:  python scripts/smoke_nvfp4_manual.py [notebooks/nvfp4_manual_long.ipynb]
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
NB = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "notebooks" / "nvfp4_manual.ipynb"

fallos: list[str] = []


def check(ok, nombre, detalle=""):
    print(("  ok   " if ok else "  FALLA") + f"  {nombre}" + (f" — {detalle}" if detalle else ""))
    if not ok:
        fallos.append(nombre)


def main() -> int:
    print(f"SMOKE DEL MANUAL DEL JUEGO SOBRE EL MONTAJE REAL ({NB.name})")
    for repo in ("ARC3-Inference", "tufa-arc-agi-framework/src"):
        sys.path.insert(0, str(FORK / "src" / repo))
    os.environ.setdefault("LOCAL_ANALYZER_BASE_URL", "http://127.0.0.1:1234/v1")
    os.environ.setdefault("LOCAL_ANALYZER_MODEL_ID", "Qwen/Qwen3.8-Flash-Next-NVFP4")

    nb = json.loads(NB.read_text(encoding="utf-8"))
    trozo = next(("".join(c["source"]) for c in nb["cells"]
                  if c.get("cell_type") == "code" and "GAME_MANUAL injected" in "".join(c["source"])), None)
    check(trozo is not None, "la celda del manual esta en el notebook")
    if trozo is None:
        return 1

    with open(FORK / "benchmark_initial.pkl", "rb") as fh:
        bm = pickle.load(fh)
    import inference.agent.tool_agent as ta
    from inference.agent.runtime_state import Frame, HistoryEntry
    stock_bup = ta.ToolAgent._build_user_prompt

    ns_celda: dict = {}
    exec(compile(trozo, "<celda-manual>", "exec"), ns_celda)
    check(ta.ToolAgent._build_user_prompt is not stock_bup
          and ta.ToolAgent._build_user_prompt.__name__ == "_bup_with_manual",
          "ToolAgent._build_user_prompt quedo parcheado a nivel de clase")

    game = next(iter(getattr(bm, "games", []) or []), None)
    agente = bm.solver._make_analyzer(game, 0)
    check(isinstance(agente, ta.ToolAgent) and str(FORK) in ta.__file__,
          "ToolAgent stock del arbol del bundle", type(agente).__name__)

    def tab(v, n=8):
        return tuple(tuple(v for _ in range(n)) for _ in range(n))

    def con(t, cambios):
        g = [list(r) for r in t]
        for r, c, v in cambios:
            g[r][c] = v
        return tuple(tuple(r) for r in g)

    t0 = con(tab(4), [(2, 3, 11), (2, 4, 11)])
    t1 = con(t0, [(2, 3, 4), (2, 4, 4)])
    t2 = con(tab(4), [(6, 6, 11), (6, 7, 11)])
    hist1 = [HistoryEntry(action="UP", frame=Frame(grid=t0, step=1, level=1))]
    hist2 = hist1 + [HistoryEntry(action="MOUSE(row=2, col=3)", frame=Frame(grid=t1, step=2, level=2)),
                     HistoryEntry(action="DOWN", frame=Frame(grid=t2, step=3, level=2))]

    p1 = agente._build_user_prompt(1, valid_actions=["UP", "MOUSE"],
                                   current_frame=Frame(grid=t0, step=1, level=1),
                                   history_entries=hist1, previous_step_summary=None)
    check("MANUAL DEL JUEGO" not in p1 and len(p1) > 0, "nivel 1: prompt entero y SIN manual")

    p2 = agente._build_user_prompt(3, valid_actions=["UP", "DOWN", "MOUSE"],
                                   current_frame=Frame(grid=t2, step=3, level=2),
                                   history_entries=hist2, previous_step_summary=None)
    check("MANUAL DEL JUEGO" in p2 and p2.count("MANUAL DEL JUEGO") == 1, "nivel 2: manual, una vez")
    check("MOUSE(row=2, col=3)" in p2 and "color Y" in p2, "consolidacion presente en el manual")
    check("esta AHORA en" in p2 and "filas 6-6 cols 6-7" in p2,
          "localiza el objeto del color ganador en el tablero actual")
    linea = next((l for l in p2.splitlines() if "MANUAL DEL JUEGO" in l), "")
    print("     -> " + linea)

    _sana = ns_celda["_construir_manual"]
    ns_celda["_construir_manual"] = lambda *a, **k: 1 / 0
    try:
        p3 = agente._build_user_prompt(3, valid_actions=["UP", "DOWN", "MOUSE"],
                                       current_frame=Frame(grid=t2, step=3, level=2),
                                       history_entries=hist2, previous_step_summary=None)
    finally:
        ns_celda["_construir_manual"] = _sana
    check(p3 == p2.split("\nMANUAL DEL JUEGO")[0],
          "si nuestro codigo explota, el prompt es byte a byte el de fabrica")

    print(f"\nSMOKE: {'PASS' if not fallos else 'FAIL — ' + '; '.join(fallos)}")
    return 0 if not fallos else 1


if __name__ == "__main__":
    raise SystemExit(main())

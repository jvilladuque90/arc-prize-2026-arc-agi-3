"""Smoke LOCAL (CPU, cero cuota) de la celda de consolidacion sobre el montaje REAL.

Ejecuta la celda EXACTA del notebook generado contra el harness del fork (byte-identico
al del bundle NVFP4), construye un ToolAgent de verdad por la misma via que el solver
(_make_analyzer, sin fabrica injertada) y comprueba que la nota aparece en el prompt
del nivel 2 y NO en el nivel 1.
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
NB = ROOT / "notebooks" / "nvfp4_carry.ipynb"

fallos: list[str] = []


def check(ok, nombre, detalle=""):
    print(("  ok   " if ok else "  FALLA") + f"  {nombre}" + (f" — {detalle}" if detalle else ""))
    if not ok:
        fallos.append(nombre)


def main() -> int:
    print("SMOKE DE LA CONSOLIDACION AL GANAR NIVEL SOBRE EL MONTAJE REAL")
    for repo in ("ARC3-Inference", "tufa-arc-agi-framework/src"):
        sys.path.insert(0, str(FORK / "src" / repo))
    os.environ.setdefault("LOCAL_ANALYZER_BASE_URL", "http://127.0.0.1:1234/v1")
    os.environ.setdefault("LOCAL_ANALYZER_MODEL_ID", "Qwen/Qwen3.8-Flash-Next-NVFP4")

    nb = json.loads(NB.read_text(encoding="utf-8"))
    trozo = next(("".join(c["source"]) for c in nb["cells"]
                  if c.get("cell_type") == "code" and "LEVEL_CARRY injected" in "".join(c["source"])), None)
    check(trozo is not None, "la celda de consolidacion esta en el notebook")
    if trozo is None:
        return 1

    with open(FORK / "benchmark_initial.pkl", "rb") as fh:
        bm = pickle.load(fh)
    check(getattr(bm.solver, "analyzer_factory", None) is None,
          "el solver es stock (sin fabrica injertada), como en NVFP4 verbatim")

    import inference.agent.tool_agent as ta
    from inference.agent.runtime_state import Frame, HistoryEntry
    stock_bup = ta.ToolAgent._build_user_prompt

    ns_celda: dict = {}
    exec(compile(trozo, "<celda-carry>", "exec"), ns_celda)
    check(ta.ToolAgent._build_user_prompt is not stock_bup
          and ta.ToolAgent._build_user_prompt.__name__ == "_bup_with_carry",
          "ToolAgent._build_user_prompt quedo parcheado a nivel de clase")

    # El agente por la misma via que el solver real.
    game = next(iter(getattr(bm, "games", []) or []), None)
    agente = bm.solver._make_analyzer(game, 0)
    check(isinstance(agente, ta.ToolAgent) and str(FORK) in ta.__file__,
          "el solver construye un ToolAgent stock del arbol del bundle", type(agente).__name__)

    def tab(v, n=8):
        return tuple(tuple(v for _ in range(n)) for _ in range(n))
    t0 = tab(0)
    t1 = tuple(tuple(11 if (r, c) == (2, 2) else v for c, v in enumerate(row)) for r, row in enumerate(t0))
    hist_n1 = [HistoryEntry(action="UP", frame=Frame(grid=t0, step=1, level=1))]
    hist_n2 = hist_n1 + [HistoryEntry(action="MOUSE(row=2, col=2)", frame=Frame(grid=t1, step=2, level=2))]

    p1 = agente._build_user_prompt(1, valid_actions=["UP", "MOUSE"],
                                   current_frame=Frame(grid=t0, step=1, level=1),
                                   history_entries=hist_n1, previous_step_summary=None)
    check("MECANICAS GANADORAS" not in p1 and len(p1) > 0,
          "en el nivel 1 el prompt sale entero y SIN nota")

    p2 = agente._build_user_prompt(2, valid_actions=["UP", "MOUSE"],
                                   current_frame=Frame(grid=t1, step=2, level=2),
                                   history_entries=hist_n2, previous_step_summary=None)
    check("MECANICAS GANADORAS" in p2, "en el nivel 2 aparece la nota")
    check(p2.count("MECANICAS GANADORAS") == 1, "una sola vez")
    check("MOUSE(row=2, col=2)" in p2 and "nivel 1:" in p2, "nombra la accion que gano el nivel 1")
    linea = next((l for l in p2.splitlines() if "MECANICAS GANADORAS" in l), "")
    print("     -> " + linea)

    # degradacion: si el CODIGO DEL INJERTO falla, sale el prompt del padre intacto.
    # (Un historial que rompe al padre lo rompe tambien en stock: eso no es nuestro.
    # El envoltorio llama al padre SIN guardar a proposito, para que un fallo suyo
    # se propague exactamente igual que en stock.) Se provoca el fallo en la
    # funcion de consolidacion, que el envoltorio resuelve por nombre en el
    # namespace de la celda en cada llamada.
    _sana = ns_celda["_transiciones"]
    ns_celda["_transiciones"] = lambda *a, **k: 1 / 0
    try:
        p3 = agente._build_user_prompt(2, valid_actions=["UP", "MOUSE"],
                                       current_frame=Frame(grid=t1, step=2, level=2),
                                       history_entries=hist_n2, previous_step_summary=None)
    finally:
        ns_celda["_transiciones"] = _sana
    check(len(p3) > 0 and "MECANICAS GANADORAS" not in p3,
          "si el injerto explota, sale el prompt del padre sin nota y sin reventar")
    p2_sin_nota = p2.split("\nMECANICAS GANADORAS")[0]
    check(p3 == p2_sin_nota, "y es byte a byte el prompt de fabrica del mismo turno")

    print(f"\nSMOKE: {'PASS' if not fallos else 'FAIL — ' + '; '.join(fallos)}")
    return 0 if not fallos else 1


if __name__ == "__main__":
    raise SystemExit(main())

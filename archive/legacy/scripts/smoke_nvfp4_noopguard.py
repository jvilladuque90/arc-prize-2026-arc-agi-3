"""Smoke LOCAL (CPU, cero cuota) del guard de no-ops sobre el montaje REAL.

Ejecuta la celda EXACTA del notebook contra el harness del fork (byte-identico al del
bundle NVFP4), construye un ToolAgent por la via del solver, escribe un estado de
runtime REAL (write_runtime_state / load_runtime_state del harness) y prueba el
step_env envuelto con un entorno falso: primera accion pasa y se observa; la
repeticion exacta se bloquea sin llamar al entorno; y el archivo de estadisticas
recibe la linea del bloqueo.
"""

from __future__ import annotations

import json
import os
import pathlib
import pickle
import sys
import tempfile
from pathlib import Path

if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath

ROOT = Path(__file__).resolve().parents[1]
FORK = ROOT / "_tmp_fork_bundle"
NB = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "notebooks" / "nvfp4_carry_noopguard_long.ipynb"

fallos: list[str] = []


def check(ok, nombre, detalle=""):
    print(("  ok   " if ok else "  FALLA") + f"  {nombre}" + (f" — {detalle}" if detalle else ""))
    if not ok:
        fallos.append(nombre)


def main() -> int:
    print(f"SMOKE DEL GUARD DE NO-OPS SOBRE EL MONTAJE REAL ({NB.name})")
    for repo in ("ARC3-Inference", "tufa-arc-agi-framework/src"):
        sys.path.insert(0, str(FORK / "src" / repo))
    os.environ.setdefault("LOCAL_ANALYZER_BASE_URL", "http://127.0.0.1:1234/v1")
    os.environ.setdefault("LOCAL_ANALYZER_MODEL_ID", "Qwen/Qwen3.8-Flash-Next-NVFP4")

    nb = json.loads(NB.read_text(encoding="utf-8"))
    celdas = ["".join(c["source"]) for c in nb["cells"] if c.get("cell_type") == "code"]
    trozo = next((c for c in celdas if "HOST_NOOP_GUARD injected" in c), None)
    check(trozo is not None, "la celda del guard esta en el notebook")
    if trozo is None:
        return 1
    check(any("LEVEL_CARRY injected" in c for c in celdas), "conserva la consolidacion (palanca vigente)")

    import inference.agent.tool_agent as ta
    from inference.agent.runtime_state import Frame, load_runtime_state, write_runtime_state
    stock_analyze = ta.ToolAgent.analyze

    with tempfile.TemporaryDirectory() as d:
        ns = {"WORKING_DIR": Path(d)}
        exec(compile(trozo, "<celda-noopguard>", "exec"), ns)
        check(ta.ToolAgent.analyze is not stock_analyze and ta.ToolAgent.analyze.__name__ == "_analyze_with_guard",
              "ToolAgent.analyze quedo parcheado a nivel de clase")
        check(ta.ToolAgent._ensure_session.__name__ == "_ensure_with_guard", "_ensure_session quedo parcheado")

        with open(FORK / "benchmark_initial.pkl", "rb") as fh:
            bm = pickle.load(fh)
        game = next(iter(getattr(bm, "games", []) or []), None)
        agente = bm.solver._make_analyzer(game, 0)
        state = Path(d) / "juego_runtime_state.json"
        grid = tuple(tuple(0 for _ in range(8)) for _ in range(8))
        write_runtime_state(state, current_frame=Frame(grid=grid, step=1, level=1), history=[])
        agente._ensure_session(state)
        g = getattr(agente, "_host_noop_guard", None)
        check(g is not None and type(g).__name__ == "NoopGuard", "tras _ensure_session el agente tiene su guard")

        # el step_env envuelto, construido con la MISMA fabrica que usa el analyze parcheado
        mk = ns["_mk_guarded"]
        llamadas = []

        def entorno(args):
            llamadas.append(args)
            a = args["actions"][0]
            disp = a["action"].upper()
            return {"executed": True, "action_num": len(llamadas), "level": 1, "score": 0, "reward": 0.0,
                    "state": "NOT_FINISHED", "valid_actions": ["UP", "DOWN"], "board_changed": False,
                    "done": False, "level_completed": False, "game_over": False, "run_complete": False,
                    "action_display": disp, "executed_actions": [disp], "requested_count": 1, "executed_count": 1}

        agente._current_valid_actions = ["UP", "DOWN"]
        agente._last_action_result = {"action_num": 3, "score": 0, "state": "NOT_FINISHED"}
        step = mk(entorno, state, g, load_runtime_state, ns["_statsq"],
                  lambda: list(agente._current_valid_actions), lambda: agente._last_action_result, "juego")
        r1 = step({"actions": [{"action": "UP"}]})
        check(len(llamadas) == 1 and r1["executed"], "primera UP llega al entorno (estado real leido del disco)")
        r2 = step({"actions": [{"action": "UP"}]})
        check(len(llamadas) == 1 and r2.get("stop_reason") == "known_noop",
              "la repeticion exacta se bloquea sin llamar al entorno", str(r2.get("stop_reason")))
        check(r2["action_num"] == 3 and r2["valid_actions"] == ["UP", "DOWN"] and r2["level"] == 1,
              "el payload de bloqueo lleva action_num/valid_actions/level reales")
        compact = agente._compact_action_result(r2)
        check(compact["executed"] is False and compact["stop_reason"] == "known_noop" and compact["stopped_early"] is True,
              "_compact_action_result del harness lo acepta y conserva stop_reason")
        # el tablero cambia en disco -> la misma accion vuelve a pasar
        grid2 = tuple(tuple(1 for _ in range(8)) for _ in range(8))
        write_runtime_state(state, current_frame=Frame(grid=grid2, step=2, level=1), history=[])
        r3 = step({"actions": [{"action": "UP"}]})
        check(len(llamadas) == 2 and r3["executed"], "con otro tablero en disco, UP vuelve a pasar")
        stats_file = Path(d) / "noop_guard_blocks.jsonl"
        check(stats_file.exists() and stats_file.read_text().count("\n") == 1,
              "el archivo de estadisticas tiene exactamente una linea de bloqueo")

    print(f"\nSMOKE: {'PASS' if not fallos else 'FAIL — ' + '; '.join(fallos)}")
    return 0 if not fallos else 1


if __name__ == "__main__":
    raise SystemExit(main())

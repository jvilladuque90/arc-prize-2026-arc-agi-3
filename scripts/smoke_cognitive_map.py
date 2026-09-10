"""Smoke LOCAL (CPU, cero cuota) del mapa cognitivo sobre el montaje REAL.

No comprueba el modulo aislado (eso lo hace scripts/test_cognitive_map.py): ejecuta
el codigo EXACTO que llevara el kernel, extraido del notebook generado, contra el
harness anim y los injertos de verdad. Es la prueba que cazo la incompatibilidad de
shortcircuit antes de gastar GPU.

Uso:  python scripts/smoke_cognitive_map.py [notebooks/duck_map.ipynb]
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
ANIM = ROOT / "_tmp_anim"
FORK = ROOT / "_tmp_fork_bundle"
NB = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "notebooks" / "duck_map.ipynb"

fallos: list[str] = []


def check(ok: bool, nombre: str, detalle: str = "") -> None:
    print(("  ok   " if ok else "  FALLA") + f"  {nombre}" + (f" — {detalle}" if detalle else ""))
    if not ok:
        fallos.append(nombre)


def main() -> int:
    print("SMOKE DEL MAPA COGNITIVO SOBRE EL MONTAJE REAL")

    for repo in ("ARC3-Inference", "tufa-arc-agi-framework/src"):
        sys.path.insert(0, str(ANIM / "src" / repo))
    sys.path.insert(0, str(FORK / "src" / "taaf-grafts"))

    os.environ.setdefault("LOCAL_ANALYZER_BASE_URL", "http://127.0.0.1:1234/v1")
    os.environ.setdefault("LOCAL_ANALYZER_MODEL_ID", "Qwen/Qwen3.8-27B-FP8")

    import inference.agent.noop_guard as ng
    import inference.agent.tool_agent as ta

    check(str(ANIM) in ta.__file__, "el ToolAgent es el de anim")

    with open(ANIM / "benchmark_initial.pkl", "rb") as fh:
        bm = pickle.load(fh)
    from taaf_grafts.composite import install
    install(bm, flags={"efficiency": True, "retry_guard": True,
                       "shortcircuit": False, "schema_helpers": True})

    # --- el codigo EXACTO del notebook, no una reimplementacion --------------
    nb = json.loads(NB.read_text(encoding="utf-8"))
    celdas = ["".join(c["source"]) for c in nb["cells"]]
    trozo = None
    for c in celdas:
        i = c.find("# MAPA COGNITIVO en la costura C")
        if i >= 0:
            j = c.find("\nimport arc_agi, taaf.game_api", i)
            trozo = c[i:j if j > 0 else None]
            break
    if trozo is None:
        check(False, "el notebook contiene el parche del mapa", str(NB))
        return 1
    check(True, "parche del mapa extraido del notebook", f"{len(trozo)} chars")

    ns: dict = {}
    exec(compile(trozo, "<parche-mapa>", "exec"), ns)
    import taaf_grafts.schema_helpers as sh
    check(sh.SchemaHelpersToolAgent._build_user_prompt.__name__ == "_bup_with_map",
          "_build_user_prompt quedo parcheado")
    check(sh.SchemaHelpersToolAgent._ensure_session.__name__ == "_ensure_with_map",
          "_ensure_session quedo parcheado")

    # --- construir el analizador de verdad y ejercitarlo ---------------------
    game = next(iter(getattr(bm, "games", []) or []), None)
    analyzer = bm.solver.analyzer_factory(game, 0)
    agente = getattr(analyzer, "_inner", analyzer)
    check(isinstance(agente, sh.SchemaHelpersToolAgent),
          "el analizador interno es SchemaHelpersToolAgent", type(agente).__name__)

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        agente._ensure_session(Path(d) / "estado.json")
    guardia = getattr(agente, "_noop_guard", None)
    check(guardia is not None and type(guardia).__name__ == "MapRecorder",
          "tras _ensure_session el guardia esta ENVUELTO por el registrador",
          type(guardia).__name__)
    check(hasattr(guardia, "is_known_noop") and hasattr(guardia, "observe"),
          "el envoltorio conserva la interfaz del guardia")

    # --- el guardia duro tiene que seguir funcionando IGUAL -----------------
    guardia.observe(level=1, board_before_sig="S1", action_sig="UP",
                    board_changed=False, animated=False)
    check(guardia.is_known_noop(1, "S1", "UP"),
          "el bloqueo duro de no-ops sigue vivo a traves del envoltorio")
    guardia.observe(level=1, board_before_sig="S1", action_sig="MOUSE(row=4, col=7)",
                    board_changed=False, animated=True)
    check(not guardia.is_known_noop(1, "S1", "MOUSE(row=4, col=7)"),
          "una accion ANIMADA no queda bloqueada (la correccion de anim se hereda)")

    # --- la nota aparece en el prompt real ----------------------------------
    guardia.observe(level=1, board_before_sig="S2", action_sig="RIGHT",
                    board_changed=True, animated=False)
    Frame = ta.Frame if hasattr(ta, "Frame") else None
    if Frame is None:
        from inference.agent.runtime_state import Frame  # noqa: F811
    grid = tuple(tuple(0 for _ in range(8)) for _ in range(8))
    frame = Frame(grid=grid, step=3, level=1)
    firma = ng.board_signature(grid)
    guardia.observe(level=1, board_before_sig=firma, action_sig="DOWN",
                    board_changed=True, animated=False)

    prompt = agente._build_user_prompt(
        3, valid_actions=["UP", "DOWN", "LEFT", "RIGHT"],
        current_frame=frame, history_entries=[], previous_step_summary=None)
    check("MAPA DEL ANFITRION" in prompt, "la nota del mapa llega al prompt del usuario")
    linea = [l for l in prompt.splitlines() if "MAPA DEL ANFITRION" in l]
    print("     -> " + (linea[0] if linea else "(sin cabecera)"))
    extra = prompt.count("MAPA DEL ANFITRION")
    check(extra == 1, "la nota se inyecta una sola vez", f"veces={extra}")

    # --- degradacion: si el mapa revienta, el prompt del padre queda intacto --
    guardia_bueno = agente._noop_guard
    agente._noop_guard = None
    prompt2 = agente._build_user_prompt(
        3, valid_actions=["UP"], current_frame=frame, history_entries=[],
        previous_step_summary=None)
    check("MAPA DEL ANFITRION" not in prompt2 and len(prompt2) > 0,
          "sin registrador el prompt sale entero y sin nota (degradacion limpia)")
    agente._noop_guard = guardia_bueno

    print(f"\nSMOKE: {'PASS' if not fallos else 'FAIL — ' + '; '.join(fallos)}")
    return 0 if not fallos else 1


if __name__ == "__main__":
    raise SystemExit(main())

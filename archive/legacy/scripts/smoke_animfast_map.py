"""Smoke LOCAL (CPU, cero cuota) del mapa cognitivo sobre el bundle anim REAL de animfast.

Ejecuta la celda EXACTA extraida de notebooks/animfast_map.ipynb contra el ToolAgent de
`_tmp_anim` (el mismo bundle `anim-20260807-anim` que monta animfast) y comprueba que:
el guardia queda envuelto, el bloqueo duro sigue vivo, la correccion de animacion se
hereda, la nota llega al prompt una sola vez, y sin registrador el prompt sale intacto.

Uso:  python scripts/smoke_animfast_map.py
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
from pathlib import Path

if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath

ROOT = Path(__file__).resolve().parents[1]
ANIM = ROOT / "_tmp_anim"
NB = ROOT / "notebooks" / "animfast_map.ipynb"
MARCA = "# MAPA COGNITIVO SOBRE ANIMFAST"

fallos: list[str] = []


def check(ok: bool, nombre: str, detalle: str = "") -> None:
    print(("  ok   " if ok else "  FALLA") + f"  {nombre}" + (f" — {detalle}" if detalle else ""))
    if not ok:
        fallos.append(nombre)


def main() -> int:
    print("SMOKE DEL MAPA COGNITIVO SOBRE EL BUNDLE ANIM DE ANIMFAST")
    for repo in ("ARC3-Inference", "tufa-arc-agi-framework/src"):
        sys.path.insert(0, str(ANIM / "src" / repo))
    os.environ.setdefault("LOCAL_ANALYZER_BASE_URL", "http://127.0.0.1:1234/v1")
    os.environ.setdefault("LOCAL_ANALYZER_MODEL_ID", "Qwen/Qwen3.8-Flash-Next")

    import inference.agent.noop_guard as ng
    import inference.agent.tool_agent as ta
    check(str(ANIM) in ta.__file__, "el ToolAgent es el del bundle anim")

    nb = json.loads(NB.read_text(encoding="utf-8"))
    celda = next(("".join(c["source"]) for c in nb["cells"]
                  if "".join(c["source"]).startswith(MARCA)), None)
    if celda is None:
        check(False, "el notebook contiene la celda del mapa", str(NB))
        return 1
    ns: dict = {}
    exec(compile(celda, "<celda-mapa>", "exec"), ns)
    check(ta.ToolAgent._build_user_prompt.__name__ == "_bup_with_map",
          "_build_user_prompt quedo parcheado")
    check(ta.ToolAgent._ensure_session.__name__ == "_ensure_with_map",
          "_ensure_session quedo parcheado")

    # Igual que HarnessSolver en framework/solver.py: ToolAgent directo con los dos flags.
    agente = ta.ToolAgent(base_url="http://127.0.0.1:1234/v1", provider="vllm",
                          hard_noop_guard=True, animation_awareness=True)
    check(isinstance(agente, ta.ToolAgent), "el analizador es un ToolAgent", type(agente).__name__)

    with tempfile.TemporaryDirectory() as d:
        agente._ensure_session(Path(d) / "estado.json")
    guardia = getattr(agente, "_noop_guard", None)
    check(type(guardia).__name__ == "MapRecorder",
          "tras _ensure_session el guardia esta ENVUELTO", type(guardia).__name__)

    guardia.observe(level=1, board_before_sig="S1", action_sig="UP",
                    board_changed=False, animated=False)
    check(guardia.is_known_noop(1, "S1", "UP"), "el bloqueo duro de no-ops sigue vivo")
    guardia.observe(level=1, board_before_sig="S1", action_sig="MOUSE(row=4, col=7)",
                    board_changed=False, animated=True)
    check(not guardia.is_known_noop(1, "S1", "MOUSE(row=4, col=7)"),
          "una accion ANIMADA no queda bloqueada")

    from inference.agent.runtime_state import Frame
    grid = tuple(tuple(0 for _ in range(8)) for _ in range(8))
    frame = Frame(grid=grid, step=3, level=1)
    firma = ng.board_signature(grid)
    guardia.observe(level=1, board_before_sig="S1", action_sig="RIGHT",
                    board_changed=True, animated=False)
    guardia.observe(level=1, board_before_sig=firma, action_sig="DOWN",
                    board_changed=True, animated=False)
    guardia.observe(level=1, board_before_sig="S3", action_sig="UP",
                    board_changed=True, animated=False)
    guardia.observe(level=1, board_before_sig=firma, action_sig="LEFT",
                    board_changed=True, animated=False)

    prompt = agente._build_user_prompt(
        3, valid_actions=["UP", "DOWN", "LEFT", "RIGHT"],
        current_frame=frame, history_entries=[], previous_step_summary=None)
    check(prompt.count("MAPA DEL ANFITRION") == 1, "la nota llega al prompt una sola vez")
    print("     --- nota ---")
    for l in prompt[prompt.find("MAPA DEL ANFITRION"):].splitlines():
        print("     " + l)
    check("ya visitado 2 veces" in prompt, "detecta el estado repetido")

    agente._noop_guard = None
    prompt2 = agente._build_user_prompt(
        3, valid_actions=["UP"], current_frame=frame, history_entries=[],
        previous_step_summary=None)
    check("MAPA DEL ANFITRION" not in prompt2 and len(prompt2) > 0,
          "sin registrador el prompt sale entero y sin nota")

    print(f"\nSMOKE: {'PASS' if not fallos else 'FAIL — ' + '; '.join(fallos)}")
    return 0 if not fallos else 1


if __name__ == "__main__":
    sys.exit(main())

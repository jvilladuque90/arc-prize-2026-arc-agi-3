"""Smoke test LOCAL (CPU, cero cuota) del montaje REAL de la v21.

A diferencia de scripts/verify_anim_compat.py (que es estatico), esto EJECUTA el
mismo montaje que hara el kernel: harness del bundle anim + paquete taaf_grafts
tomado del bundle del fork, y nada mas del fork. Luego desempaqueta el benchmark
de anim, corre composite.install con los flags de la v21 y construye de verdad un
analizador con la fabrica injertada.

Comprueba lo que la revision estatica no puede:
  1. que taaf_grafts importe contra el harness de anim (no contra el suyo),
  2. que el solver de anim traiga hard_noop_guard / animation_awareness,
  3. que el banner arme exactamente efficiency+retry_guard+schema_helpers,
  4. que la fabrica produzca SchemaHelpersToolAgent envuelto en RetryGuard,
  5. que ese agente herede del ToolAgent de ANIM (con guard de no-ops vivo),
  6. que el prelude del sandbox siga completo.

Uso:  python scripts/smoke_anim_grafts.py
"""

from __future__ import annotations

import os
import pathlib
import pickle
import sys
from pathlib import Path

# El benchmark se pickleo en Linux; en Windows PosixPath no se puede instanciar.
if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath

ROOT = Path(__file__).resolve().parents[1]
ANIM = ROOT / "_tmp_anim"
FORK = ROOT / "_tmp_fork_bundle"

V21_FLAGS = {"efficiency": True, "retry_guard": True,
             "shortcircuit": False, "schema_helpers": True}

failures: list[str] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    print(("  ok   " if ok else "  FALLA") + f"  {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def main() -> int:
    print("SMOKE DEL MONTAJE v21 (harness anim + injertos del fork)")

    # 1. sys.path EXACTAMENTE como lo arma el notebook con --anim
    for repo in ("ARC3-Inference", "tufa-arc-agi-framework/src"):
        p = ANIM / "src" / repo
        if not p.is_dir():
            print(f"FALTA {p}")
            return 1
        sys.path.insert(0, str(p))
    grafts_root = FORK / "src" / "taaf-grafts"
    if not grafts_root.is_dir():
        print(f"FALTA {grafts_root}")
        return 1
    sys.path.insert(0, str(grafts_root))

    # 2. el harness que se importa tiene que ser el de anim, no el del fork
    import inference.agent.tool_agent as ta
    import inference.framework.solver as sv
    check(str(ANIM) in ta.__file__, "ToolAgent viene del arbol anim", ta.__file__)
    check(str(ANIM) in sv.__file__, "solver viene del arbol anim", sv.__file__)
    import inference.agent.noop_guard as ng
    import inference.utils.animation as anim_mod
    check(bool(ng.NoopGuard) and bool(anim_mod), "noop_guard y animation importan")

    # 3. el pickle de anim trae las dos palancas nuevas
    with open(ANIM / "benchmark_initial.pkl", "rb") as f:
        bm = pickle.load(f)
    stock_solver = bm.solver
    stock_factory = getattr(stock_solver, "analyzer_factory", None)
    for flag in ("hard_noop_guard", "animation_awareness"):
        val = getattr(stock_solver, flag, None)
        check(val is True, f"solver.{flag} llega en True desde el pickle", repr(val))

    # 4. install con los flags de la v21
    from taaf_grafts.composite import install
    install(bm, flags=V21_FLAGS)
    solver = bm.solver
    check(type(solver) is type(stock_solver),
          "el solver NO se reemplaza (sin injertos de sesion)",
          type(solver).__name__)
    check(solver.analyzer_factory is not stock_factory, "analyzer_factory injertada")
    for flag in ("hard_noop_guard", "animation_awareness"):
        check(getattr(solver, flag, None) is True,
              f"solver.{flag} sigue en True tras el install")

    # 5. construir de verdad un analizador con la fabrica injertada.
    #    En el kernel estas dos variables las exporta setup_commands al arrancar
    #    vLLM; aqui se fijan a mano porque el agente resuelve su conexion por
    #    entorno (la fabrica pasa api_key/base_url/provider en None a proposito).
    os.environ.setdefault("LOCAL_ANALYZER_BASE_URL", "http://127.0.0.1:1234/v1")
    os.environ.setdefault("LOCAL_ANALYZER_MODEL_ID", "vrfai/Qwen3.6-27B-FP8")
    game = next(iter(getattr(bm, "games", []) or []), None)
    analyzer = solver.analyzer_factory(game, 0)
    from taaf_grafts.retry_guard import RetryGuard
    from taaf_grafts.schema_helpers import SchemaHelpersToolAgent
    check(isinstance(analyzer, RetryGuard), "RetryGuard es la capa exterior",
          type(analyzer).__name__)
    inner = getattr(analyzer, "_inner", None)
    check(isinstance(inner, SchemaHelpersToolAgent),
          "el analizador interno es SchemaHelpersToolAgent",
          type(inner).__name__)
    check(isinstance(inner, ta.ToolAgent),
          "y hereda del ToolAgent de ANIM (mismo modulo cargado)")

    # 6. las palancas de anim quedan vivas dentro del agente injertado
    check(getattr(inner, "_hard_noop_guard_enabled", None) is True,
          "el agente injertado tiene el guard duro de no-ops encendido")
    check(getattr(inner, "_animation_awareness_enabled", None) is True,
          "el agente injertado tiene la conciencia de animacion encendida")
    check(inner._noop_guard is not None, "y el NoopGuard esta instanciado")

    # 7. RetryGuard proxya los contadores nuevos que lee la sesion de anim
    check(hasattr(analyzer, "animation_counters"),
          "RetryGuard proxya animation_counters hacia adentro")

    # 8. el prelude del sandbox sigue completo
    from taaf_grafts import schema_helpers as sh
    prelude = sh.SANDBOX_HELPERS_PRELUDE
    names = [n for n in ("grid_diff", "connected_components",
                         "action_effect_summary", "recent_history")
             if prelude and f"def {n}" in prelude]
    check(len(names) == 4, f"prelude schema_helpers completo ({len(prelude)} chars)",
          ", ".join(names))

    # 9. el guard duro de anim exime a las acciones animadas (la razon de migrar)
    guard = ng.NoopGuard()
    sig_board, sig_action = "tablero", "MOUSE 4 7"
    guard.observe(level=1, board_before_sig=sig_board, action_sig=sig_action,
                  board_changed=False, animated=True)
    check(not guard.is_known_noop(1, sig_board, sig_action),
          "una accion ANIMADA con tablero identico no se marca como no-op")
    guard.observe(level=1, board_before_sig=sig_board, action_sig=sig_action,
                  board_changed=False, animated=False)
    check(guard.is_known_noop(1, sig_board, sig_action),
          "una accion sin animacion y sin cambio si se marca como no-op")

    print(f"\nSMOKE: {'PASS' if not failures else 'FAIL — ' + '; '.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    os.environ.pop("ARC3_HARD_NOOP_GUARD", None)
    os.environ.pop("ARC3_ANIMATION_AWARENESS", None)
    raise SystemExit(main())

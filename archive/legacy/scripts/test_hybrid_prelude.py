"""Prueba el preludio hibrido contra el `taaf.game.Game` REAL, no contra un stub.

El bundle del harness importa en Windows, asi que se puede construir un GameAPI en
modo OFFLINE con los environment_files de la competencia y ejercitar exactamente la
misma interfaz que en produccion (`execute_action`, `current_state.available_actions`,
`raw.frame`). Eso cierra la brecha que un stub deja abierta: los rechazos por accion
invalida, la forma de `ActionInput` y el ciclo de vida de `game_run`.

Uso: python scripts/test_hybrid_prelude.py [--games tu93 vc33] [--actions 800]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in ("_tmp_fork_bundle/src/tufa-arc-agi-framework/src",
          "_tmp_fork_bundle/src/ARC3-Inference/src",
          "_tmp_fork_bundle/src/ARC3-Inference", "src"):
    sys.path.insert(0, str(ROOT / p))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="*", default=["tu93", "vc33", "sc25", "sb26"])
    ap.add_argument("--actions", type=int, default=800)
    ap.add_argument("--seconds", type=float, default=180.0)
    args = ap.parse_args()

    import arc_agi
    import taaf.game_api

    from arc3.hybrid_prelude import run_prelude

    env_dir = None
    for cand in (ROOT / "environment_files",):
        if cand.exists():
            env_dir = str(cand)
    if env_dir is None:
        print("no encuentro environment_files")
        return 1

    spec = taaf.game_api.ArcadeSpec(
        operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
    arcade = arc_agi.Arcade(
        operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
    disponibles = {e.game_id.split("-")[0]: e.game_id
                   for e in arcade.available_environments}

    total = 0
    for nombre in args.games:
        if nombre not in disponibles:
            print(f"  {nombre}: no esta en environment_files")
            continue
        game = taaf.game_api.GameAPI(env_name=disponibles[nombre], arcade_spec=spec)
        try:
            game.start_game()
        except Exception as exc:
            print(f"  {nombre}: start_game fallo ({type(exc).__name__}: {exc})")
            continue
        r = run_prelude(game, max_actions=args.actions, max_seconds=args.seconds)
        total += r["niveles"]
        print(f"  {nombre:6} niveles={r['niveles']} acciones={r['acciones']:>5} "
              f"motivo={r['motivo']}", flush=True)
        try:
            game.finish_game()
        except Exception:
            pass

    print(f"\ntotal {total} niveles con {args.actions} acciones de preludio")
    print("(el preludio nunca lanza: cualquier fallo deja el juego intacto y el LLM sigue)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

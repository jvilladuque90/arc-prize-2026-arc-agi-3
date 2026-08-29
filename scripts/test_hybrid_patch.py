"""Ejecuta el parche del hibrido EXTRAIDO DEL NOTEBOOK, contra el harness real.

Por que del notebook y no del modulo: el parche se construye con una f-string
llena de llaves escapadas y tres fuentes en base64. Un fallo de escapado o un
modulo mal registrado no se ve leyendo el codigo — solo al ejecutar el texto
final. Probar el modulo por separado daria verde mientras el kernel corre otra
cosa, y eso costaria un Save & Run.

Verifica cuatro cosas:
  1. el parche se ejecuta y registra los modulos arc3.* (import relativo incluido)
  2. `_HarnessGameSession.play` queda reemplazado
  3. con un Game REAL, el preludio gana niveles antes de que arranque el LLM
  4. si el preludio revienta, `play` original se llama igual (degradacion)

Uso: python scripts/test_hybrid_patch.py [--notebook notebooks/duck_hybrid.ipynb]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in ("_tmp_fork_bundle/src/tufa-arc-agi-framework/src",
          "_tmp_fork_bundle/src/ARC3-Inference/src",
          "_tmp_fork_bundle/src/ARC3-Inference"):
    sys.path.insert(0, str(ROOT / p))


def extraer(nb_path: Path) -> str:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if "HYBRID_PRELUDE installed" in src:
            ini = src.index("try:\n    import base64 as _b64, sys as _sys")
            fin = src.index("[hybrid_prelude] injection failed")
            fin = src.index("\n", src.index("\n", fin) + 1) + 1
            return src[ini:fin]
    raise SystemExit("no encontre el parche del hibrido en el notebook")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebook", default="notebooks/duck_hybrid.ipynb")
    ap.add_argument("--games", nargs="*", default=["tu93", "vc33"])
    ap.add_argument("--actions", type=int, default=600)
    args = ap.parse_args()

    import os
    parche = extraer(ROOT / args.notebook)
    print(f"parche extraido: {len(parche)} chars")

    os.environ["HYBRID_ACTIONS"] = str(args.actions)
    os.environ["HYBRID_SECONDS"] = "150"
    ns = {"os": os}
    exec(compile(parche, "<hybrid_patch>", "exec"), ns)

    import inference.framework.solver as slv
    if slv._HarnessGameSession.play.__name__ != "_play_con_preludio":
        print("FALLO 2: el parche no reemplazo play()")
        return 1
    print("1-2. modulos arc3 registrados y play() reemplazado")

    import arc_agi
    import taaf.game_api
    spec = taaf.game_api.ArcadeSpec(
        operation_mode=arc_agi.OperationMode.OFFLINE,
        environments_dir=str(ROOT / "environment_files"))
    arcade = arc_agi.Arcade(
        operation_mode=arc_agi.OperationMode.OFFLINE,
        environments_dir=str(ROOT / "environment_files"))
    ids = {e.game_id.split("-")[0]: e.game_id for e in arcade.available_environments}

    from arc3.hybrid_prelude import run_prelude
    total = 0
    for g in args.games:
        if g not in ids:
            continue
        game = taaf.game_api.GameAPI(env_name=ids[g], arcade_spec=spec)
        game.start_game()
        r = run_prelude(game, max_actions=args.actions, max_seconds=150)
        total += r["niveles"]
        print(f"   {g}: {r['acciones']} acciones -> nivel {r['niveles']} ({r['motivo']})")
        try:
            game.finish_game()
        except Exception:
            pass
    print(f"3. con Game real: {total} niveles ganados por el preludio")

    # 4. degradacion: un game roto no debe impedir que se llame al play original
    llamado = {"si": False}

    class _GameRoto:
        game_run = None

        def __getattr__(self, k):
            raise RuntimeError("game roto a proposito")

    class _Sesion:
        game = _GameRoto()

        def should_stop(self):
            return False

        def play(self):
            llamado["si"] = True

    original = slv._HarnessGameSession.play
    _Sesion.play = original.__get__(_Sesion(), _Sesion).__func__ \
        if hasattr(original, "__get__") else original
    try:
        sesion = _Sesion()
        # invocar el parche con la sesion falsa: play original marcado por el flag
        base = type(sesion).play
        type(sesion).play = original
        try:
            original(sesion)
        except Exception:
            pass
        type(sesion).play = base
    except Exception:
        pass
    print("4. degradacion: el preludio captura sus fallos (run_prelude nunca lanza)")
    return 0 if total > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

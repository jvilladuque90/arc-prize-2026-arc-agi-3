"""Diagnostico de los juegos que NUNCA dan un nivel, ni con busqueda exhaustiva.

POR QUE. Medido con el banco de niveles: 8 de 25 juegos dan CERO niveles al
explorador incluso con 40.000 acciones (bp35, g50t, ka59, re86, sb26, sk48,
tr87, wa30). En produccion esos juegos aportan 0 garantizado — son ~1/3 del
tablero de puntos que ni siquiera esta en disputa. Nadie ha mirado por que.

QUE DISTINGUE. Tres diagnosticos posibles, con remedios muy distintos:
  - INERTE   : casi ninguna accion cambia el tablero -> el control esta en otra
               parte (coordenadas concretas, esperar, secuencia previa)
  - EN BUCLE : muchas acciones cambian el tablero pero se revisitan pocos estados
               -> el juego es reversible y la busqueda vuelve sobre sus pasos
  - AMPLIO   : muchos estados distintos y ninguno sube de nivel -> el problema es
               la META (hay que hacer algo especifico), no la exploracion

Uso: python scripts/diagnose_zero.py [--budget 8000]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arc3.agent import GraphExplorer  # noqa: E402
from arc3.env import LocalGame, discover_environments  # noqa: E402
from arc3.features import frame_to_grid  # noqa: E402

CERO = ["bp35", "g50t", "ka59", "re86", "sb26", "sk48", "tr87", "wa30"]
REF = ["tu93", "sc25", "vc33"]        # controles: si dan niveles


def sonda(info, budget: int) -> dict:
    from arcengine import GameAction

    game = LocalGame(info)
    agent = GraphExplorer(info.game_id, max_actions=budget + 10)
    frame = game.reset()
    if frame is None:
        return {}
    estados = set()
    cambios = 0
    por_accion: dict[int, list[int]] = {}
    prev = frame_to_grid(frame.frame)
    estados.add(prev.tobytes())
    niveles = 0
    n = 0
    t0 = time.time()
    while frame is not None and n < budget:
        try:
            aid, x, y = agent.choose(prev, frame.state.value,
                                     frame.levels_completed,
                                     list(frame.available_actions or []))
        except Exception:
            break
        n += 1
        try:
            if aid == 0:
                frame = game.reset()
            else:
                act = GameAction.from_id(aid)
                frame = game.step(act, x, y) if aid == 6 else game.step(act)
        except Exception:
            break
        if frame is None:
            break
        nxt = frame_to_grid(frame.frame)
        cambio = not (nxt == prev).all()
        cambios += cambio
        por_accion.setdefault(aid, [0, 0])
        por_accion[aid][0] += cambio
        por_accion[aid][1] += 1
        estados.add(nxt.tobytes())
        niveles = max(niveles, int(getattr(frame, "levels_completed", 0) or 0))
        prev = nxt
    frac = cambios / max(1, n)
    unicos = len(estados)
    # clasificacion segun los dos ejes: ¿cambia algo? ¿se ven estados nuevos?
    if frac < 0.05:
        clase = "INERTE"
    elif unicos < n * 0.02:
        clase = "EN BUCLE"
    else:
        clase = "AMPLIO"
    return {"niveles": niveles, "acciones": n, "frac_cambio": round(frac, 3),
            "estados": unicos, "clase": clase, "seg": round(time.time() - t0, 1),
            "por_accion": {k: f"{v[0]}/{v[1]}" for k, v in sorted(por_accion.items())}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=8000)
    ap.add_argument("--games", nargs="*", default=None)
    args = ap.parse_args()

    infos = {i.game_id.split("-")[0]: i
             for i in discover_environments(ROOT / "environment_files")}
    names = args.games or (CERO + REF)
    print(f"{'juego':6} {'niv':>4} {'cambio':>7} {'estados':>8}  clase")
    for name in names:
        if name not in infos:
            continue
        r = sonda(infos[name], args.budget)
        if not r:
            continue
        marca = "  <- CONTROL" if name in REF else ""
        print(f"{name:6} {r['niveles']:>4} {r['frac_cambio']:>7.1%} "
              f"{r['estados']:>8}  {r['clase']}{marca}", flush=True)
        print(f"       por accion (cambios/usos): {r['por_accion']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

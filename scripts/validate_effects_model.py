"""Valida el detector de efectos por PREDICCION FUERA DE MUESTRA.

POR QUE ASI Y NO POR CONSISTENCIA. La primera version del detector era 100%
"consistente" — y estaba mal: reportaba el MISMO desplazamiento para cuatro
acciones distintas. Sobre tableros densos (630-855 celdas no-fondo de 4096)
siempre existe algun offset que alinea muchas celdas por casualidad, asi que un
detector puede repetir el mismo error una y otra vez y parecer fiable.

Por eso aqui se mide lo unico que no se puede fingir: se ajusta la tabla con la
PRIMERA mitad del historial y se predice la SEGUNDA. Si el modelo de movimiento
es real, acierta transiciones que no vio.

Tres cifras por juego:
  - aciertos   : de las transiciones held-out cuya accion se clasifico "move",
                 cuantas se desplazaron como predecia la tabla
  - distintas  : cuantas direcciones DIFERENTES aprendio (4 botones que dan la
                 misma direccion es la firma del detector roto)
  - cobertura  : transiciones held-out con una prediccion disponible

Uso: python scripts/validate_effects_model.py [--steps 60] [--games ...]
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arc3.effects_model import effects_from_history, shift_between  # noqa: E402
from arc3.env import LocalGame, discover_environments  # noqa: E402

ACTIONS = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"]


class _F:
    def __init__(self, grid):
        self.grid = grid


class _E:
    def __init__(self, action, grid):
        self.action = action
        self.frame = _F(grid)


def grid_of(frame):
    if frame is None:
        return None
    f = getattr(frame, "frame", None)
    return None if f is None or len(f) == 0 else [[int(v) for v in r] for r in f[-1]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="*", default=None)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from arcengine import GameAction
    infos = {i.game_id.split("-")[0]: i
             for i in discover_environments(ROOT / "environment_files")}
    names = args.games or sorted(infos)

    tot_hit = tot_pred = 0
    by_conf: dict[float, list[int]] = {}
    dist_hist = []
    for name in names:
        if name not in infos:
            continue
        rng = random.Random(args.seed)
        game = LocalGame(infos[name])
        grid = grid_of(game.reset())
        if grid is None:
            continue
        entries = [_E(None, grid)]
        for _ in range(args.steps):
            a = rng.choice(ACTIONS)
            nxt = grid_of(game.step(getattr(GameAction, a)))
            if nxt is None:
                break
            entries.append(_E(a, nxt))
        if len(entries) < 12:
            print(f"  {name}: historial corto")
            continue

        half = len(entries) // 2
        # ventana=0 -> usa todo el tramo de ajuste, sin recorte de recencia
        table = effects_from_history(entries[:half + 1], window=0)
        moves = {a: tuple(d["shift"]) for a, d in table.items() if d["kind"] == "move"}
        distintas = len(set(moves.values()))
        dist_hist.append(distintas)

        hit = pred = 0
        for i in range(half + 1, len(entries)):
            a = entries[i].action
            if a not in moves:
                continue
            obs = shift_between(entries[i - 1].frame.grid, entries[i].frame.grid)
            if obs is None:
                continue
            pred += 1
            ok = obs == moves[a]
            hit += ok
            # ¿la confianza declarada predice el acierto? Si lo hace, un umbral
            # convierte las afirmaciones malas en incertidumbre honesta en vez de
            # meter hechos falsos en el prompt.
            by_conf.setdefault(round(table[a]["conf"], 1), [0, 0])
            by_conf[round(table[a]["conf"], 1)][0] += ok
            by_conf[round(table[a]["conf"], 1)][1] += 1
        tot_hit += hit
        tot_pred += pred
        acc = f"{hit}/{pred} = {hit/pred:.0%}" if pred else "sin held-out con movimiento"
        print(f"  {name}: {len(moves)} acciones 'move', {distintas} direcciones distintas "
              f"| held-out {acc}")

    print()
    if tot_pred:
        print(f"PREDICCION FUERA DE MUESTRA: {tot_hit}/{tot_pred} = {tot_hit/tot_pred:.1%}")
    else:
        print("PREDICCION FUERA DE MUESTRA: sin casos")
    if by_conf:
        print("\nprecision segun la CONFIANZA declarada (sirve de filtro?):")
        acum_h = acum_n = 0
        for conf in sorted(by_conf, reverse=True):
            h, n = by_conf[conf]
            acum_h += h
            acum_n += n
            print(f"  conf={conf:.1f}: {h:3}/{n:3} = {h/n:5.1%}   "
                  f"| acumulado conf>={conf:.1f}: {acum_h:3}/{acum_n:3} = {acum_h/acum_n:5.1%}")
    if dist_hist:
        degenerados = sum(1 for d in dist_hist if d == 1)
        print(f"juegos donde TODAS las acciones dieron la misma direccion "
              f"(firma del detector roto): {degenerados}/{len(dist_hist)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

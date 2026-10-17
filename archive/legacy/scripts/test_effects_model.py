"""Verifica que la carga del seam C tiene CONTENIDO en juegos reales.

Esta es la prueba que la v1 no paso. `plan_moves` se adopto (726 llamadas) pero
era vacia en los juegos sin movimiento, asi que el turno gastado no devolvia
nada. Antes de gastar un envio, medimos sobre los 25 juegos locales:

  - en cuantos la nota sale NO vacia (cobertura),
  - cuantas acciones quedan clasificadas como "sin efecto" (informacion util
    aunque el juego no sea de movimiento),
  - y si los desplazamientos detectados coinciden con los del detector del
    banco micro (control cruzado de signos: el bug de invertir los signos por
    seguir al fondo ya nos costo una tarde).

Uso: python scripts/test_effects_model.py [--games ls20 tu93 ...] [--steps 40]
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arc3.effects_model import effects_from_history, render_effects_note  # noqa: E402
from arc3.env import LocalGame, discover_environments  # noqa: E402

ACTIONS = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"]


class _Frame:
    def __init__(self, grid):
        self.grid = grid


class _Entry:
    """Imita HistoryEntry del harness: (action, frame.grid)."""
    def __init__(self, action, grid):
        self.action = action
        self.frame = _Frame(grid)


def grid_of(frame):
    if frame is None:
        return None
    f = getattr(frame, "frame", None)
    if f is None or len(f) == 0:
        return None
    return [[int(v) for v in row] for row in f[-1]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="*", default=None)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from arcengine import GameAction
    rng = random.Random(args.seed)
    infos = {i.game_id.split("-")[0]: i
             for i in discover_environments(ROOT / "environment_files")}
    names = args.games or sorted(infos)

    con_nota, sin_nota, total_inertes, total_moves = 0, [], 0, 0
    for name in names:
        if name not in infos:
            print(f"  {name}: no encontrado")
            continue
        game = LocalGame(infos[name])
        frame = game.reset()
        grid = grid_of(frame)
        if grid is None:
            print(f"  {name}: sin frame inicial")
            continue
        # historial con acciones aleatorias, como haria el agente al explorar
        entries = [_Entry(None, grid)]
        for _ in range(args.steps):
            a = rng.choice(ACTIONS)
            try:
                act = getattr(GameAction, a)
            except AttributeError:
                continue
            nxt = grid_of(game.step(act))
            if nxt is None:
                break
            entries.append(_Entry(a, nxt))

        table = effects_from_history(entries)
        note = render_effects_note(table)
        inertes = sum(1 for d in table.values() if d["kind"] == "sin efecto")
        moves = sum(1 for d in table.values() if d["kind"] == "move")
        total_inertes += inertes
        total_moves += moves
        if note:
            con_nota += 1
            head = note.splitlines()[1] if len(note.splitlines()) > 1 else ""
            print(f"  {name}: {len(table)} acciones | {moves} mueven, "
                  f"{inertes} inertes | ej: {head.strip()}")
        else:
            sin_nota.append(name)
            print(f"  {name}: NOTA VACIA  <-- aqui el turno se perderia")

    n = len([g for g in names if g in infos])
    print(f"\ncobertura: {con_nota}/{n} juegos con nota no vacia "
          f"({con_nota/n:.0%})" if n else "sin juegos")
    print(f"acciones que mueven: {total_moves} | acciones inertes detectadas: "
          f"{total_inertes} (cada una ahorra turnos)")
    if sin_nota:
        print(f"vacias en: {', '.join(sin_nota)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

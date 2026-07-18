"""Política de sondeo: juega cada environment y produce el dataset de features.

Estrategia por juego:
  1. RESET y features del frame inicial.
  2. Round-robin sobre las acciones simples disponibles (ACTION1..5, 7) para
     perfilar qué hace cada una (¿cambia el frame?, ¿mueve un objeto?, ¿sube nivel?).
  3. Sondeo de ACTION6 (click x,y) sobre una malla gruesa de puntos, para mapear
     regiones interactivas.
  4. Si el juego llega a GAME_OVER se hace RESET y se continúa hasta agotar budget.

Cada paso emite una fila con features de transición + features del frame resultante.
"""

from __future__ import annotations

import random
import time
from typing import Any, Optional

from arcengine import FrameDataRaw, GameAction, GameState

from .env import LocalGame
from .features import frame_to_grid, grid_features, transition_features

SIMPLE_ACTIONS = [
    GameAction.ACTION1,
    GameAction.ACTION2,
    GameAction.ACTION3,
    GameAction.ACTION4,
    GameAction.ACTION5,
    GameAction.ACTION7,
]


def _click_grid(n: int = 8) -> list[tuple[int, int]]:
    """Malla n x n de puntos (x, y) centrados en tiles de 64/n."""
    step = 64 // n
    half = step // 2
    return [(x * step + half, y * step + half) for y in range(n) for x in range(n)]


def probe_game(
    game: LocalGame,
    budget: int = 300,
    click_grid_n: int = 8,
    seed: int = 0,
    time_limit_s: Optional[float] = None,
) -> list[dict[str, Any]]:
    """Sondea un juego y devuelve filas de features (una por acción ejecutada)."""
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    t0 = time.time()

    frame = game.reset()
    if frame is None:
        return rows
    prev_grid = frame_to_grid(frame.frame)
    prev_levels = frame.levels_completed

    clicks = _click_grid(click_grid_n)
    rng.shuffle(clicks)
    click_i = 0
    step_i = 0

    while step_i < budget:
        if time_limit_s is not None and time.time() - t0 > time_limit_s:
            break
        avail = frame.available_actions or []
        simple = [a for a in SIMPLE_ACTIONS if not avail or a.value in avail]
        use_click = (GameAction.ACTION6.value in avail or not avail) and (
            not simple or step_i % 3 == 2
        )

        x = y = None
        if use_click and click_i < len(clicks):
            action = GameAction.ACTION6
            x, y = clicks[click_i]
            click_i += 1
        elif simple:
            action = simple[step_i % len(simple)]
        elif GameAction.ACTION6.value in avail:
            action = GameAction.ACTION6
            x, y = rng.randrange(64), rng.randrange(64)
        else:
            break

        nxt = game.step(action, x=x, y=y)
        step_i += 1
        if nxt is None:
            continue

        next_grid = frame_to_grid(nxt.frame)
        row: dict[str, Any] = {
            "game_id": game.info.game_id,
            "step": step_i,
            "action_id": action.value,
            "click_x": -1 if x is None else x,
            "click_y": -1 if y is None else y,
            "state": nxt.state.value,
            "levels_completed": nxt.levels_completed,
            "win_levels": nxt.win_levels,
            "level_up": nxt.levels_completed > prev_levels,
            "game_over": nxt.state == GameState.GAME_OVER,
            "win": nxt.state == GameState.WIN,
        }
        row.update(transition_features(prev_grid, next_grid))
        row.update({f"nf_{k}": v for k, v in grid_features(next_grid).items()})
        rows.append(row)

        prev_levels = nxt.levels_completed
        prev_grid = next_grid
        frame = nxt

        if nxt.state in (GameState.GAME_OVER, GameState.WIN):
            frame = game.reset() or frame
            prev_grid = frame_to_grid(frame.frame)
            prev_levels = frame.levels_completed

    return rows

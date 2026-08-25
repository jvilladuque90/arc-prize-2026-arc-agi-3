"""Captura TRAZAS GANADORAS: el GraphExplorer juega los 25 juegos locales y se
registra cada subida de nivel con su contexto.

PARA QUE. El banco de inferencia de META (candidato fuerte de STRATEGY §10)
necesita verdad verificable sobre "cual era el objetivo". La unica fuente
objetiva es una partida que COMPLETO el nivel: el estado justo antes de la
subida es, por definicion, el estado-meta. El GraphExplorer (nuestro stack de
julio, 24/25 niveles offline) es el musculo barato para conseguirla en CPU.

Por cada subida de nivel se guarda:
  - start_grid : tablero al empezar ese nivel
  - window     : las ultimas WINDOW transiciones (accion, tablero) antes de subir
                 (suficiente para identificar el objeto movil y su trayectoria)
  - pre_grid   : tablero inmediatamente anterior a la subida (el estado-meta)
  - trigger    : la accion que produjo la subida

Uso: python scripts/capture_traces.py [--budget 90] [--out traces_goal.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arc3.agent import GraphExplorer  # noqa: E402
from arc3.env import LocalGame, discover_environments  # noqa: E402
from arc3.features import frame_to_grid  # noqa: E402  (el explorador espera numpy)

WINDOW = 25


def grid_of(frame):
    if frame is None:
        return None
    f = getattr(frame, "frame", None)
    if f is None or len(f) == 0:
        return None
    return [[int(v) for v in row] for row in f[-1]]


def capture(info, budget_s: float, max_actions: int = 40000) -> dict:
    from arcengine import GameAction

    game = LocalGame(info)
    agent = GraphExplorer(info.game_id, max_actions=max_actions)
    frame = game.reset()
    grid = grid_of(frame)
    if grid is None:
        return {}
    gname = info.game_id.split("-")[0]
    levels = 0
    start_grid = grid
    window: list = []           # [(accion_str, grid_despues), ...] recortada a WINDOW
    events = []
    t0 = time.time()
    n = 0
    while frame is not None and time.time() - t0 < budget_s and n < max_actions:
        try:
            # el explorador trabaja sobre numpy; el registro guarda listas
            aid, x, y = agent.choose(
                frame_to_grid(frame.frame), frame.state.value,
                frame.levels_completed, list(frame.available_actions or []))
        except Exception:
            break
        n += 1
        try:
            if aid == 0:
                frame = game.reset()
                accion = "RESET"
            else:
                act = GameAction.from_id(aid)
                frame = game.step(act, x, y) if aid == 6 else game.step(act)
                accion = f"MOUSE({y},{x})" if aid == 6 else f"ACTION{aid}"
        except Exception:
            break
        nxt = grid_of(frame)
        if nxt is None:
            break
        if accion == "RESET":
            # un RESET reinicia el nivel: la ventana previa ya no describe este intento
            window = []
            start_grid = nxt
        else:
            window.append((accion, nxt))
            if len(window) > WINDOW:
                window.pop(0)
        lv = int(getattr(frame, "levels_completed", 0) or 0)
        if lv > levels:
            events.append({
                "level": lv,
                "start_grid": start_grid,
                "pre_grid": grid,          # el estado que disparo la subida
                "trigger": accion,
                "window": [(a, g) for a, g in window[-WINDOW:]],
                "action_index": n,
            })
            levels = lv
            start_grid = nxt
            window = []
        grid = nxt
        if getattr(frame, "state", None) is not None and frame.state.value == "WIN":
            break
    return {"game": gname, "levels": levels, "actions": n,
            "seconds": round(time.time() - t0, 1), "events": events}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=90.0)
    ap.add_argument("--games", nargs="*", default=None)
    ap.add_argument("--out", default="traces_goal.json")
    ap.add_argument("--max-actions", type=int, default=40000)
    args = ap.parse_args()

    infos = {i.game_id.split("-")[0]: i
             for i in discover_environments(ROOT / "environment_files")}
    names = args.games or sorted(infos)
    out = []
    for name in names:
        if name not in infos:
            continue
        r = capture(infos[name], args.budget, args.max_actions)
        if r:
            print(f"  {name}: {r['levels']} niveles, {r['actions']} acciones, "
                  f"{r['seconds']}s, {len(r['events'])} eventos", flush=True)
            out.append(r)
        else:
            print(f"  {name}: sin frame inicial", flush=True)
    total = sum(r["levels"] for r in out)
    (ROOT / args.out).write_text(json.dumps(out), encoding="utf-8")
    print(f"\n{total} niveles en {len(out)} juegos -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

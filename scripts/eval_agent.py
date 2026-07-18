"""Evalúa el GraphExplorer sobre los environments locales.

Uso:
  python scripts/eval_agent.py [--games ls20,ft09] [--max-actions 3000]
                               [--time-limit 120] [--out eval_out.csv]

Reporta niveles completados, acciones y tiempo por juego (la métrica de la competencia
es niveles completados sobre juegos ocultos).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
from arcengine import GameAction  # noqa: E402

from arc3.agent import GraphExplorer  # noqa: E402
from arc3.env import LocalGame, discover_environments  # noqa: E402
from arc3.features import frame_to_grid  # noqa: E402


def play_game(game: LocalGame, max_actions: int, time_limit_s: float) -> dict:
    agent = GraphExplorer(game.info.game_id, max_actions=max_actions)
    t0 = time.time()
    frame = game.reset()
    steps = 0
    best_levels = 0
    win = False
    while frame is not None and not agent.done and time.time() - t0 < time_limit_s:
        grid = frame_to_grid(frame.frame)
        aid, x, y = agent.choose(
            grid, frame.state.value, frame.levels_completed,
            list(frame.available_actions or []),
        )
        action = GameAction.from_id(aid)
        frame = game.step(action, x=x, y=y) if aid == 6 else (
            game.reset() if aid == 0 else game.step(action)
        )
        steps += 1
        if frame is not None:
            best_levels = max(best_levels, frame.levels_completed)
            if frame.state.value == "WIN":
                win = True
                break
    return {
        "game_id": game.info.game_id,
        "levels": best_levels,
        "win_levels": game.info.baseline_actions and None or None,  # se rellena fuera
        "win": win,
        "actions": steps,
        "seconds": round(time.time() - t0, 1),
        "nodes": len(agent._nodes),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="")
    ap.add_argument("--max-actions", type=int, default=3000)
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--out", default=str(ROOT / "features_out" / "eval_agent.csv"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    infos = discover_environments(ROOT / "environment_files")
    if args.games:
        prefixes = [g.strip() for g in args.games.split(",") if g.strip()]
        infos = [i for i in infos if any(i.game_id.startswith(p) for p in prefixes)]

    rows = []
    for info in infos:
        try:
            game = LocalGame(info, seed=args.seed)
            r = play_game(game, args.max_actions, args.time_limit)
        except Exception as e:
            print(f"  ERROR {info.game_id}: {e}")
            continue
        # win_levels desde el primer frame del juego
        r["win_levels"] = (game.env.observation_space.win_levels
                           if game.env.observation_space else 0)
        rows.append(r)
        print(f"  {r['game_id']}: {r['levels']}/{r['win_levels']} niveles, "
              f"{r['actions']} acciones, {r['seconds']}s, {r['nodes']} nodos"
              f"{' WIN' if r['win'] else ''}")

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    total = df["levels"].sum()
    print(f"\nTOTAL niveles: {total} en {len(df)} juegos "
          f"(media {total / max(len(df), 1):.2f}/juego) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

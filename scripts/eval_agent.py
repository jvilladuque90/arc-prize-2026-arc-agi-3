"""Evalúa el GraphExplorer sobre los environments locales con el runner paralelo.

Usa exactamente el mismo camino de código que el notebook de submission
(arc_agi.Arcade en modo OFFLINE + arc3.runner.run_games).

Uso:
  python scripts/eval_agent.py [--games ls20,ft09] [--budget 600] [--workers 4]
                               [--max-actions 15000] [--out features_out/eval_agent.csv]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ["ONLY_RESET_LEVELS"] = "true"

import pandas as pd  # noqa: E402
from arc_agi.base import Arcade, OperationMode  # noqa: E402

from arc3.runner import run_games  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="")
    ap.add_argument("--budget", type=float, default=600.0, help="presupuesto TOTAL en s")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-actions", type=int, default=15000)
    ap.add_argument("--max-game-s", type=float, default=120.0)
    ap.add_argument("--out", default=str(ROOT / "features_out" / "eval_agent.csv"))
    args = ap.parse_args()

    arcade = Arcade(operation_mode=OperationMode.OFFLINE,
                    environments_dir=str(ROOT / "environment_files"))
    game_ids = [e.game_id for e in arcade.available_environments]
    if args.games:
        prefixes = [g.strip() for g in args.games.split(",") if g.strip()]
        game_ids = [g for g in game_ids if any(g.startswith(p) for p in prefixes)]
    print(f"{len(game_ids)} juegos, budget total {args.budget}s, {args.workers} workers")

    results = run_games(arcade, game_ids, total_budget_s=args.budget,
                        workers=args.workers, max_actions=args.max_actions,
                        max_game_s=args.max_game_s)

    df = pd.DataFrame(results).sort_values("game_id")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))
    print(f"\nTOTAL niveles: {df.levels.sum()} en {len(df)} juegos "
          f"(media {df.levels.mean():.2f}) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

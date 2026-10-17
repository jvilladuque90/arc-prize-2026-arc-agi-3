"""Extrae el dataset de features de todos los environments locales.

Uso:
  python scripts/extract_features.py [--env-root environment_files] [--out features_out]
                                     [--budget 300] [--games ls20,ft09] [--seed 0]

Salidas en --out:
  transitions.parquet (o .csv si no hay pyarrow)  -> una fila por (juego, paso)
  action_summary.csv                              -> perfil de efecto por (juego, acción)
  games_summary.csv                               -> una fila por juego
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from arc3.env import LocalGame, discover_environments  # noqa: E402
from arc3.features import action_effect_summary  # noqa: E402
from arc3.probe import probe_game  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-root", default=str(ROOT / "environment_files"))
    ap.add_argument("--out", default=str(ROOT / "features_out"))
    ap.add_argument("--budget", type=int, default=300, help="acciones por juego")
    ap.add_argument("--games", default="", help="prefijos separados por coma (vacío = todos)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time-limit", type=float, default=120.0, help="segundos máx por juego")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    infos = discover_environments(Path(args.env_root))
    if args.games:
        prefixes = [g.strip() for g in args.games.split(",") if g.strip()]
        infos = [i for i in infos if any(i.game_id.startswith(p) for p in prefixes)]
    print(f"{len(infos)} environments a sondear")

    all_rows = []
    action_rows = []
    game_rows = []
    for info in infos:
        t0 = time.time()
        try:
            game = LocalGame(info, seed=args.seed)
            rows = probe_game(game, budget=args.budget, seed=args.seed,
                              time_limit_s=args.time_limit)
        except Exception as e:  # un juego roto no debe tumbar el resto
            print(f"  ERROR en {info.game_id}: {e}")
            continue
        all_rows.extend(rows)
        for s in action_effect_summary(rows):
            action_rows.append({"game_id": info.game_id, **s})
        game_rows.append(
            {
                "game_id": info.game_id,
                "tags": ",".join(info.tags or []),
                "n_steps": len(rows),
                "max_levels_completed": max((r["levels_completed"] for r in rows), default=0),
                "win_levels": rows[-1]["win_levels"] if rows else 0,
                "n_game_overs": sum(r["game_over"] for r in rows),
                "p_change": float(pd.Series([r["n_changed"] > 0 for r in rows]).mean()) if rows else 0.0,
                "seconds": round(time.time() - t0, 2),
            }
        )
        gr = game_rows[-1]
        print(f"  {info.game_id}: {gr['n_steps']} pasos, niveles {gr['max_levels_completed']}/"
              f"{gr['win_levels']}, p_change={gr['p_change']:.2f}, {gr['seconds']}s")

    df = pd.DataFrame(all_rows)
    try:
        df.to_parquet(out_dir / "transitions.parquet", index=False)
        print(f"transitions.parquet: {df.shape}")
    except Exception:
        df.to_csv(out_dir / "transitions.csv", index=False)
        print(f"transitions.csv: {df.shape}")
    pd.DataFrame(action_rows).to_csv(out_dir / "action_summary.csv", index=False)
    pd.DataFrame(game_rows).to_csv(out_dir / "games_summary.csv", index=False)
    print(f"Salidas en {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

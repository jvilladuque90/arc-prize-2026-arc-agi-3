"""Genera notebooks/features.ipynb embebiendo src/arc3 (fuente única de verdad).

El notebook resultante es autocontenido y 100% offline:
  1. localiza el root de la competencia bajo /kaggle/input
  2. instala arcengine + arc_agi desde arc_agi_3_wheels (--no-index)
  3. reconstruye src/arc3 desde el código embebido
  4. sondea todos los environments y guarda features en /kaggle/working

Uso:  python scripts/build_features_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "arc3"
OUT = ROOT / "notebooks" / "features.ipynb"

SETUP = '''\
# ARC-AGI-3 feature engineering — kernel offline (save-and-run, sin internet)
import os, subprocess, sys
from pathlib import Path

COMP_ROOT = None
for dirpath, dirnames, _ in os.walk("/kaggle/input"):
    if "environment_files" in dirnames:
        COMP_ROOT = Path(dirpath)
        break
assert COMP_ROOT is not None, "no se encontró environment_files bajo /kaggle/input"
print("COMP_ROOT =", COMP_ROOT)

wheels = COMP_ROOT / "arc_agi_3_wheels"
subprocess.run([sys.executable, "-m", "pip", "install", "--no-index",
                f"--find-links={wheels}", "arcengine", "arc-agi"], check=True)
import arcengine, arc_agi
print("arcengine OK")
'''

RUN = '''\
import sys, time
sys.path.insert(0, "/kaggle/working/src")
import pandas as pd
from pathlib import Path
from arc3.env import LocalGame, discover_environments
from arc3.features import action_effect_summary
from arc3.probe import probe_game

BUDGET = int(os.environ.get("ARC3_BUDGET", "500"))
TIME_LIMIT_S = float(os.environ.get("ARC3_TIME_LIMIT", "240"))
SEED = 0

out_dir = Path("/kaggle/working/features_out"); out_dir.mkdir(parents=True, exist_ok=True)
infos = discover_environments(COMP_ROOT / "environment_files")
print(len(infos), "environments")

all_rows, action_rows, game_rows = [], [], []
for info in infos:
    t0 = time.time()
    try:
        rows = probe_game(LocalGame(info, seed=SEED), budget=BUDGET, seed=SEED,
                          time_limit_s=TIME_LIMIT_S)
    except Exception as e:
        print("ERROR", info.game_id, e); continue
    all_rows.extend(rows)
    for s in action_effect_summary(rows):
        action_rows.append({"game_id": info.game_id, **s})
    game_rows.append({
        "game_id": info.game_id, "tags": ",".join(info.tags or []),
        "n_steps": len(rows),
        "max_levels_completed": max((r["levels_completed"] for r in rows), default=0),
        "win_levels": rows[-1]["win_levels"] if rows else 0,
        "n_game_overs": sum(r["game_over"] for r in rows),
        "seconds": round(time.time() - t0, 2)})
    print(game_rows[-1])

pd.DataFrame(all_rows).to_parquet(out_dir / "transitions.parquet", index=False)
pd.DataFrame(action_rows).to_csv(out_dir / "action_summary.csv", index=False)
pd.DataFrame(game_rows).to_csv(out_dir / "games_summary.csv", index=False)
print("filas:", len(all_rows))
'''

REPORT = '''\
summary = pd.read_csv("/kaggle/working/features_out/games_summary.csv")
actions = pd.read_csv("/kaggle/working/features_out/action_summary.csv")
print("=== juegos ==="); print(summary.to_string(index=False))
print()
print("=== acciones con efecto (p_change > 0.2) ===")
print(actions[actions.p_change > 0.2].to_string(index=False))
'''


def code_cell(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


def main() -> int:
    files = {f"src/arc3/{p.name}": p.read_text(encoding="utf-8")
             for p in sorted(SRC.glob("*.py"))}
    writer = ["# Reconstruye src/arc3 embebido (generado por build_features_notebook.py)\n",
              "import os\n", "os.makedirs('/kaggle/working/src/arc3', exist_ok=True)\n",
              f"SOURCES = {json.dumps(files, indent=1)}\n",
              "for path, code in SOURCES.items():\n",
              "    with open('/kaggle/working/' + path, 'w', encoding='utf-8') as f:\n",
              "        f.write(code)\n",
              "print('src/arc3 reconstruido:', list(SOURCES))\n"]
    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                    "name": "python3"},
                     "language_info": {"name": "python", "version": "3.12.0"}},
        "cells": [code_cell(SETUP), code_cell("".join(writer)),
                  code_cell(RUN), code_cell(REPORT)],
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"generado {OUT} con {len(files)} fuentes embebidas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

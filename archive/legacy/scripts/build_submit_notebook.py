"""Genera notebooks/submit.ipynb: la submission dual-mode (rerun gateway / offline).

Patrón confirmado en los notebooks top (ver docs/STRATEGY.md):
  - KAGGLE_IS_COMPETITION_RERUN=1 -> esperar http://gateway:8001/api/games y jugar los
    juegos ocultos via arc_agi Arcade en OperationMode.COMPETITION. El score sale de las
    partidas; el submission.parquet es un dummy.
  - Save & Run All (fase A) -> jugar los 25 environment_files offline (validación gratis).

El agente es GraphExplorer (src/arc3, embebido) — CPU puro, sin GPU: no gasta cuota G4.

Uso:  python scripts/build_submit_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "arc3"
OUT = ROOT / "notebooks" / "submit.ipynb"

SETUP = '''\
# ARC-AGI-3 submission - GraphExplorer (CPU, sin GPU)
import os, subprocess, sys, time
from pathlib import Path

NOTEBOOK_START = time.time()
TRUE_SUBMISSION = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true"}
os.environ["ONLY_RESET_LEVELS"] = "true"   # RESET reinicia el nivel, no el juego
print("TRUE_SUBMISSION =", TRUE_SUBMISSION)

COMP_ROOT = None
for dirpath, dirnames, _ in os.walk("/kaggle/input"):
    if "arc_agi_3_wheels" in dirnames:
        COMP_ROOT = Path(dirpath)
        break
assert COMP_ROOT is not None, "wheelhouse de la competencia no encontrado"
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--no-index",
                       "--no-warn-conflicts", "--disable-pip-version-check",
                       f"--find-links={COMP_ROOT / 'arc_agi_3_wheels'}", "arc-agi"],
                      stdout=subprocess.DEVNULL)
import arc_agi
print("arc_agi OK, COMP_ROOT =", COMP_ROOT)

# Kaggle exige un submission.parquet; el score real sale de las partidas contra el gateway.
import pandas as pd
pd.DataFrame([["1_0", "1", True, 1]],
             columns=["row_id", "game_id", "end_of_game", "score"]).to_parquet(
    "/kaggle/working/submission.parquet", index=False)
'''

RUN = '''\
sys.path.insert(0, "/kaggle/working/src")
from urllib.request import urlopen

from arc_agi.base import Arcade, OperationMode
from arc3.runner import run_games

# En rerun cada acción es un request HTTP al gateway (latencia-bound): la concurrencia
# multiplica el throughput (el milestone winner usaba 28). Offline es CPU-bound.
WORKERS = 14 if TRUE_SUBMISSION else 4
TOTAL_BUDGET_S = (8 * 3600 - 900) - (time.time() - NOTEBOOK_START) if TRUE_SUBMISSION else 1200
MAX_ACTIONS_PER_GAME = 15000
MAX_GAME_S = 2400.0 if TRUE_SUBMISSION else 90.0


def wait_for_gateway(base_url, timeout_s=600.0):
    deadline = time.monotonic() + timeout_s
    last = ""
    while time.monotonic() < deadline:
        try:
            with urlopen(f"{base_url}api/games", timeout=10) as r:
                if r.status < 500:
                    return
        except Exception as e:
            last = repr(e)
        time.sleep(5)
    raise RuntimeError(f"gateway no respondió: {last}")


if TRUE_SUBMISSION:
    base_url = os.environ.get("ARC_BASE_URL", "http://gateway:8001/")
    wait_for_gateway(base_url)
    arcade = Arcade(arc_api_key=os.environ.get("ARC_API_KEY", "test-key-123"),
                    arc_base_url=base_url,
                    operation_mode=OperationMode.COMPETITION,
                    environments_dir="")
else:
    arcade = Arcade(operation_mode=OperationMode.OFFLINE,
                    environments_dir=str(COMP_ROOT / "environment_files"))

game_ids = [e.game_id for e in arcade.available_environments]
print(len(game_ids), "juegos,", WORKERS, "workers, budget", int(TOTAL_BUDGET_S), "s")

try:
    card_id = arcade.open_scorecard(tags=["agent", "graph-explorer"])
except Exception as e:
    print("open_scorecard:", e)
    card_id = None

results = run_games(arcade, game_ids, total_budget_s=TOTAL_BUDGET_S, workers=WORKERS,
                    max_actions=MAX_ACTIONS_PER_GAME, max_game_s=MAX_GAME_S,
                    card_id=card_id)
'''

REPORT = '''\
try:
    sc = arcade.close_scorecard(card_id) if card_id else None
    if sc is not None:
        print(sc.model_dump_json(indent=2)[:4000])
except Exception as e:
    print("close_scorecard:", e)

import pandas as pd
df = pd.DataFrame(results).sort_values("game_id")
df.to_csv("/kaggle/working/results.csv", index=False)
print(df.to_string(index=False))
print("TOTAL:", df.levels.sum(), "niveles en", len(df), "juegos")
'''


def code_cell(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


def main() -> int:
    files = {f"src/arc3/{p.name}": p.read_text(encoding="utf-8")
             for p in sorted(SRC.glob("*.py"))}
    writer = ["# Reconstruye src/arc3 embebido (generado por build_submit_notebook.py)\n",
              "import os\n", "os.makedirs('/kaggle/working/src/arc3', exist_ok=True)\n",
              f"SOURCES = {json.dumps(files, indent=1)}\n",
              "for path, code in SOURCES.items():\n",
              "    with open('/kaggle/working/' + path, 'w', encoding='utf-8') as f:\n",
              "        f.write(code)\n",
              "print('src/arc3 reconstruido')\n"]
    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                    "name": "python3"},
                     "language_info": {"name": "python", "version": "3.12.0"}},
        "cells": [code_cell(SETUP), code_cell("".join(writer)),
                  code_cell(RUN), code_cell(REPORT)],
    }
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"generado {OUT} con {len(files)} fuentes embebidas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

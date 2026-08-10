"""Genera notebooks/explorer054.ipynb: réplica fiel de la submission pública 0.54.

Estrategia (ver docs/DESIGN.md §4.6 prioridad 1): nuestra brecha principal es el explorador
(0.25 nuestro vs 0.54 público). La réplica usa el pipeline EXACTO probado del notebook
público poby7722 (vendor/my_agent_v47.py, con atribución):
  - harness OFICIAL ARC-AGI-3-Agents (main.py --agent myagent): el Swarm corre TODOS los
    juegos CONCURRENTES (un thread por juego, 8h y 15000 acciones cada uno) — a diferencia
    de nuestro runner que repartía presupuesto (~40 min/juego). Esta es la brecha #1.
  - Explore2: grafo de estados, contador de agotamiento que se RESETEA al descubrir estados
    nuevos, sin deadsig, clicks por fill/(1+size) + rejilla, máscara de contador aprendida.
CPU puro (sin GPU): no gasta cuota G4.

Uso:  python scripts/build_explorer054_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT_SRC = ROOT / "vendor" / "my_agent_v47.py"
OUT = ROOT / "notebooks" / "explorer054.ipynb"

INSTALL = '''\
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-index", "--find-links",
    "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels",
    "arc-agi", "arcengine", "python-dotenv"])
print("deps OK")
'''

RUN = '''\
import os, shutil, subprocess
if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    subprocess.run(["curl", "--fail", "--retry", "999", "--retry-all-errors",
                    "--retry-delay", "5", "--retry-max-time", "600",
                    "http://gateway:8001/api/games"], check=True)
    shutil.copytree("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents",
                    "/kaggle/working/ARC-AGI-3-Agents", dirs_exist_ok=True)
    shutil.copy("/kaggle/working/my_agent.py",
                "/kaggle/working/ARC-AGI-3-Agents/agents/templates/my_agent.py")
    with open("/kaggle/working/ARC-AGI-3-Agents/agents/__init__.py", "w") as f:
        f.write("""from typing import Type
from dotenv import load_dotenv
from .agent import Agent, Playback
from .swarm import Swarm
from .templates.random_agent import Random
from .templates.my_agent import MyAgent
load_dotenv()
AVAILABLE_AGENTS: dict[str, Type[Agent]] = {"random": Random, "myagent": MyAgent}
""")
    with open("/kaggle/working/ARC-AGI-3-Agents/.env", "w") as f:
        f.write("""SCHEME=http
HOST=gateway
PORT=8001
ARC_API_KEY=test-key-123
ARC_BASE_URL=http://gateway:8001/
OPERATION_MODE=online
RECORDINGS_DIR=/kaggle/working/server_recording
""")
    env = dict(os.environ, MPLBACKEND="agg")
    subprocess.run(["python", "main.py", "--agent", "myagent"],
                   cwd="/kaggle/working/ARC-AGI-3-Agents", env=env, check=False)
else:
    print("Save & Run: instalacion validada; el agente juega solo en el rerun (igual que el original 0.54)")
'''

PARQUET = '''\
import os
if not os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    import pandas as pd
    pd.DataFrame(data=[["1_0", "1", True, 1]],
                 columns=["row_id", "game_id", "end_of_game", "score"]).to_parquet(
        "/kaggle/working/submission.parquet", index=False)
    print("submission.parquet dummy escrito")
'''


def code_cell(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}


def main() -> int:
    agent_code = AGENT_SRC.read_text(encoding="utf-8")
    writer = ("# Escribe el agente (vendor/my_agent_v47.py, ver atribución en el header)\n"
              f"AGENT_SOURCE = {json.dumps(agent_code)}\n"
              "with open('/kaggle/working/my_agent.py', 'w', encoding='utf-8') as f:\n"
              "    f.write(AGENT_SOURCE)\n"
              "print('my_agent.py escrito:', len(AGENT_SOURCE), 'chars')\n")
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"}},
          "cells": [code_cell(INSTALL), code_cell(writer),
                    code_cell(RUN), code_cell(PARQUET)]}
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"generado {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

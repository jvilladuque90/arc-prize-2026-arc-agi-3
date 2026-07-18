"""Descarga los datos de la competencia arc-prize-2026-arc-agi-3 con la Kaggle CLI.

Lee credenciales desde .env (KAGGLE_API_TOKEN, kaggle_username) y descomprime en el
root del repo (los datos vienen con la estructura ARC-AGI-3-Agents/, environment_files/,
arc_agi_3_wheels/). Nada de esto se versiona en git (ver .gitignore).

Uso:  python scripts/download_data.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMP = "arc-prize-2026-arc-agi-3"


def load_env(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        # .env como fuente de verdad: sobreescribe el entorno (evita tokens obsoletos).
        os.environ[key.strip()] = val.strip()


def main() -> int:
    load_env(ROOT / ".env")
    if "kaggle_username" in os.environ:
        os.environ["KAGGLE_USERNAME"] = os.environ["kaggle_username"]
    if "KAGGLE_API_TOKEN" not in os.environ and "KAGGLE_KEY" not in os.environ:
        print("ERROR: faltan credenciales Kaggle en .env", file=sys.stderr)
        return 1

    print(f"Descargando '{COMP}' en {ROOT} ...")
    subprocess.run(
        ["kaggle", "competitions", "download", "-c", COMP, "-p", str(ROOT)],
        check=True,
    )
    for zpath in ROOT.glob("*.zip"):
        print(f"Descomprimiendo {zpath.name} ...")
        with zipfile.ZipFile(zpath) as z:
            z.extractall(ROOT)
        zpath.unlink()
    print("Listo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

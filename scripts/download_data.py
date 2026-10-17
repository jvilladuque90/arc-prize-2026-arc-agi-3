"""Download the competition data (arc-prize-2026-arc-agi-3) with the Kaggle CLI.

Reads credentials from .env (KAGGLE_API_TOKEN, kaggle_username) and unzips into the repo root (the data ships as
ARC-AGI-3-Agents/, environment_files/, arc_agi_3_wheels/). None of it is versioned: see .gitignore.

Usage:  python scripts/download_data.py
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
        # .env is the source of truth: it overrides the environment (avoids stale tokens).
        os.environ[key.strip()] = val.strip()


def main() -> int:
    load_env(ROOT / ".env")
    if "kaggle_username" in os.environ:
        os.environ["KAGGLE_USERNAME"] = os.environ["kaggle_username"]
    if "KAGGLE_API_TOKEN" not in os.environ and "KAGGLE_KEY" not in os.environ:
        print("ERROR: Kaggle credentials are missing from .env", file=sys.stderr)
        return 1

    print(f"Downloading '{COMP}' into {ROOT} ...")
    subprocess.run(["kaggle", "competitions", "download", "-c", COMP, "-p", str(ROOT)], check=True)
    for zpath in ROOT.glob("*.zip"):
        print(f"Unzipping {zpath.name} ...")
        with zipfile.ZipFile(zpath) as z:
            z.extractall(ROOT)
        zpath.unlink()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

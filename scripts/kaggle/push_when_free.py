"""Push kernels one by one as Kaggle's batch GPU slots (2) free up.

Usage: python scripts/kaggle/push_when_free.py franzenc1 franzent1 [--private] [--max-minutes 25]

Each kernel is pushed with push_kernels.py --gpu; a push refused with "Maximum batch GPU session count" is retried every
two minutes. Prints what it did and exits 0 when every kernel was pushed, 2 when the time cap ran out first (relaunch it).
GPU launches are authorised by the user before this script is run (AGENTS.md rule 7).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PUSH = ROOT / "scripts" / "kaggle" / "push_kernels.py"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("kernels", nargs="+")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=25)
    a = ap.parse_args()
    pending = list(a.kernels)
    t0 = time.time()
    while pending and time.time() - t0 < a.max_minutes * 60:
        cmd = [sys.executable, str(PUSH), pending[0], "--gpu"] + (["--private"] if a.private else [])
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
        out = (r.stdout or "") + (r.stderr or "")
        if "successfully pushed" in out:
            print(f"{time.strftime('%H:%M:%S')} pushed {pending.pop(0)}: "
                  + next((l for l in out.splitlines() if "Kernel version" in l), ""), flush=True)
            continue
        if "Maximum batch GPU session count" in out:
            print(f"{time.strftime('%H:%M:%S')} {pending[0]}: no free GPU slot, retry in 2 min", flush=True)
            time.sleep(120)
            continue
        print(out[-2000:])
        return 1
    if pending:
        print(f"time cap reached; still pending: {pending}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

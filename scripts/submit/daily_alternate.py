"""Alternate the daily submission between kernels (decision of 2026-10-17).

Picks the kernel by the parity of the UTC date and delegates to submit_at_reset.py (idempotent per UTC day, retries
until the quota opens). Every firing on the same UTC day picks the SAME kernel, so the late retry firings never switch
candidate. Fallback: if the chosen kernel is not in COMPLETE state (e.g. its Save & Run failed), the next one in the
rotation is used.

Edit ROTATION to change the candidates. `--dry-run` shows the choice without submitting. Fired by the scheduled task armed with scripts/submit/arm_submit_task.ps1 -Alternate.
"""

from __future__ import annotations

import datetime
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / ".work" / "logs" / "daily_submit.log"

ROTATION = [
    ("juliancamilovilla/arc-agi3-franzen-m2",
     "franzen-m2 v1 (alternating): faithful, unedited copy of dfranzen/arc-agi-3-milestone-2-solution. "
     "Control: 6+ hidden samples, mean 25.7. Credit: Daniel Franzen, Tufa Labs. docs/RESULTS.md."),
    ("juliancamilovilla/arc-agi3-franzen-dprime",
     "franzen-dprime v1 (alternating): faithful copy of shiiin9/affectify-arc-31-54-in-a-single-sub, i.e. Franzen M2 with a "
     "single new cell, the D' slot priority of the scheduler (one public sample of 31.54, not conclusive). "
     "Credit: shiiin9 / Affectify (Osaka Metropolitan University), Daniel Franzen, Tufa Labs."),
]


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%dT%H:%M:%SZ}  [alternate] {msg}\n")


def is_complete(slug: str) -> bool:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        owner, name = slug.split("/")
        return "COMPLETE" in str(getattr(api.kernels_status(owner, name), "status", "")).upper()
    except Exception as exc:  # noqa: BLE001
        log(f"could not query the status of {slug} ({type(exc).__name__}); assuming it is fine")
        return True


def main() -> int:
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()
        if "kaggle_username" in os.environ:
            os.environ["KAGGLE_USERNAME"] = os.environ["kaggle_username"]
    today = datetime.datetime.now(datetime.timezone.utc).date()
    first = today.toordinal() % len(ROTATION)
    for step in range(len(ROTATION)):
        kernel, message = ROTATION[(first + step) % len(ROTATION)]
        if is_complete(kernel):
            log(f"{today}: it is the turn of {kernel}" + ("" if step == 0 else " (fallback)"))
            if "--dry-run" in sys.argv:
                print(f"[dry-run] would submit {kernel} (version in notebooks/kernel_versions.json)")
                return 0
            return subprocess.run([sys.executable, str(ROOT / "scripts" / "submit" / "submit_at_reset.py"), kernel, message],
                                  cwd=str(ROOT)).returncode
        log(f"{today}: {kernel} is not COMPLETE; trying the next one")
    log("no kernel in the rotation is COMPLETE; nothing submitted")
    return 1


if __name__ == "__main__":
    sys.exit(main())

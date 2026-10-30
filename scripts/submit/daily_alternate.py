"""Alternate the daily submission between kernels (decision of 2026-10-17; control / S1 from 2026-10-26; X1 alone from 2026-10-29).

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
    # Final week (ROADMAP item 5, from 2026-10-29): one stacked kernel every day. It is the only kernel that carries the
    # layout fix (RESULTS §3.1), so there is no fallback: a kernel without it can score 0.00 on an old-layout container.
    ("juliancamilovilla/arc-agi3-franzen-x2",
     "franzen-x2 v2 (final week, daily): copy of dfranzen/arc-agi-3-milestone-2-solution with infrastructure fixes "
     "(scored rerun recognised by ARC_BASE_URL; 55-min server wait; both Kaggle input layouts) plus three cells: lalalia9's "
     "solved-level memory (public 31.46, CC-BY 4.0, after rellik13's M85), 7 pre-seeded helper functions for the "
     "persistent library, and heliosli's stagnation reset (one fresh context after 96K tokens on an unfinished level). "
     "Gates: 43.5 (X2) / 40.1 (X1) / 50.7 (C1) vs control 42.2 (all within gate noise); X1 hidden 26.91, 25.64. "
     "Credit: Daniel Franzen, Tufa Labs, lalalia9, rellik13/sirikilohit, heliosli. docs/ROADMAP.md item 5."),
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
        return "COMPLETE" in str(getattr(api.kernels_status(slug), "status", "")).upper()   # signature: kernels_status(kernel)
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

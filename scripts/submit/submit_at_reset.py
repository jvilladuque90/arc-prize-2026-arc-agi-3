"""Wait for the daily UTC quota reset and submit a kernel to the hidden test set.

The competition allows ONE submission per UTC day. When the day's quota is spent the API answers 400 with
FAILED_PRECONDITION and the CLI only prints "400 Client Error", so the error body (which does state the reason) is
logged here. Success is verified against the submissions list, because the CLI prints nothing on success.

The script is idempotent per UTC day (safe to fire several times a night) and retries until the quota opens.
A submission that ended in ERROR/CANCELED does not count as the day's submission and is retried.

Usage:
  python scripts/submit/submit_at_reset.py juliancamilovilla/arc-agi3-franzen-m2 "message"
  python scripts/submit/submit_at_reset.py <kernel> "<message>" --now        # try once, do not wait
Environment: SUBMIT_MAX_MINUTES (default 240) bounds the retry window.
"""

from __future__ import annotations

import datetime
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMP = "arc-prize-2026-arc-agi-3"
LOG = ROOT / ".work" / "logs" / "daily_submit.log"


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"{stamp}  {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def load_env() -> None:
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()
    if "kaggle_username" in os.environ:
        os.environ["KAGGLE_USERNAME"] = os.environ["kaggle_username"]


def main() -> int:
    # First log line BEFORE any third-party import: on 2026-09-11 the scheduled task left no trace at all, which is
    # exactly what an ImportError for `kaggle` in an interpreter without the package would produce.
    log(f"start: {sys.executable} argv={sys.argv[1:3]}")
    args = [a for a in sys.argv[1:] if a != "--now"]
    wait = "--now" not in sys.argv
    kernel, message = args[0], args[1]
    version = json.loads((ROOT / "notebooks" / "kernel_versions.json").read_text(encoding="utf-8"))[kernel]

    load_env()
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as exc:  # noqa: BLE001
        log(f"ERROR importing kaggle in {sys.executable}: {type(exc).__name__}: {exc}")
        return 1
    api = KaggleApi()
    api.authenticate()

    # Idempotent: the task fires several times per night for robustness; a second firing after a successful
    # submission must not spend 240 minutes retrying.
    today = datetime.datetime.now(datetime.timezone.utc).date()
    try:
        for s in api.competition_submissions(COMP)[:5]:
            date = getattr(s, "date", None)
            status = str(getattr(s, "status", "")).upper()
            if date is not None and getattr(date, "date", lambda: None)() == today:
                # An ERROR/CANCELED submission (e.g. "A system error. Please try resubmitting", 2026-10-03) does not
                # count as the day's submission; without this exception nothing retried after it failed.
                if "ERROR" in status or "CANCEL" in status:
                    log(f"today's submission is {status} (ref={getattr(s, 'ref', '?')}); retrying")
                    continue
                log(f"a submission already exists today UTC (ref={getattr(s, 'ref', '?')}, {date}, {status}); "
                    f"not resubmitting")
                return 0
    except Exception as exc:  # noqa: BLE001 — if the check fails, try anyway
        log(f"warning: could not check today's submissions ({type(exc).__name__}); continuing")

    # The reset instant is NOT computed: we TRY and retry until the quota opens, which makes the submission immune to
    # the machine's time zone, server clock skew and the task starting early or late. The quota-exhausted 400 is
    # FAILED_PRECONDITION with the reason in the body (hidden by the CLI); any other error is logged and retried too,
    # because a network failure must not cost the day's submission.
    max_minutes = int(os.environ.get("SUBMIT_MAX_MINUTES", "240")) if wait else 1
    deadline = time.time() + max_minutes * 60
    log(f"armed: {kernel} v{version} — retrying for up to {max_minutes} min or until the quota opens")
    attempt = 0
    while True:
        attempt += 1
        try:
            api.competition_submit_code("submission.parquet", message, COMP, kernel, int(version))
            log(f"SUBMITTED {kernel} v{version} (attempt {attempt})")
            break
        except Exception as exc:
            resp = getattr(exc, "response", None)
            body = resp.text[:300] if resp is not None else str(exc)[:300]
            if attempt == 1 or attempt % 15 == 0:
                log(f"attempt {attempt}: {body}")
            if time.time() >= deadline:
                log(f"EXHAUSTED after {attempt} attempts: {body}")
                return 1
            time.sleep(60)

    # The CLI prints nothing on success -> verify against the list.
    time.sleep(15)
    subs = api.competition_submissions(COMP)
    if not subs:
        log("VERIFICATION: the submissions list came back empty")
        return 1
    newest = subs[0]
    log(f"newest submission: ref={getattr(newest, 'ref', '?')} date={getattr(newest, 'date', '?')} "
        f"status={getattr(newest, 'status', '?')}")
    desc = str(getattr(newest, "description", ""))
    ok = message[:40] in desc
    log("OK (verified in the list)" if ok else f"WARNING: unexpected description: {desc[:120]}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

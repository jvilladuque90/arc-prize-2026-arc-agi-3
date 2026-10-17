# Roadmap (updated 2026-10-17)

[Español](ROADMAP.es.md) · [Back to the README](../README.md)

Final submission deadline: **2026-11-02**. One submission per day.

## Running now

- **Daily alternation** (from 2026-10-18 UTC): the faithful Franzen copy (control) on one day, the D′ scheduler variant on the
  next, chosen by the parity of the UTC date by `scripts/daily_alternate.py`. If the chosen kernel is not in `COMPLETE` state, the
  other one is submitted instead. Armed as a Windows scheduled task with five triggers per day (the late ones retry after a
  platform error, which is only visible hours after the submission).
- Both kernels are public; the control is a verbatim copy and D′ is a verbatim copy of its author's notebook.

## Decision rule

The control has 6 hidden samples (mean 25.7, sd 1.6). D′ is kept only if its mean over **at least 4 samples** exceeds the control's
by more than ~3 points; otherwise the control wins. From **2026-10-29** only the better configuration is submitted, to exploit the
leaderboard's maximum.

## Next arms (one change each, in this order)

1. **Replay bench (V1)** — the tool that lets us judge serving and numeric changes without spending submissions.
2. **Stagnation reset (V5)** — a single cell; targets hard games stuck in loops.
3. **Pre-seeded helper functions (V4)** — saves the first minutes of every game without adding prose.
4. **Calibrated FP8 KV scales (V2)** — validated with the replay bench first.

Each arm needs: a unit test against the real patched harness, a local gate (breakage only), then hidden-set samples against the
control. Ideas that cannot be validated (lossy serving tricks, model swaps) are not shipped. Details and ranking:
[IDEAS](IDEAS.md).

## Rules of engagement

- One variable per arm; the control stays a faithful copy.
- Local gates screen out broken changes; they never rank improvements ([RESULTS](RESULTS.md), §3).
- GPU launches and submissions are announced before they happen.
- Credentials (`.env`) and competition data are never committed; every kernel is public.

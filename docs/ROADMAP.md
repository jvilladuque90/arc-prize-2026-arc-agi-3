# Roadmap (updated 2026-10-17)

[Español](ROADMAP.es.md) · [Back to the README](../README.md)

Final submission deadline: **2026-11-02**. One submission per day.

## Running now

- **Daily alternation** (from 2026-10-18 UTC): the faithful Franzen copy (control) on one day, the D′ scheduler variant on the
  next, chosen by the parity of the UTC date by `scripts/submit/daily_alternate.py`. If the chosen kernel is not in `COMPLETE` state, the
  other one is submitted instead. Armed as a Windows scheduled task with five triggers per day (the late ones retry after a
  platform error, which is only visible hours after the submission).
- Both kernels are public; the control is a verbatim copy and D′ is a verbatim copy of its author's notebook.

## Decision rule

The control has 8 hidden samples (mean 25.3, sd 1.5; the fixed control scored 24.00 on 27 Oct); D′ has 2 (27.30, 25.30). An exact 0.00 is an infrastructure failure and does not count (RESULTS §3.1). D′ is kept only if its mean over **at least 4 samples** exceeds the control's
by more than ~3 points; otherwise the control wins. From **2026-10-29** only the better configuration is submitted, to exploit the
leaderboard's maximum.

## Next arms (one change each, in this order)

0. **Infrastructure fixes** (done 26 Oct): every notebook recognises a scored rerun by `ARC_BASE_URL` and waits up to 55 min for the server (RESULTS §3.1); the corrected control re-gated at 42.15.
1. **Solved-level memory (S1 / V10)** — *gated 44.91 on 26 Oct* (`notebooks/franzen_s1.ipynb`, from `lalalia9`'s public 31.46 notebook); **now alternating daily with the control** (`scripts/submit/daily_alternate.py`), D′ paused at two samples. Moved ahead of M1/C1 because it is the only public addition with a reported gain.
2. **Pre-seeded helper functions (M1 / V4)** — *built and unit-tested* (`notebooks/franzen_m1.ipynb`); its gate failed twice on a dataset mount (not the code); third gate pushed 27 Oct (v3, private: Kaggle now refuses public kernels with the competition as a source until it ends).
3. **Stagnation reset (C1 / V5)** — *built and unit-tested* (`notebooks/franzen_c1.ipynb`, ported from `heliosli`); gate queued behind M1. Its effect can only appear in the hidden rerun.
4. **MTP depth 4 (T1 / V3)** — *built 27 Oct* (`notebooks/franzen_t1.ipynb`); judged on the gate's decode tok/s and accept length (repeatable to ~1 %), not on its score.
5. **Final week (from 29 Oct)**: with ~5 submissions left no arm can be confirmed on the hidden set (sd 1.5), so one *stacked* kernel — S1 plus whatever passed its gate without losing throughput — is submitted daily. The leaderboard keeps the maximum (control best 27.63), so this cannot lower the final score. Pre-registered rule, written here before 29 Oct.
6. **Replay bench (V1)** and **calibrated FP8 KV scales (V2)** — after the competition unless time appears.

Each arm needs: a unit test against the real patched harness, a local gate (breakage only), then hidden-set samples against the
control. Ideas that cannot be validated (lossy serving tricks, model swaps) are not shipped. Details and ranking:
[IDEAS](IDEAS.md).

## Where the diagrams are

The system diagram, the experiment-loop diagram and the proven / tried / untried component map are in
[APPROACH](APPROACH.md#diagram-the-system). `python scripts/check_repo.py --tests` checks that docs, registry, notebooks and tests agree.

## Rules of engagement

- One variable per arm; the control stays a faithful copy.
- Local gates screen out broken changes; they never rank improvements ([RESULTS](RESULTS.md), §3).
- GPU launches and submissions are announced before they happen.
- Credentials (`.env`) and competition data are never committed; every kernel is public.

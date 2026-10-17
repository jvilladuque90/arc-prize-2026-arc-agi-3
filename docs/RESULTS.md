# Results, experiments and corrections

[Español](RESULTS.es.md) · [Back to the README](../README.md)

Every number below is a **hidden-set public score** unless marked *local* (a 10- or 15-game demo run on Kaggle). Section numbers
(§) point to the original Spanish lab notebook in [archive/docs/DESIGN.md](../archive/docs/DESIGN.md).

## 1. Hidden-set history

| Period | Stack | Scores | Notes |
|---|---|---|---|
| Aug | State-graph explorer (no LLM) | 0.25 | the "floor" |
| early Sep | Duck harness + Qwen 27B | 1.15 → 1.59 | best of this era; switching the *base* is what moved it (§8.38–8.42) |
| 10–29 Sep | Duck harness + Qwen3.8-Flash-Next NVFP4 on vLLM | 3.55 · 3.54 · 2.69 · 3.95 · 3.03 · 2.80 · 2.77 · 3.98 · 3.09 · 2.96 · 2.66 | n = 11, mean 3.18, sd ≈ 0.5 |
| 30 Sep – 2 Oct | same stack + public "agentfix" patches | 3.78 · 2.49 · 4.79 | one run ended in a platform `ERROR` (not our code) |
| 3 Oct → | **Franzen Milestone-2 solution, unchanged** | 25.01 · 25.17 · 23.39 · 27.15 · 27.63 · 25.97 | n = 6, **mean 25.7, sd 1.6** |
| 9 Oct | + H1 (sandbox robustness fixes) | 24.17 | within noise |
| 10 Oct | + E1 (`reasoning_effort=medium`) | **20.98** | ~3 sd *below* the control |
| 18 Oct → | alternating control / D′ scheduler | pending | see [ROADMAP](ROADMAP.md) |

Leaderboard on 2026-10-17 (4,425 teams): 1st 68.9, 2nd 67.5, 3rd 50.4, 10th 41.7, 25th 36.9, 100th 33.2. Most of the 25th–100th
band are unmodified copies of the same public solution, lifted by taking the maximum over many daily submissions.

## 2. Experiments

Verdicts: **+** helped, **0** within noise, **−** hurt, **?** not decided.

### Changing the base system (the only thing that worked)

| Change | Result | Verdict |
|---|---|---|
| Qwen 27B → Qwen3.8-Flash-Next (NVFP4, vLLM) | 1.59 → 3.55 | **+** |
| vLLM/NVFP4 → Franzen's SGLang INT4 stack (128K context, fast serving, UNDO, persistent functions, priority gate) | 3–5 → **25.7** | **+** |

Why the second one worked (measured on the saved logs of four independent runs): decode 700–780 tok/s vs 195–270, a KV pool of
1.01 M tokens vs 0.1 M, and the three public winners of the milestone all share long retained history, FP8 KV cache, observable
game-over/undo, and persistent agent-written code.

### Tweaks on top of a base (≈25 tested)

| Idea | Evidence | Verdict |
|---|---|---|
| Text notes/memory written by the harness (7 variants: objects, preconditions, narratives, level inheritance, …) | hidden: all ≤ base | **0/−** |
| "Game manual" and consolidation notes at level-up | local: worse than control; hidden 2.66 vs base 3.2 | **−** |
| Action-batching reminder | local best (4.62), hidden worst (3.27) | **0** |
| Labeled grid image | local 10–4 by game, p = 0.18 | **0** |
| Thinking off | worst of the series | **−** |
| 11 hand-built "grafts" (efficiency, retry guard, schema helpers, …) | three arms, same negative direction | **−** |
| State-graph note in the prompt (v1 had a vocabulary bug; v2 fixed) | local 21 vs 19 levels at equal clock, p = 1.0 | **0** (§8.83–8.84) |
| Animation-aware solver on the old stack | local best, hidden 3.27 | **0** |
| H1 — keep the persistent-function library after a sandbox timeout; cap huge tool results | gate 46.63 vs 46.31; hidden 24.17 | **0** (safe fix) |
| D1 — deeper history at equal KV (8 slots, 24K drain) | gate 29.76 vs 46.31: −29 % actions, −17 % tok/s | **−** |
| E1 — `reasoning_effort=medium` (template default is `xhigh`) | 15 hard games ×3.1 (5.38 → 16.66, 11/3/1 games, p = 0.006); 10 easy games 46.3 → 38.0; **hidden 20.98** | **−** (local result did not transfer) |
| First-stint length of the scheduler | CPU simulation on the demo curves: curve is ~linear (~2.2 pts/min), no concavity to exploit | dropped |
| Slot occupancy (8.3–8.6 of 10 in use) | server is saturated: filling all slots gives +4–7 % tokens | **0** |
| D′ scheduler priority (public notebook, one sample 31.54) | other samples of the same code: 27.6–31.5 | **?** |

### Diagnostic: the hidden set is a mix (§8.87)

Franzen's demo plays 10 of the 25 public games. Running the unchanged solution on the **other 15** gave **5.38**, against 46.3 on
his 10 — a 25-game mix of **21.7**, close to the 25.7 seen on the hidden set. First level: **2.6 min** on the easy games and
**11.6 min** on the hard ones; tokens per action ≈ 600 vs ≈ 2,200. The gap is *speed of understanding a new game*.

## 3. Measurement

- Hidden-set run-to-run sd: **1.6** (n = 6 on the control; a community snapshot of copies reports 3.9). With one submission per
  day, 4-vs-4 samples resolve only ~4–8 points.
- A 10-game local demo has a per-run sd of the mean of **≈ 4** (individual games swing by tens of points between identical runs).
- Local gates therefore *screen out broken changes*; they do not rank improvements. The earlier 25-game bench was even
  anti-correlated with the hidden score (r = −0.35 over four arms, §8.81).
- The leaderboard keeps each team's maximum, so repeated submissions of an unchanged base are a (modest) lever: with mean 25.7 and
  sd 1.6–3.9 the expected maximum of ~28 draws is ~29–34.

Reproducibility note: the experiment notebooks in `notebooks/` were regenerated with English comments after the kernels H1, D1,
E1 and G15 had been pushed to Kaggle. The code that runs is identical; only comments and the embedded modules' docstrings differ.

## 4. Corrections to earlier claims

The archived lab notebook contains these mistakes, kept for transparency:

1. **Pooled vs paired statistics (§8.76 → §8.77).** A pooled proportions test called a 87.8 % → 81.4 % change significant
   (z = −2.72); paired by game it was 8–5, p = 0.58 — no effect.
2. **A two-variable arm (§8.76)** changed the image labels *and* lowered the resolution, so it could not isolate either.
3. **A pre-registered rule stated without checking the data (§8.74)** — it had already become unreachable.
4. **"The easy-game gate repeats to ±0.5" (§8.87) was wrong.** Three runs gave similar means (47.1 / 46.3 / 46.6), but game by
   game they differ by tens of points; the sd of a run's mean is ≈ 4.
5. **"Local gates predict the hidden set" (implicit in §8.89).** The medium-effort arm tripled the local hard-game score and
   scored below the control on the hidden set.
6. **A multi-agent audit (§8.85) was oversized**: ~7 M tokens, it hit a session limit before consolidating; the consolidation was
   done by hand from the reports that had reached disk.

## 5. What we would do differently

Start by reading the top public solutions for what they *share*; keep a faithful control and change one thing; budget for the
noise before running a test (how many submissions would this need?); never trust a local gate to rank; and write each
experiment's expected effect down before the result arrives.

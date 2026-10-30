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
| 18–20 Oct | alternating D′ / control / D′ | 27.30 · 24.30 · 25.30 | D′: n = 2, mean 26.3; control now n = 7, mean 25.5, sd 1.5 |
| 21–23 Oct | control / D′ / control | **0.00 · 0.00 · 0.00** | placeholder parquet: rerun not recognised, see §3.1 (not samples) |
| 27–28 Oct | alternating control (fixed) / S1 | 24.00 (control) · **26.20** (S1) | rerun fix confirmed on the scored branch; control now n = 8, mean 25.3, sd 1.5; S1 n = 1 |
| 29 Oct → | **X1** = control + S1 + M1, all fixes, daily | **26.91** · 30 Oct pending | final-week rule, [ROADMAP](ROADMAP.md) item 5; the 29 Oct score also proves the layout fix on the scored path |

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
| D′ scheduler priority (public notebook, one sample 31.54) | our hidden samples: 27.30, 25.30 | **?** (paused after two samples; replaced in the rotation by S1) |
| C1 — stagnation reset (V5) | gate 50.68 (highest of the week, within sd ≈ 4), 1,493 actions, decode 668 tok/s, accept 2.71 — identical serving to the control; the reset itself cannot fire in a 25-min demo | **0** on the gate (not broken); hidden effect unknown |
| T1 — MTP depth 4 (V3) | server never started: `Qwen QSA requires speculative_num_draft_tokens <= the QSA compress ratio (4); got 5` — a hard limit of the Pennyroyal/QSA kernels, so depth 3 is the maximum | **dead** (not tunable) |
| M1 — pre-seeded helper functions (V4) | gate 45.48 vs control 42.15 (paired by game 4–1–5, p = 0.38; the two games that moved most, r11l and tr87, moved the same way under S1, i.e. they are the noisy ones); patch applied, 1,405 actions, 866 K tokens, no errors | **0** on the gate (not broken; no ranking) |
| X1 — S1 + M1 stacked (final-week kernel) | gate 40.09 (control 42.15, S1 44.91, M1 45.48: all within the gate's sd ≈ 4); both patches active (memory blocks pinned for 1–7 levels, library seeded); 1,130 actions, 841 K tokens; ran on an **old-layout container** and finished, which validates the layout fix. One game (tr87) was given up by the scheduler at level 2 (52 of 58 actions) — the control also idled there in one run | **0** on the gate (not broken); hidden: 26.91 (29 Oct, n = 1), submitted daily from 29 Oct |
| S1 — solved-level memory (lalalia9, V10) | gate 44.91 vs control 42.15 the same day (sd of a gate mean ≈ 4: no ranking); memory block pinned for 2–4 levels per game, no errors | **?** hidden: 26.20 (n = 1, +0.9 over the control mean, inside 1 sd) |

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

### 3.1 Incident: three exact-zero submissions (2026-10-21 to 23) — cause found

The unchanged control and the D′ kernel both returned `COMPLETE` with a score of **0.00**, after 27.30 / 24.30 / 25.30 on the three
previous days. What we know: both produced a submission file of exactly **3,411 bytes** (normal runs: 3,800–3,900), so they failed in
the same way, most likely without playing at all; Franzen's datasets and models were last updated on 28–29 Sep, so the inputs did not
change; the leaderboard did not collapse (47 of the top 100 teams submitted on 22 Oct); Kaggle does not expose the rerun logs.
**Cause (found on 26 Oct, confirmed by another team's public notebook):** Franzen's notebook decides between "scored rerun" and
"interactive demo" from the environment variable `KAGGLE_IS_COMPETITION_RERUN`. Kaggle stopped injecting it around 21 Oct, so the
scoring rerun took the *demo* branch, played the bundled public games and wrote the 1-row placeholder parquet — the 3,411-byte
file. Kaggle still injects `ARC_BASE_URL` (the gateway) in every scoring rerun and never in an interactive run, so the fix is one
line: a rerun is recognised by either variable (`scripts/kaggle/build_arm.py rerun`, applied to the control and every arm).
Handling: an exact 0.00 is not a sample of the arm; the earlier hypotheses (slow server start, GPU session cap) were wrong; the
status check of the alternating submitter, which never worked (wrong API signature), is also fixed.

**Second infrastructure fix (26 Oct gate of the corrected control):** the server was not ready after Franzen's 12-minute
start-up limit (`DEADLINE at 722s … releasing the benchmark with the server still loading`; earlier gates were READY at
~8–9 min), so the ten demo games spent part of their 25 minutes on connection retries and the gate scored 22.3 instead of
~46. In a scored rerun each game has 532 minutes, so waiting for a ready server costs nothing: every notebook now waits up to
55 minutes (`build_arm.py startup`), as the public notebooks do. That gate is not a sample of anything.
With both fixes the control gate reached READY at 889 s and scored 42.15; the S1 gate reached READY at **1,711 s** (28.5 min,
well past the old limit) and scored 44.91.

**Third infrastructure fix (28 Oct):** six gates (M1 and C1 on 23–24 Oct and 28 Oct, T1 on 28 Oct, public and private, old and
freshly created kernels) died 5 s into the notebook because `/kaggle/input/datasets/dfranzen/taaf-kaggle-source-bundle-copy` was
not there to copy, while the identical metadata mounted on 26 Oct and on every script-kernel probe, and the dataset itself is
unchanged since 28 Sep. Two fresh-slug copies pushed five minutes apart on 28 Oct split the same way (M1-b mounted and ran, C1-b did not), so the
failure is per container, not per kernel. Two gates that waited 10 minutes and then listed `/kaggle/input` (T1 v2, C1-b v2) found the cause:
**some containers still use Kaggle's old flat layout** (`/kaggle/input/taaf-kaggle-source-bundle-copy`,
`/kaggle/input/pennyroyal-v253`, `/kaggle/input/arc-prize-2026-arc-agi-3`) while others use the new one
(`/kaggle/input/datasets/<user>/<slug>`, `/kaggle/input/competitions/<slug>`); models sit under `/kaggle/input/models` in both.
Franzen's notebook hard-codes the new layout. Every notebook now resolves each input path to whichever layout exists
(`build_arm.py layout`, `wheels`, `envfiles`); a scored rerun landing on an old-layout container would otherwise also fail, so
no kernel without this fix should be submitted again.

### 3.2 Audit of public baselines (2026-10-26)

Leaderboard: 71.48 / 68.91 / 56.02 at the top; Franzen himself 38.85. None of those versions is public. Of the public
kernels and ~30 GitHub repositories reviewed, nothing beats the Franzen control beyond noise: the most-voted "improved" notebook
is a byte-identical copy; the "31.54" kernels are D′ (two of our hidden samples of the same code: 27.3, 25.3); a "30.58" rewrite
reports its own effect as unmeasured; a GitHub repository claiming 35.35 holds only a README. The one mechanism with a reported
gain is `lalalia9`'s solved-level memory (31.46, one sample) — built here as S1 (IDEAS V10). Several of these notebooks also
raise the server start-up timeout to 45–55 minutes and one documents the rerun-detection failure of §3.1.

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

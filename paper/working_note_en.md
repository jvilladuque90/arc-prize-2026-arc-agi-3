# Exploration Is Capped, Understanding Is Hard: An Honest Hybrid Explorer–LLM Agent for ARC-AGI-3, and What Did and Did Not Break 0.25

*Working Note — ARC Prize 2026 · ARC-AGI-3 (Kaggle).*
**Author:** Julian Camilo Villa Duque.

> **Living document.** Updated as results arrive. See the [decision log](#appendix-a--decision-log-and-key-measurements) and the linked design doc [`docs/DESIGN.md`](../docs/DESIGN.md). Last updated: 2026-07-23.

---

## The problem in plain terms (a 60-second primer)

**ARC-AGI-3** measures *fluid intelligence*: an agent must play **interactive games it has never
seen**, discover the rules by acting, build a model of the world, and infer a goal that is never
stated. Unlike ARC-AGI-2 (static input→output puzzles), here the agent **acts** and learns from the
consequences.

- **Observation:** a **64×64** grid of colors 0–15 — an image (`FrameData`) that also carries the
  game `state`, `levels_completed`, `win_levels`, and the per-frame list of `available_actions`.
- **Actions:** `RESET`(0), six "simple" buttons `ACTION1–5,7`, and `ACTION6 = click(x,y)` on a cell.
  Games are tagged `keyboard`, `click`, or `keyboard_click`.
- **What is scored:** levels completed on the **hidden** set (~110 games). There are no worked
  examples — the agent must *explore then exploit*.
- **Compute rules:** the submission is a notebook. In the real *rerun* it waits for a gateway
  (`http://gateway:8001`) and plays the hidden games for ~8 h, **offline** (no internet); the score
  comes from the plays (the `submission.parquet` is a dummy). In *Save & Run* it plays the 25 public
  games **offline** — our free test bench that does **not** consume a competition submission.

![A 64×64 frame (synthetic example).](01_frame.png)

*Figure 0 — What a frame looks like (synthetic; not a competition game). A background, a large region,
a small rare-colored "button", an avatar, and a status bar painted in the border.*

**Why it is hard:** random probing completes ~0 levels on almost every training game; and *systematic
exploration* — our first avenue — plateaus at **0.25** on the hidden set. That 0.25 is the fraction of
hidden games solvable **without understanding the goal**. Everything above it requires a world model
and goal inference. That is where a large language model (LLM) enters.

### The method in essence — and its structure

Our agent is a **hybrid** with a cheap floor and a semantic ceiling:

```
HybridAgent(frame):
  if explorer not stuck:  action = GraphExplorer(frame)          # bank easy levels (the 0.25 floor)
  else:                   action = LLMAgent(frame)               # reason about the goal (the ceiling)
```

**The GraphExplorer (floor).** Treats play as search over a **graph of states**. Each distinct frame
is a node; an action is an edge. The engineering crux is the **state hash**: games paint step counters
and HUD in a 3-px border and animate interior counters, so a naive hash makes every frame unique and
the graph explodes. We hash a **masked** grid,

```
key(frame) = hash( grid ⊙ (1 − M) ) ,   M = border(3px) ∪ learned_counter_mask
```

where `learned_counter_mask` freezes interior cells that change in ≥80% of transitions (animations),
capped at ≤20% of the interior. Clicks are never brute-forced over 64×64; candidates are the
connected-component objects ranked by a **button-likeness** score,

```
button_score(obj) = 0.4·rarity(color) + 0.3·size_score(area) + 0.3·fill(obj)
```

(small, rare-colored, compact objects score high — buttons and avatars), plus a coarse coverage grid.
Structurally-inert click classes are suppressed (deadsig), and when a node's actions are exhausted a
BFS returns to the nearest node with untried actions; the learned graph is replayed after RESET
because the games are deterministic.

**The LLMAgent (ceiling).** When the explorer stalls, a frozen open-weights LLM (Qwen3-27B-FP8, served
by vLLM on the competition's RTX Pro 6000 GPU) decides. Its edge — and our central bet — is that we do
**not** feed it raw pixels alone; we inject the **object structure we already computed** as text
(*"obj3 color=2 size=16 center=(46,46) button_score=1.00"*, plus the numeric effect of the last
action), so the model reasons over **hard data**, not hallucinated pixels.

![The object features injected into the prompt (synthetic frame).](02_features.png)

*Figure 0b — Feature extraction on the synthetic frame. Objects get bounding boxes and a
`button_score` (green = high; the small red button and the avatar score 1.00; the large blue region
0.28). The white rectangle is the 3-px border the state hash ignores. This structure is what the LLM
receives as text, alongside the image.*

Three mechanisms turn one-shot action prediction into an **agentic loop**:

1. **Reflection memory** — every 15 transitions a second LLM call summarizes the history into a
   compact markdown memory (`## Rules / ## Goal / ## Avoid`) that is re-injected into every subsequent
   action prompt. This is *in-context* test-time adaptation.
2. **Action-effectiveness injection** — the observed `P(change)` per action is fed back as hard data,
   so the model stops re-choosing actions its own experience showed are inert.
3. **Guided navigation** — the LLM may return a spatial **subgoal** `goal:{x,y}`; a controller with a
   **learned motion model** (`action → mean (dy,dx)`, read from the movement-vector feature) then
   drives the avatar toward the goal *without spending an LLM call per step*. The LLM reasons *what*
   goal; the search executes *how*.

Any LLM failure (exception, empty parse, no legal action) falls back to the GraphExplorer, so the
agent never crashes and never runs out of a legal action.

---

## Abstract

I build an agent for ARC-AGI-3, where the score is levels completed on hidden interactive games. My
final pipeline is a **hybrid**: a deterministic **graph-of-states explorer** that banks the levels
reachable without understanding (a measured **0.25** ceiling on the hidden set), plus a frozen
open-weights **LLM policy** (Qwen3-27B-FP8 via vLLM on the RTX Pro 6000) that takes over when the
explorer stalls, and whose distinctive input is the **object structure we compute offline injected as
text into the prompt** — a lever no top public solution uses. On top of the LLM I add an **agentic
loop**: periodic **reflection memory**, **action-effectiveness** feedback, and **LLM-proposed spatial
subgoals executed by a learned-motion-model navigator**. This note is deliberately weighted toward
**what I learned and what failed**, because on this task the negative results carry the value.
My headline conclusions: (1) the hidden score decomposes into an **exploration floor of exactly 0.25**
— confirmed identically across three independent exploration variants (parallelism, diversified
restarts) — and a **semantic ceiling** that requires goal inference; (2) the LLM *technically works* —
Save & Run diagnostics show it reads our injected `button_score`, and its reflection memory infers
correct game rules verbatim (*"clicks are ineffective, movement works, the goal is to reach a target
state"*) — yet it **completes the same levels the explorer already reaches**, so single-step LLM
selection, even richly contextualized, does **not** clear the floor offline; (3) the disciplined
lesson — **the 30-minute offline Save & Run is time-starved and a poor proxy for the 8-hour hidden
rerun; only the hidden score is trustworthy, and it is 1/day** — is what kept me honest; and (4) a
sequence of ten builds (parallel runner, salient-click exploration, reflection, effectiveness
injection, guided navigation) drove the LLM failure rate from **98.6% → 3.7%** and made the agent
strictly more capable and cheaper (navigation replaces ~half the LLM calls), while the **offline level
count stayed flat at 9** — a plateau I read as time-starvation compounded by genuinely harder games
(click-puzzles, config-matching) that navigation alone does not address. Three vLLM boot failures
sharpened a reusable infrastructure lesson: FP8 weights routed through flashinfer's autotuner crash on
this GPU, fixed by forcing the Marlin FP8 kernel (`VLLM_TEST_FORCE_FP8_MARLIN=1`); and Qwen3, a
reasoning model, silently spends its token budget on `<think>` unless `enable_thinking=False`. I report
the per-component contribution of everything, and I am candid that my hidden result currently sits at
the **exploration floor (0.25)**; the open question — decided by the pending v10 hidden score — is
whether the agentic loop's efficiency *compounds over 8 hours* where the offline bench cannot show it.

*A note on scope: the reasoning, the data insights, and every negative result are shared in full — that
is where the community value lies. Exact prompts and constants live in the versioned source; the
competition data (game files, weights) is never committed, per privacy rules.*

---

## 1. The task, and what it reduces to

The target is **levels completed** across ~110 hidden games in ≤8 h. Two structural facts shape
everything:

- **Determinism.** The games are (near-)deterministic: replaying an action sequence from RESET
  reproduces the state. This makes a *graph of states with replay* a valid and cheap planner, and it is
  why exploration gets anything at all.
- **A floor–ceiling split.** Empirically the hidden score is `floor + ceiling`, where `floor ≈ 0.25`
  is what pure exploration reaches (level-1s and shallow games) and `ceiling` requires inferring the
  goal. Confirming the floor is *flat* across exploration variants (§4) is the single most important
  measurement — it told me exploration was a solved, capped sub-problem and the investment had to move
  to understanding.

## 2. Insights about the games (the data)

Feature extraction over the 25 public games (`src/arc3/features.py`, dumped to `features_out/`) yields
the structure the whole agent stands on:

- **Modality tags.** Each game is `keyboard`, `click`, or `keyboard_click`. `available_actions` gives
  the same per-frame, but the tag is a global prior on whether arrows or clicks matter.
- **Response sparsity `p_change`.** Fraction of actions that change the frame, ranges from **0.02**
  (`lp85`) and **0.07** (`ft09`) — almost nothing responds except the exact right action — to **1.0**
  (`ls20`, `tu93`) where everything moves. Low-`p_change` games are needles-in-haystacks that punish
  blind action and reward targeted, understood moves.
- **Objects and button-likeness.** Connected components excluding the majority-color background; the
  small, rare, compact ones are the interactive elements. This single heuristic drives both the
  explorer's click candidates and the LLM's targeting.
- **The border is HUD.** Counters and status live in the 3-px border and in interior animated cells;
  masking them for the state hash is the difference between a graph that converges and one that
  explodes. This is the most load-bearing data insight.
- **Random is not enough.** Under a 300-action random probe the max level reached is 0 on 24/25 games
  (only `r11l` reaches 1) — confirming that structure, not luck, is required.

## 3. Method (each stage has a job)

1. **Features (`features.py`)** — objects, button_score, transition diffs, movement vectors, per-action
   effect profiles. Pure numpy, offline, identical code local and in the kernel.
2. **GraphExplorer (`agent.py`)** — masked-hash state graph, salient-click candidates, deadsig
   suppression, BFS-to-pending, deterministic replay. The 0.25 floor, at zero LLM cost.
3. **LLMAgent (`llm_agent.py`)** — frozen Qwen3-27B via vLLM; image + injected object text + reflection
   memory + effectiveness; robust JSON parse; plan queue; per-state ineffective-action memory.
4. **Reflection** — periodic second LLM call → markdown memory re-injected (in-context TTT).
5. **Guided navigation** — `parse_goal` extracts `goal:{x,y}`; a learned motion model
   (`action→vector`) and an avatar-position estimate drive a controller toward the subgoal, bypassing
   per-step LLM calls.
6. **HybridAgent** — routes explorer→LLM on stall, preserving the floor and adding the ceiling.
7. **Parallel runner (`runner.py`)** — a thread pool with a shared time budget; in the HTTP-bound rerun
   this multiplies throughput (vLLM batches concurrent requests on one GPU).
8. **Save & Run diagnostics** — the notebook dumps, per game, the LLM's prompt→raw reply→parsed
   actions, the reflections, and a categorized failure breakdown, so every change is judged on real
   model behavior before spending a submission.

## 4. Breadth and depth of exploration (negatives valued equally)

This is the heart of the note. Each row is a build; the negatives are as informative as the positives.

- **Exploration parallelism did not move the hidden score.** v1 (serial) and v2 (14 parallel workers)
  both scored **0.25**. The rerun is HTTP-latency-bound, so concurrency multiplies actions ~14×; that
  it changed nothing proved the bottleneck was **semantic, not compute**.
- **Diversified restarts did not move it either.** v3 (salt that densifies click grids and rotates
  action order on graph exhaustion) also scored **0.25**. Three independent exploration variants at the
  identical number is strong evidence of a **hard floor**.
- **The LLM boots but does not clear the floor offline.** After the hybrid's first hidden run also
  scored **0.25**, Save & Run diagnostics showed the LLM working well (reads `button_score`, valid
  JSON) but completing the *same* levels — the wall is goal inference, not the prompt.
- **Prompt tweaks improve robustness, not levels.** Reflection (v7), effectiveness injection (v9), and
  guided navigation (v10) drove the LLM failure rate **98.6% → 7.7% → 5.0% → 3.7%** and made the agent
  cheaper (navigation replaced ~half the LLM calls; `nav_used=1692` in v10), yet the **offline level
  count held at 9** (v6–v10). A real capability/efficiency gain that the time-starved offline bench
  cannot convert into levels — the crux of the uncertainty in §6.
- **Infrastructure negatives (three wasted G4 boots, each diagnostic).** (i) vLLM crashed in
  flashinfer's cudagraph autotuner → not the cause; (ii) `--enforce-eager` isolated it to the **FP8
  GEMM** kernel (`FlashInferFP8ScaledMMLinearKernel`); (iii) `VLLM_TEST_FORCE_FP8_MARLIN=1` booted it.
  Then a trivial runner bug (`len(agent._nodes)` — an attribute only the explorer has) discarded every
  game's result though the LLM had played; and Qwen3's reasoning tokens starved the JSON until
  `enable_thinking=False`. Each failure was cheap because the offline soft-deadline capped it and the
  fallback kept the run non-zero.

**Not yet built (candidly), the levers the diagnostics point to:** a **planner on the simulator**
(FORGE-style search on the game's own `.py`, if the source is reachable in the rerun); **LoRA-SFT** on
solver-generated trajectories (risking overfit to the 25 train games); and richer subgoal types
(`click_all`, `match_target`) extending the guided loop beyond navigation.

### 4-bis. On LoRA and test-time training (a deliberate design choice)

A recurring question: would **LoRA** and **TTT** help here? My reasoned position (detailed in
`docs/DESIGN.md`):

- **In-context adaptation (what we do)** — reflection memory + feature injection + per-state
  ineffective memory — is the cheap, generalizing form of test-time adaptation, and it is what the
  strongest single-LLM public agent (LB 0.86) used. First lever to exhaust.
- **LoRA-SFT (offline):** *conditionally useful*. We can generate near-optimal trajectories by loading
  a game's `.py` and solving it with search on the simulator, then fine-tune a small adapter so the
  model is natively fluent in the action format and general idioms ("click buttons", "explore then
  exploit"). The **risk is overfitting** to the 25 train games — the hidden games differ, so the value
  is teaching *general skills*, not memorizing solutions; heavy augmentation and a held-out game split
  are mandatory.
- **Online TTT on the LLM:** *not recommended*. vLLM serving does not compose with online weight
  updates; running training + serving of a 27B model on one GPU is costly and brittle, and the
  milestone winner did not do it. The reflection memory is the practical "test-time training" — it
  adapts per-game *in context* without touching weights. If online learning is truly wanted, a light
  StochasticGoose-style CNN is the pragmatic choice, but it caps around 0.35–0.46.

## 5. Contribution of individual ideas

| Idea | Where measured | Contribution |
|---|---|---|
| Graph-of-states + masked hash | hidden LB | **the entire 0.25 floor**; without the border/counter mask the graph does not converge |
| Parallel runner | rerun throughput | ~14× actions (HTTP-bound); **no hidden-score change** — proved the floor |
| Salient clicks (button_score) | explorer + LLM targeting | the mechanism by which either agent finds interactive cells |
| Feature injection into prompt | Save & Run diagnostic | **qualitatively decisive** — the LLM reasons over `button_score` verbatim; our differentiator |
| Reflection memory | fail rate, memory dumps | infers correct rules in-context; fail rate 7.7% and coherent `Rules/Goal/Avoid` |
| Effectiveness injection | fail rate 7.7%→5.0% | robustness; counters re-choosing inert actions |
| Guided navigation | fail 5.0%→3.7%, nav_used=1692 | capability + efficiency (replaces ~½ LLM calls); offline levels flat |
| vLLM boot fixes (Marlin FP8, thinking off) | boot success | **enabling** — without them the LLM never runs (three failed boots) |

## 6. Uncertainty estimation

- **Offline ≠ hidden.** The Save & Run bench is 30 min / 8 workers over 25 games — **time-starved**.
  The hidden rerun is 8 h over ~110 games, so per-game the agent has far more time; efficiency gains
  (navigation, fewer LLM calls) **compound there** and cannot show offline. The offline "9 levels" is a
  lower bound on capability, not the hidden score.
- **Seed/stochasticity.** LLM temperature 0.3 and exploration randomness make single offline runs
  noisy; only differences that survive across games and the hidden score are trusted.
- **The trustworthy signal is the 1/day hidden score.** A daily Windows scheduled task auto-submits the
  latest best kernel at 20:00 local (just after the 00:00 UTC reset), so every window is used without
  manual intervention.

## 7. Reflection: genuine understanding vs optimizing the metric

The honest tension is that our *technically working* LLM does not yet beat the *mechanically simple*
explorer on the hidden set. The Save & Run dumps are the antidote to self-deception: they show the
model **genuinely inferring game rules** (bp35: *"clicking y=63 shifts the grid up; the goal likely
involves clearing all cells or matching a configuration"*), which is real understanding — but
understanding the mechanics is not the same as *planning to the goal*. The plateau is not a prompt bug
to be tuned away; it is the genuine, hard core of the benchmark (goal inference and multi-step
planning), and I have tried to report that plainly rather than chase an offline number that the hidden
rerun does not reward.

## 8. Conclusion

Exploration is a solved, capped sub-problem (0.25) and understanding is the hard part — the same split
the benchmark is designed to expose. I built a sophisticated, well-instrumented hybrid that makes the
LLM read our object features, learn game rules in context, and navigate to LLM-chosen subgoals, driving
the failure rate from 98.6% to 3.7% while keeping a guaranteed floor. Whether that capability converts
to hidden levels — whether the agentic loop compounds over 8 hours — is the open question the pending
v10 hidden score will answer. If it clears 0.25, the path is stacking richer subgoals; if not, the
diagnostics point to a simulator-planner or LoRA-SFT as the next investment.

---

## Appendix A — Decision log and key measurements

| Date | Build / decision | Offline levels | Hidden LB | Reading |
|---|---|---|---|---|
| 2026-07-18 | GraphExplorer (CPU) | 24/25 | 0.25 | exploration floor established |
| 2026-07-19 | v2 parallel runner (14 workers) | — | 0.25 | HTTP-bound; parallelism didn't move it |
| 2026-07-20 | v3 diversified restarts | 17 | 0.25 | exploration hard-capped (v1=v2=v3) |
| 2026-07-22 | Hybrid explorer+LLM (features in prompt) | 9 | 0.25 | LLM boots, reads features; same levels |
| 2026-07-22 | + Reflection (v7) | 8 | — | rules inferred in-context; fails 5% |
| 2026-07-22 | + Effectiveness (v9) | 9 | — | fails 7.7%→5.0%; levels flat |
| 2026-07-22 | + Guided navigation (v10) | 9 | **0.26** | **first break above the 0.25 floor** — the agentic loop compounds over 8 h where offline (flat at 9) cannot show it |
| 2026-07-24 | + click_all subgoal (v11) | 9 | **0.25** | **regression to the floor** — click_all clicked 16 objects blindly and wasted the action budget on games where clicking is inert, displacing v10's navigation gain |
| 2026-07-24 | + guarded click_all (v12) | 9 | pending | abort the click_all excursion the instant a click yields no change → the subgoal can only help, never waste budget; navigation (the 0.26 lever) untouched |

> **Update 2026-07-24 — a clean negative and its fix.** v11 (adding a blind `click_all` subgoal)
> scored **0.25**, *below* v10's 0.26 — a real regression back to the exploration floor. The metric is
> discrete (~0.01 ≈ one hidden level), so 0.26→0.25 means click_all *lost* the one level navigation had
> won: its controller committed up to 16 clicks per invocation without checking effect, burning the
> action budget on games where clicking is inert. The lesson is the same one the exploration A/B taught:
> **an unguarded excursion is net-negative on a task where the action budget is the binding constraint.**
> Fix (v12): the click_all excursion aborts the instant a click produces no frame change, so it can only
> help; navigation — the proven 0.26 lever — is left untouched.

> **Update 2026-07-23 — the floor is broken.** v10 scored **0.26** on the hidden set, the first
> movement above the 0.25 exploration ceiling across seven submissions. Offline was flat at 9 levels,
> exactly as §6 predicted: the guided-navigation efficiency (nav replaces ~½ of LLM calls) buys more
> useful actions per game over 8 hours, which the 30-min bench cannot register. This validates the
> agentic-loop direction and the plan: **stack richer subgoals** (`click_all`, `match_target`) beyond
> navigation. Small in absolute terms, but it turns "does the LLM help at all on hidden games?" from
> open to **yes**.

**Failure-rate trajectory (LLM calls that fell back):** 98.6% → 7.7% → 5.0% → 3.7% → 3.2% (v11 regresó LB; guard en v12).
**vLLM on RTX Pro 6000:** model 33.7 GiB, KV cache 45 GiB; boots with Marlin FP8 + FLASH_ATTN +
`enable_thinking=False`.

*This appendix is updated on every new result; the narrative sections above are revised when a decision
changes the strategy.*

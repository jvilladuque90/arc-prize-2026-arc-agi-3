# What moves an ARC-AGI-3 agent: a base system, measured honestly

*ARC Prize 2026 Paper Track (ARC-AGI-3). A faithful reproduction of the strongest public agent, about 25 controlled changes on top of it, and where the metric is actually lost.*

[Español](WRITEUP.es.md) · [Back to the README](../README.md)

**Team:** Julian Camilo Villa · **Competition:** ARC Prize 2026, ARC-AGI-3 (Kaggle) · **Final stack:** hidden-set mean 25.2 (n = 5); best submission 27.63 (control)
**Submission IDs:** 56893893 (best, 27.63, control kernel [arc-agi3-franzen-m2](https://www.kaggle.com/code/juliancamilovilla/arc-agi3-franzen-m2)); 57924975 (final stack X2, 27.30, [arc-agi3-franzen-x2](https://www.kaggle.com/code/juliancamilovilla/arc-agi3-franzen-x2))  
**Code:** [github.com/jvilladuque90/arc-prize-2026-arc-agi-3](https://github.com/jvilladuque90/arc-prize-2026-arc-agi-3) (MIT); every kernel is a one-cell variant of `notebooks/franzen_m2.ipynb`

## Summary

We spent three months on ARC-AGI-3, the interactive version of the benchmark, where an agent must discover the rules of unseen
games by acting. Our own pipeline peaked at a mean of 3.2. Adopting Daniel Franzen's public Milestone-2 solution as a faithful
control took us to 25.3. We then tested about 25 changes on top of a base, ten of them on the hidden set. None moved the score
outside the noise, and one lowered it by three standard deviations.

The useful results are about measurement rather than a new trick:

1. **The base system is the lever.** Serving speed, context length and memory set the ceiling; prompt-level additions did not.
2. **Local gates cannot rank changes.** A 10-game gate has a standard deviation of about 4 per run. Its ranking never matched the hidden set.
3. **The metric is lost on unsolved levels, not on extra actions.** 55 % of the level-weighted score is lost on levels the agent never
   solves. Only 2 % is lost on solved levels that took too many actions.
4. **The server is bound by KV memory, not compute.** More parallel games decode faster at peak but generate 21 % fewer tokens.

![Score by base system](figures/cover.png)

## 1. Approach

### 1.1 Base system

The control is an unmodified copy of Franzen's Milestone-2 notebook, which builds on Tufa Labs' harness:

- **Model:** Qwen3.8-Flash-Next with Intel's AutoRound INT4 weights and Albucino's MTP draft model for speculative decoding.
- **Serving:** Pennyroyal, an SGLang fork for the RTX PRO 6000, with a 128K context, an FP8 KV cache of 1.01 M tokens, and 10 games
  decoding in parallel.
- **Agent:** the model reasons with its full thinking kept in history. It acts through a Python tool. It sees the board as an upscaled
  image plus animation frames, and diagnoses each game over from a diff image. It can undo, and it keeps the helper functions it writes
  for the rest of the game.

The three public milestone winners share long retained history, an FP8 KV cache, observable game over and undo, and persistent
agent-written code. We took that shared core as the strongest prior available.

![Score by base system](figures/fig1_history.png)

### 1.2 Our additions

Each addition is one notebook cell that patches the harness by exact-anchor replacement. Each one has a unit test against the real
harness file.

| Arm | What it changes | Source |
|---|---|---|
| **S1** solved-level memory | When a level is cleared, its facts and a one-line rule written by the model are pinned at the end of the system prompt | `lalalia9` (CC-BY 4.0), after `rellik13` |
| **M1** seeded helpers | Each game starts with 7 tested utilities, chosen from what the model rewrote most often in 70 saved logs | ours |
| **C1** stagnation reset | After 96K tokens on one unfinished level, clear the reasoning context once | `heliosli` |
| **H1** sandbox fixes | Keep the function library after a sandbox timeout; cap huge tool outputs | ours |
| **X1, X2** | X1 = S1 + M1; X2 = X1 + C1 (the final submission) | — |

We also tried serving changes: deeper MTP drafting (T1), 12 parallel games (T2), lower reasoning effort (E1) and deeper history (D1).

### 1.3 Protocol

- **One variable per arm, against a faithful control.** Each arm differs from the control in one cell.
- **A local gate only screens out breakage.** The gate is a 25-minute run on 10 public games. It also measures decode tokens per second
  and the speculative acceptance length, which repeat to about 1 % and so can judge serving changes.
- **Ranking comes only from the hidden set.** We submitted once per day, compared by game with sign tests, and wrote each decision rule
  down before its result arrived.

## 2. Results

### 2.1 Hidden set

![Hidden-set samples by arm](figures/fig2_arms.png)

The control's eight samples span 23.4 to 27.6, with a mean of 25.3 and a standard deviation of 1.6. At one submission per day, telling
two arms apart by 2 points needs about 8 samples each. No arm got close to that many.

- **E1 hurt.** Lowering the reasoning effort tripled the local score on hard games, then scored 20.98 on the hidden set.
- **Everything else stayed in the band.** S1, D′ and X1 sat slightly above the control mean. X2, our final stack, ended at 25.2 over five
  samples.

### 2.2 Why local gates cannot rank

![Gate score versus hidden score](figures/fig3_gates.png)

Identical gate runs differ by tens of points on single games. The best gate of the final month, C1 at 50.7, would have been a false
lead. An older local bench was anti-correlated with the hidden set across four arms (r = −0.35).

### 2.3 Where the score is lost

![Metric decomposition](figures/fig4_metric_loss.png)

The metric squares the ratio of baseline actions to actions taken, per level, and weights later levels more. We split the 50 game runs
of our gates by level. Solved levels already score close to full: 40 % of the weight comes from levels solved in no more actions than
the baseline. The loss is on levels never solved, usually the later ones that weigh most. On hard public games each action costs
about 2,200 reasoning tokens against 600 on easy ones. The gap is how fast the model understands a new game.

### 2.4 Throughput is capped by memory

![Throughput at 10 and 12 streams](figures/fig5_concurrency.png)

If unsolved levels are the loss, more reasoning tokens per game should help, so we pushed the server.

- **Time:** the 9-hour budget is already used in full, and loading 185 GB of model files at Kaggle's input bandwidth costs 15 to 28 minutes.
- **Deeper MTP drafting:** impossible. The attention kernels cap draft tokens at 4.
- **12 parallel games:** peak decode rose from 687 to 759 tokens per second. The KV pool then saturated, games were evicted and
  re-prefilled, and total output fell from 842 K to 662 K tokens.

Franzen's configuration of 10 parallel games sits at the optimum.

## 3. Why it works, and why the tweaks did not

An ARC-AGI-3 game is solved by forming a hypothesis about hidden rules, testing it with a few actions, and revising it. In this
system each revision is paid in reasoning tokens. The score is therefore roughly the number of hypothesis cycles a game receives,
times the chance that a cycle is right. The base system raises both terms. Fast serving raises the cycle count. Long retained
history, diff images and persistent code mean each cycle starts from everything already learned. Our additions only reshaped
context that the model already had, so they changed neither term much. The data agree: the loss sits on levels never
understood (§2.3), and the cycle budget is capped by KV memory (§2.4). The method transfers to any agentic benchmark scored
on a hidden set: reproduce the strongest public system first, instrument the mechanism, and size each test against the noise.

## 4. Lessons

1. **Count the samples before running.** With a standard deviation of 1.6 and one sample per day, most ideas cannot be confirmed in
   time. Know that before spending the submission.
2. **Instrument the mechanism, not the score.** Tokens per second, acceptance length and KV usage repeat to about 1 %. They settled two
   serving questions that score samples never could.
3. **Infrastructure fails silently.** Three submissions scored exactly 0 when Kaggle stopped setting the rerun variable. Six gates died
   when containers mounted inputs in an older directory layout. Both fixes are in every notebook (`docs/RESULTS.md` §3.1).
4. **Harness-written notes do not help; the agent's own data might.** Seven note-writing variants all scored at or below the base. The
   only text addition with a public positive report, S1, is the agent's own rule.

## 5. What we would try next

The remaining gap, from 25 to the 70s of the leaders, is speed of understanding. It does not come from action economy or scheduling.
The candidates that address it are a stronger or more token-efficient model, calibrated FP8 KV scales, and a replay bench that judges
serving changes from saved prompts without spending submissions.

## Acknowledgements

Daniel Franzen for publishing the Milestone-2 solution in full; Tufa Labs for the harness; John Pezzulli (Pennyroyal), Mamy
Ratsimbazafy and Gabriel Olympie for SGLang work; Intel (AutoRound) and Albucino (MTP draft); `lalalia9`, `rellik13` and `heliosli`
for S1 and C1; the ARC Prize Foundation and Kaggle.

All numbers come from [docs/RESULTS.md](RESULTS.md). The figures are generated by `scripts/make_writeup_figures.py`.

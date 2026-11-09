# What Moves an ARC-AGI-3 Agent: The Base System, Measured Honestly

*ARC Prize 2026 Paper Track (ARC-AGI-3). A faithful reproduction of the strongest public agent, about 25 controlled changes
on top of it, and where the metric is actually lost.*

[Español](WRITEUP.es.md) · [Back to the README](../README.md)

**Julian Camilo Villaduque** · ARC Prize 2026, ARC-AGI-3 · Best submission **27.63** (ID 56893893, notebook
[arc-agi3-franzen-m2](https://www.kaggle.com/code/juliancamilovilla/arc-agi3-franzen-m2)); final stack X2 mean 25.2, best 27.30
(ID 57924975, [arc-agi3-franzen-x2](https://www.kaggle.com/code/juliancamilovilla/arc-agi3-franzen-x2))

Code (MIT): https://github.com/jvilladuque90/arc-prize-2026-arc-agi-3

![Cover](figures/cover.png)

---

## In plain words

ARC-AGI-3 drops an agent into small video games it has never seen. There are no instructions. It must press buttons, watch
what happens, work out the rules and clear the levels. Fewer actions earn more points. A human does this in minutes.

My agent is a language model that plays by writing short Python programs. It sees each frame as an image and remembers
what it tried. I did not invent that agent. Daniel Franzen published it, and I copied it exactly. Then I asked one question:
**which changes really raise the hidden score, and which only look like they do?**

My contribution is a **measurement method** and a **diagnosis**:

1. **The base system is the lever.** Swapping my own pipeline for Franzen's took me from 3.2 to 25.3. Twenty-five careful
   tweaks on top of it moved nothing beyond the noise. One lowered the score by three standard deviations.
2. **The points are lost on levels the agent never understands, not on wasted moves.** 55 % of the score is lost on levels
   never solved. Only 2 % is lost on solved levels that took too many actions.
3. **The server runs out of memory before it runs out of compute.** Running more games at once makes the GPU faster at
   peak, yet it produces 21 % less thinking.
4. **Quick local tests cannot rank ideas.** Their noise is larger than any effect I chased.

Every experiment is reported, negatives included, so the next team need not pay for them again.

![Score by base system](figures/fig1_history.png)

---

## 1. The system

### 1.1 The base I copied

The control is an unmodified copy of Franzen's Milestone-2 notebook, which builds on Tufa Labs' harness.

| Part | What it does |
|---|---|
| Model | Qwen3.8-Flash-Next, Intel AutoRound INT4 weights, Albucino's MTP draft model for speculative decoding |
| Serving | Pennyroyal (an SGLang fork for the RTX PRO 6000): 128K context, FP8 KV cache of 1.01 M tokens, 10 games decoding at once |
| Agent | Keeps its full reasoning in history; acts through a Python tool; sees the board as an image plus animation frames; reads each game over from a diff image; can undo; keeps the helper functions it writes |
| Scheduler | Gives the GPU to the games making the most progress per token over a 9-hour budget |

### 1.2 What I added

Each addition is one notebook cell that patches the harness by exact text replacement. Each one has a unit test against the
real harness file.

| Arm | What it changes | Source |
|---|---|---|
| **S1** solved-level memory | When a level is cleared, its facts and a one-line rule written by the model are pinned to the system prompt | `lalalia9` (CC-BY 4.0), after `rellik13` |
| **M1** seeded helpers | Each game starts with 7 tested utilities, chosen from what the model rewrote most often in 70 saved logs | mine |
| **C1** stagnation reset | After 96K tokens stuck on one level, clear the reasoning context once | `heliosli` |
| **H1** sandbox fixes | Keep the function library after a timeout; cap huge tool outputs | mine |
| **X1, X2** | X1 = S1 + M1; X2 = X1 + C1 (my final submission) | — |

I also tested serving changes: deeper speculative drafting (T1), 12 parallel games (T2), lower reasoning effort (E1) and
deeper history (D1).

### 1.3 How I tested

- **One variable per arm, against a faithful control.** Each arm differs from the control in exactly one cell.
- **A local gate only catches breakage.** It is a 25-minute run on 10 public games. It also records decode speed and
  speculative acceptance. Those repeat to about 1 %, so they can judge serving changes when the score cannot.
- **Only the hidden set ranks.** I submitted once a day, compared arms game by game with sign tests, and wrote each
  decision rule down before its result arrived.

---

## 2. Results

### 2.1 The hidden set

![Hidden-set samples by arm](figures/fig2_arms.png)

The control's eight samples run from 23.4 to 27.6: mean 25.3, standard deviation 1.6. With one submission a day, telling two
arms apart by 2 points takes about 8 samples each. No arm got close to that many.

- **E1 hurt.** Lowering the reasoning effort tripled the local score on hard games, then scored 20.98 on the hidden set.
- **Everything else stayed inside the band.** S1, D′ and X1 sat slightly above the control mean. X2, my final stack, ended at
  25.2 over five samples.

### 2.2 Why local tests cannot rank

![Gate score versus hidden score](figures/fig3_gates.png)

Identical gate runs differ by tens of points per game. My best gate of the final month, C1 at 50.7, would have been a
false lead. An older local bench was even anti-correlated with the hidden set across four arms (r = −0.35).

### 2.3 Where the score is lost

![Metric decomposition](figures/fig4_metric_loss.png)

The metric squares the ratio of baseline actions to actions taken, per level, and weights later levels more. I split 50 gate
game runs by level. Solved levels already score close to full: 40 % of the weight comes from levels solved within
the baseline. The loss is on levels never solved, usually the later ones that weigh most. On hard public games each action
costs about 2,200 reasoning tokens, against 600 on easy ones. **The gap is how fast the model understands a new game**, not
how efficiently it moves.

### 2.4 Thinking is capped by memory

![Throughput at 10 and 12 streams](figures/fig5_concurrency.png)

If the loss is on levels not understood, more thinking per game should help. So I tried to squeeze more tokens out of the
server.

- **Deeper drafting:** impossible. The attention kernels cap draft tokens at 4.
- **12 parallel games:** peak decode rose from 687 to 759 tokens per second. Then the KV pool filled up, games were evicted
  and re-read, and total output fell from 842 K to 662 K tokens.

Franzen's choice of 10 parallel games sits at the optimum.

---

## 3. Why it works, and why the tweaks did not

An ARC-AGI-3 game is solved by guessing a hidden rule, testing the guess with a few actions, and revising it. Here every
revision is paid in reasoning tokens. So the score is roughly:

**score ≈ (hypothesis cycles a game receives) × (chance that a cycle is right)**

The base system raises both terms. Fast serving raises the number of cycles. Long history, diff images and persistent code
mean each cycle starts from everything already learned. My additions only rearranged context the model already had, so they
moved neither term. The data agree. The loss sits on levels never understood (§2.3), and the number of cycles is capped by
KV memory (§2.4).

The lesson transfers to any agentic benchmark scored on a hidden set. Reproduce the strongest public system first, measure
the mechanism, and size every test against the noise before running it.

## 4. Lessons

1. **Count the samples before running.** With a standard deviation of 1.6 and one sample a day, most ideas cannot be
   confirmed in time. Know that before spending the submission.
2. **Measure the mechanism, not the score.** Tokens per second, acceptance length and KV usage repeat to about 1 %. They
   settled two serving questions that score samples never could.
3. **Infrastructure fails silently.** Three of my submissions scored exactly 0 when Kaggle stopped setting the rerun
   variable. Six gates died when containers mounted inputs in an older directory layout. Both fixes are in every notebook.
4. **Notes written by the harness do not help; the agent's own words might.** Seven note-writing variants all scored at or
   below the base. The only text addition with a positive public report, S1, is the agent's own rule.

## 5. What I would try next

The gap from 25 to the leaders' 70s is speed of understanding. It does not come from action economy or scheduling. The
candidates that attack it are a stronger or more token-efficient model, calibrated FP8 KV scales that free memory for more
thinking, and a replay bench that judges serving changes from saved prompts without spending submissions.

## Acknowledgements

Daniel Franzen for publishing the Milestone-2 solution in full; Tufa Labs for the harness; John Pezzulli (Pennyroyal), Mamy
Ratsimbazafy and Gabriel Olympie for the SGLang work; Intel (AutoRound) and Albucino (MTP draft); `lalalia9`, `rellik13` and
`heliosli` for S1 and C1; the ARC Prize Foundation and Kaggle.

All numbers come from [docs/RESULTS.md](RESULTS.md). The figures are generated by `scripts/make_writeup_figures.py` and
`scripts/make_cover.py`.

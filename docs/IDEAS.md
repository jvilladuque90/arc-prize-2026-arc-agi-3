# Research: ideas, evidence and open veins

[Español](IDEAS.es.md) · [Back to the README](../README.md)

Sources: the write-up in Franzen's public repository (`WRITEUP.md`), the public kernels of the Milestone-2 winners, a
comparison of ~25 public forks, and our own tests ([RESULTS](RESULTS.md)). Evidence labels: **[H]** hidden-set score by the
author, **[L]** local measurement, **[C]** author claim without a number, **[ours]** our test.

## 1. Ideas with external evidence

| Idea | Evidence | Where it stands for us |
|---|---|---|
| A stronger, faster base: long retained history + FP8 KV cache + fast serving | The three public milestone winners share it; one public ladder 6.4 → 22.5 over 12 days, with FP8 KV + history 49K → 69.6K worth **+8.0** [H, single samples] | **Adopted** (it *is* the base: 3 → 25.7) |
| Larger board image (4× → 10×) | "the biggest score improvement in terms of images" [C] | In the base |
| Animation frames as data in the sandbox | "the strongest benefit" of the perception changes [C]; +0.5 on an older stack [H] | In the base; on the *old* stack ours was neutral |
| Game-over diagnosis with the fatal frame and a diff image | the one-line message "led to long confused analyses" [C] | In the base |
| Keep the model's own thinking in history | removing it caused "a large regression in score" [C] | In the base |
| **No** harness-written memory / world-model notes | disabling it was "the largest improvement in this area" [C]; extra prompt text cost −2.3 at 37K context [H] | Matches our 7 failed note arms [ours] |
| Persistent agent-written functions | in the base; a test-gated per-game library scored 23.8 [H] | In the base; hardened by H1 (neutral) |
| Block trimming + bounded concurrency for prefix-cache reuse | 93 % of prompt tokens reused [L] | In the base |
| Priority scheduling by progress, tokens and continuation value | calibrated on 725 level attempts; no ablation [C] | In the base; variant D′ being tested |
| Agent-pinned "solved-level rules" | **+2.8** [H, single sample] | **Not tried** on the new base |

## 2. What was missing to make our own ideas realizable

| Blocker | Effect |
|---|---|
| **No reliable validation signal.** One submission/day, run-to-run sd 1.6–3.9 | 4-vs-4 samples resolve ~4–8 points; small gains can never be confirmed |
| **Local gates do not rank.** Per-run sd of a 10-game demo ≈ 4; a ×3 local gain became −4.7 on the hidden set | Wasted ~5 arms on false positives |
| **Memory headroom.** The GPU has 1.6–2.2 GiB free during play; KV pool is ~90 % full | Any change that adds context, slots or weights (D1, model swaps) trades against throughput |
| **Two GPU sessions at a time**, ~40 min per gate | Serialises experiments |
| **The leaders are private.** Ranks 1–2 (68.9, 67.5) are ~6 sd above the public-copy distribution | We cannot tell whether their gain is a better model, search, training or harness |
| **Windows development vs a Linux sandbox** | The 65 KB sandbox bootstrap cannot be launched locally; tests cover the host-side code only |

## 3. Open veins, ranked

| # | Idea | Expected effect | Cost / how to validate |
|---|---|---|---|
| V1 | **Mechanical replay bench**: replay the saved `*_requests.jsonl` prompts against any server config and measure tokens/s, speculative acceptance, time-to-first-token, cache hits | Enabler for V2–V4, V9: judges serving/numerics changes *without* spending submissions | CPU + a short GPU run; no new model quality needed |
| V2 | **Calibrated FP8 KV scales** (the server warns that none are supplied) | Quality lever at zero memory cost; effect unknown | One calibration pass on ARC prompts + replay bench; hidden check |
| V3 | **Throughput**: MTP depth 4 (+4–6 % derived), 12 slots (needs more KV), online MXFP8 (~9 % claimed), relaxed speculative acceptance (lossy) | Franzen: "scores were still improving near the end of runs" → time is a lever, but each item is small | Replay bench for the lossless ones; the lossy ones cannot be validated |
| V4 | **Pre-seeded helper functions** in the persistent library — **built as M1** (`src/arc3/franzen_memory.py`, 7 functions chosen from what the model rewrites in 70 saved logs: `snap` in 12 games, `crop` in 10, `summ` in 9, …) | Saves the first turns of every game; code, not prose, so consistent with "frames yes, narration no"; ~40 extra tokens per opener | Unit-tested on the real harness (seeds pass its retention validation, a new game starts seeded); needs a gate run, then hidden samples |
| V5 | **Stagnation reset** (public, untested) — **built as C1** (`src/arc3/franzen_stagnation.py`, ported from `heliosli`'s notebook): after 96K tokens on an unfinished level, clear the reasoning context and functions once | Could unstick looping hard games | Unit-tested; a 25-minute gate almost never reaches 96K tokens on one level, so only the hidden rerun exercises it; its author has no comparison either |
| V6 | **Test-gated function retention** | Better library quality; closest precedent 23.8 [H] | Moderate engineering; low expected gain here (library already holds ~15 functions per game) |
| V7 | **Keep literal constants alongside functions** (42 % of function attempts are rejected for "unavailable dependencies") | Modest | Touches host and sandbox; cannot be run end to end on Windows |
| V8 | **Scheduler priority variants** (D′ running; pace on) | Probably within noise | Daily alternation with the control |
| V9 | **A different checkpoint** (reasoning-efficient fine-tune, NVIDIA NVFP4, expert-pruned) | The only route to a large gain, but the best-documented candidate *raised* total tokens by 11.9 % in agentic use and its 4-bit format shrinks the KV pool ~20 % | Only after V1; cannot be validated on the hidden set |

## 4. Not tried, and why

- **Test-time training / fine-tuning on the public games**: the hidden games differ from the public ones, so the risk is
  overfitting and contamination; the nine-hour budget leaves no room for training inside the run.
- **Self-consistency / voting**: tokens per game are the binding resource.
- **A learned world model or a model-written simulator**: external evidence is negative (0 of 31 induced goal predicates held).
- **Reproducing the leaders**: unknown approach; the only public hints are their scores.

## 5. Note on `reasoning_effort`

Qwen3.8's chat template defaults to `xhigh` ("think carefully, validate assumptions, consider alternatives"). Lowering it to
`medium` tripled the local score on hard games but lowered the hidden score ([RESULTS](RESULTS.md)). We read this as: deliberate
reasoning matters on the hidden games we cannot see, and the local hard-game gain came from games where extra caution is wasted.
Adaptive effort is therefore *not* a priority.

# ARC Prize 2026 — ARC-AGI-3: an open-source agent and its engineering log

[Español](README.es.md) · [How it works](docs/APPROACH.md) · [Results & lessons](docs/RESULTS.md) · [Research & open ideas](docs/IDEAS.md) · [Roadmap](docs/ROADMAP.md)

This repository documents a competition entry for the [ARC Prize 2026 – ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3)
Kaggle competition, where an agent must learn the rules of **interactive grid games it has never seen** and complete as many
levels as possible, using as few actions as possible, on a single GPU with no internet.

It is both a working submission pipeline and an honest record of what was tried: what moved the hidden-set score, what did not,
and the measurement mistakes made along the way. All kernels are public and the code is MIT-licensed, as the prize requires.

## Where things stand (2026-10-17)

| | |
|---|---|
| Current base | A faithful copy of Daniel Franzen's public Milestone-2 solution (SGLang, INT4 Qwen3.8-Flash-Next, 128K context) |
| Hidden-set score of the base | **25.7 mean, sd 1.6 (n = 6)**, best 27.63 |
| Leaderboard (4,425 teams) | top 68.9 · 3rd 50.4 · 10th 41.7 · 100th 33.2 · this entry 27.63 (rank ~1,090) |
| Score before adopting that base | 3–5 |

The score history is dominated by one lesson: **changing the base system moved the score from ~3 to ~26; every one of the
roughly 25 prompt/harness tweaks tested on top of a base moved it by less than the noise, or hurt.**

## The system in one minute

An LLM plays each game by writing small Python programs. A sandbox hands it the board as an image, a "what changed" image,
the animation frames, and a segmentation of the board into objects; it answers by calling `action([...])`. It remembers its
own reasoning (128K tokens of history) and the helper functions it wrote. A scheduler shares one GPU among 110 games, and the
model is served 2–3× faster than a standard vLLM setup. Details: [docs/APPROACH.md](docs/APPROACH.md).

## What we learned (short version)

1. **The base system dominates.** Switching to a stack with long retained history and fast serving gave ~8× the score of any
   later tweak. Look at what the top public solutions share before tuning anything.
2. **Text notes written by the harness do not help** ("frames yes, narration no"): data exposed in the sandbox and images helped,
   injected prose did not (7 memory/notes arms, all neutral or negative).
3. **Local benchmarks mislead.** A change that tripled the score on 15 hard games locally (`reasoning_effort=medium`) scored
   *lower* on the hidden set (20.98 vs 25.7). Per-game noise on a 10-game demo is large (sd of a run's mean ≈ 4).
4. **One submission per day makes A/B testing slow.** With sd ≈ 1.6–3.9, four samples per arm only resolve differences of
   ~4–8 points, so small improvements cannot be confirmed — they have to be argued from mechanism and external evidence.
5. **Score is limited by levels completed, not by action efficiency**, on the games the agent can solve; the gap to the leaders is
   in how fast it *understands* a new game (first level: 2.6 min on easy games vs 11.6 min on hard ones).

Full tables, errata and the list of ideas still open: [docs/RESULTS.md](docs/RESULTS.md), [docs/IDEAS.md](docs/IDEAS.md).

## Repository map

```
README.md · README.es.md      this overview (English / Spanish)
docs/
  APPROACH.md                 how the current system works, explained simply
  RESULTS.md                  submission history, every experiment, corrections to earlier claims
  IDEAS.md                    research: ideas proven elsewhere, what blocked them, open veins
  ROADMAP.md                  what is running and what comes next
  archive/                    the original Spanish lab notebook and older design notes (historical)
src/arc3/                     harness patches applied to Franzen's code (franzen_hardening.py, franzen_effort.py)
scripts/                      kernel publishing, submission automation, notebook builder (build_arm.py)
notebooks/                    the Kaggle notebooks actually submitted or used as diagnostics
tests/                        tests for the harness patches (run against the real patched harness)
archive/legacy/               the superseded pipeline: state-graph explorer, vLLM/NVFP4 arms, Colab benches
kernel_versions.json          pushed kernel versions (needed at submit time)
```

## Reproduce

```bash
pip install -r requirements.txt
# create a .env file with two lines:  KAGGLE_API_TOKEN=...  and  kaggle_username=...   (git-ignored; never commit it)
python scripts/download_data.py                       # competition data (not versioned)
kaggle kernels pull dfranzen/arc-agi-3-milestone-2-solution -p _tmp_pub/dfranzen_arc-agi-3-milestone-2-solution -m
python scripts/build_arm.py fork                  # notebooks/franzen_m2.ipynb (faithful copy, checked cell by cell)
python scripts/prepare_harness_tree.py                # patched harness tree used by the tests
python tests/test_franzen_hardening.py && python tests/test_franzen_effort.py
python scripts/push_kernels.py franzenm2 --dry-run    # inspect; add --gpu (no --dry-run) to run on Kaggle
python scripts/submit_at_reset.py juliancamilovilla/arc-agi3-franzen-m2 "message"   # waits for the daily quota
```

Daily submissions alternate between the control and the current candidate with `scripts/daily_alternate.py`
(armed as a Windows scheduled task by `scripts/arm_submit_task.ps1`).

## Attribution

This entry stands on public work, used under its licenses and credited in every notebook:
**Daniel Franzen** (Milestone-2 solution and harness patch), **Tufa Labs** (the Duck harness — Harold Bessis, Jeroen Cottaar,
Isaiah Pressman, Andries Smit, Michal Tesnar, Stefano Viel), **John Pezzulli** (Pennyroyal, the SGLang fork), **Mamy Ratsimbazafy**
and **Gabriel Olympie** (SGLang patches), **Intel** (AutoRound quantization), **Albucino** (MTP drafter), **shiiin9 / Affectify**
(scheduler priority D′), **Scott Le Grand** and **keithtyser** (the earlier vLLM/NVFP4 base). The Qwen3.8 model weights remain under
the Qwen license.

Code in this repository: MIT, © 2026 Julian Camilo Villa Duque.

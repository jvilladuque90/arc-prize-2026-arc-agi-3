# AGENTS.md — ARC Prize 2026, ARC-AGI-3

Guide for any agent (human or AI) working in this repository. Start with [README.md](README.md); the current state, results
and plan live in `docs/` (RESULTS, IDEAS, ROADMAP). Spanish versions of the public docs end in `.es.md`.

## Current state (2026-10-17)

- Base: a faithful copy of Daniel Franzen's public Milestone-2 solution (`notebooks/franzen_m2.ipynb`). Hidden-set score
  **25.7 mean, sd 1.6 (n = 6)**.
- Daily submissions alternate between that control and the D′ scheduler variant (`scripts/submit/daily_alternate.py`, armed as the
  Windows task `ARC-AGI3-SubmitOneShot`). The decision rule is in `docs/ROADMAP.md`.
- Every experiment tried on top of a base so far was neutral or negative (`docs/RESULTS.md`). The superseded pipeline is in
  `archive/legacy/` and is not maintained.

## Rules that are not negotiable

1. **Never commit or redistribute** `.env`, `environment_files/`, `arc_agi_3_wheels/`, `ARC-AGI-3-Agents/`, or any third-party
   bundle under `_tmp_*/`. Never print credentials.
2. Kernels are **public** (`is_private: False`): the prize requires open-sourcing everything.
3. The metric is `(baseline_actions / actions_taken)^2` per completed level (capped at 115), levels weighted by their index.
   More levels with fewer actions — not "levels" alone.
4. **One variable per arm**, against a **faithful control**. Arms are notebooks that differ from the control in a single cell.
5. **Local gates screen out broken changes; they never rank improvements** (`docs/RESULTS.md` §3). Compare hidden-set
   samples against the control, and count how many samples the claimed effect would need before running.
6. Use paired statistics by game (sign test), never pooled proportions.
7. **GPU launches and submissions are announced and authorised first.** Prefer cheap, CPU-only checks before spending quota.
8. Do not inject harness-written prose into the model's prompt ("frames yes, narration no"): data exposed in the sandbox and
   images helped, notes did not.

## Layout and common commands

| Path | Purpose |
|---|---|
| `src/arc3/` | patches applied to Franzen's `tool_agent.py` by exact-anchor replacement: `franzen_hardening.py` (H1), `franzen_effort.py` (E1), `franzen_memory.py` (M1), `franzen_stagnation.py` (C1) |
| `scripts/kaggle/build_arm.py` | builds `notebooks/franzen_*.ipynb` (control copy and one-cell arms) with cell-by-cell gates |
| `scripts/kaggle/push_kernels.py` | publishes a notebook as a public Kaggle kernel (`--gpu` to run on the RTX PRO 6000; `--dry-run` first) |
| `scripts/submit/submit_at_reset.py`, `daily_alternate.py`, `arm_submit_task.ps1` | submission automation |
| `scripts/kaggle/prepare_harness_tree.py` | builds the patched harness the tests run against |
| `tests/`, `scripts/check_repo.py` | one test file per patch; `python scripts/check_repo.py --tests` checks docs, registry, notebooks, links and tests together |

## Documentation discipline

Public docs are bilingual (English first, `.es.md` beside it) and short. Record each result once in `docs/RESULTS.md`
(and `.es.md`), the open questions in `docs/IDEAS.md`, and the plan in `docs/ROADMAP.md`. Commit messages: a short subject, a
two-line body, then a pointer to the doc section. The Spanish lab notebook in `archive/docs/` is frozen.

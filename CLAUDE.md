# CLAUDE.md

@AGENTS.md

## Notes specific to Claude Code

- Windows: download kernel logs with `PYTHONUTF8=1`, otherwise they come out empty (cp1252 errors).
- Kaggle credentials live in `.env` (`set -a; . ./.env; set +a`); never print or commit them. `.env*` files are protected by
  a permission rule, so do not try to create `.env.example`.
- Scripts containing `\n` or quotes: write them with the Write tool, not Bash heredocs. Commit with `git commit -F <file>`.
- Do not launch GPU work (kernel pushes, Save & Run) or submissions without announcing them and getting a go-ahead.
- Token quota is limited: do not start large multi-agent workflows without agreeing their size first (a 13-agent audit on
  2026-10-03 spent ~7 M tokens and hit the session limit). Have agents write their reports to disk before returning.
- Background jobs are killed after roughly 30 minutes: poll with a loop capped at ~25 minutes and relaunch it.
- A kernel pushed with Kaggle's public python image runs without the GPU; use the pinned image in `scripts/push_kernels.py`.

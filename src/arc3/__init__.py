"""arc3: harness patches for the ARC-AGI-3 Kaggle entry (competition `arc-prize-2026-arc-agi-3`).

Modules:
  franzen_hardening -> H1: keep the persistent-function library after a sandbox timeout; cap huge tool results
  franzen_effort    -> E1: pin the model's `reasoning_effort` from the first request

Each module patches the source of Daniel Franzen's public harness (`tool_agent.py`) by exact-anchor replacement
and refuses to touch anything if an anchor is not found exactly once. The superseded pipeline lives in `archive/legacy/`.
"""

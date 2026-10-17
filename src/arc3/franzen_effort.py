"""E1: pin the model's `reasoning_effort` from the first request on Franzen's patched harness (docs/RESULTS.md).

Qwen3.8-Flash-Next's chat template reads `reasoning_effort` and DEFAULTS TO `xhigh`, which injects into the system
prompt "think carefully, validate key assumptions, consider alternatives, prioritise correctness". `low` injects
"keep your thinking brief" and `medium` injects nothing. Franzen runs everything at the default (his effort ladder only
steps down AFTER a truncated reply and is off).

The value reaches the server as chat_template_kwargs["reasoning_effort"], by the same path the ladder already uses.
If ARC3_STATIC_REASONING_EFFORT is empty nothing changes.

Outcome: `medium` tripled the local score on 15 hard public games but scored BELOW the control on the hidden set
(20.98 vs 25.7), so this arm is kept as a documented negative result, not as a recommendation.

The anchor must appear EXACTLY once or nothing is touched; line endings are preserved.
"""

from __future__ import annotations

A = (
    "        ladder = _reasoning_effort_ladder()\n"
    "        rung = getattr(self, \"_reasoning_effort_rung\", -1)\n"
    "        if ladder and 0 <= rung < len(ladder):\n"
    "            kwargs[\"reasoning_effort\"] = ladder[rung]\n"
    "        return kwargs\n"
)
B = (
    "        # E1: fixed effort from the start (the ladder, when active, overrides it afterwards)\n"
    "        static_effort = os.environ.get(\"ARC3_STATIC_REASONING_EFFORT\", \"\").strip()\n"
    "        if static_effort:\n"
    "            kwargs[\"reasoning_effort\"] = static_effort\n"
    "        ladder = _reasoning_effort_ladder()\n"
    "        rung = getattr(self, \"_reasoning_effort_rung\", -1)\n"
    "        if ladder and 0 <= rung < len(ladder):\n"
    "            kwargs[\"reasoning_effort\"] = ladder[rung]\n"
    "        return kwargs\n"
)
MARKER = "ARC3_STATIC_REASONING_EFFORT"
VALID_LEVELS = ("xhigh", "medium", "low")


def apply_to_source(src: str) -> str:
    """Return tool_agent.py with the fixed effort. Raises if the anchor is not unique."""
    if MARKER in src:
        return src
    nl = "\r\n" if "\r\n" in src else "\n"
    a, b = A.replace("\n", nl), B.replace("\n", nl)
    n = src.count(a)
    if n != 1:
        raise RuntimeError(f"anchor found {n} times, expected 1")
    return src.replace(a, b, 1)

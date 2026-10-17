"""H1: model-risk-free hardening of Franzen's patched harness (docs/RESULTS.md).

Two defects, both verified by execution against the real `ToolAgent`:

R1. A sandbox timeout or crash returns a result WITHOUT the `keepable_functions` key. `_record_retained_functions`
    then builds `kept = {}` and WIPES the whole persistent-function library for the rest of the game (and tells
    the model "no longer retained"). The kind of snippet that exhausts the 30 s limit (long search, loop over many
    frames) is exactly what the prompt encourages.
    Fix: if the key is absent the run said nothing about the library, so keep it. A successful run ALWAYS carries the
    key (a list, possibly empty), and a deliberate clear (`"all"` rejected) carries it too, so those paths are unchanged.

R2. `_render_tool_payload` truncates only `str` values; a list/dict `result` (180 KB in a probe) is dumped whole
    into the context. Fix: if its JSON dump exceeds 16,000 characters (~4x the 3,072-token output budget) it becomes
    text and goes through the normal truncation.

A normal run is unchanged. Each anchor must appear EXACTLY once or nothing is touched; line endings are preserved.
"""

from __future__ import annotations

A1 = (
    "        try:\n"
    "            previous = self._kept_functions\n"
    "            kept = {}\n"
)
B1 = (
    "        try:\n"
    "            if \"keepable_functions\" not in sandbox_result:\n"
    "                # H1/R1: sandbox timeout, crash or invalid reply says nothing about the library: keep it\n"
    "                # (it used to be wiped entirely)\n"
    "                return\n"
    "            previous = self._kept_functions\n"
    "            kept = {}\n"
)
A2 = (
    "            value = result.get(field)\n"
    "            if isinstance(value, str):\n"
    "                value, suppressed_rows = _suppress_board_dumps(value)\n"
)
B2 = (
    "            value = result.get(field)\n"
    "            if value is not None and not isinstance(value, str):\n"
    "                # H1/R2: a huge list/dict result used to be dumped whole into the context\n"
    "                try:\n"
    "                    dumped = json.dumps(value, ensure_ascii=False, default=str)\n"
    "                except Exception:\n"
    "                    dumped = str(value)\n"
    "                if len(dumped) > 16000:\n"
    "                    value = dumped\n"
    "                    result[field] = dumped\n"
    "            if isinstance(value, str):\n"
    "                value, suppressed_rows = _suppress_board_dumps(value)\n"
)

PAIRS = ((A1, B1), (A2, B2))
MARKER = "H1/R1"


def apply_to_source(src: str) -> str:
    """Return the hardened tool_agent.py source. Raises if any anchor is not unique."""
    if MARKER in src:
        return src
    nl = "\r\n" if "\r\n" in src else "\n"
    out = src
    for a, b in PAIRS:
        a_n, b_n = a.replace("\n", nl), b.replace("\n", nl)
        n = out.count(a_n)
        if n != 1:
            raise RuntimeError(f"anchor found {n} times, expected 1: {a.strip().splitlines()[0]!r}")
        out = out.replace(a_n, b_n, 1)
    return out

"""Tests for S1 (src/arc3/franzen_solved_levels.py): solved-level memory. CPU only.

Checks the pure helpers (rule capture, action attribution per level, block layout), then applies the runtime patch to the
REAL ToolAgent class of the patched harness and drives the four hooks on a minimal fake agent: a cleared level is recorded,
the model is asked for its rule for at most two turns, the rule is captured, and the block lands at the end of the system
prompt only when history is evicted/trimmed (prefix-cache safe). Needs the patched harness tree
(`python scripts/kaggle/prepare_harness_tree.py`, or ARC3_HARNESS_TREE).

Usage:  python tests/test_franzen_solved_levels.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
TREE = Path(os.environ.get("ARC3_HARNESS_TREE", ROOT / ".work" / "pub" / "franzen" / "src" / "ARC3-Inference"))
sys.path.insert(0, str(TREE))

failures: list[str] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    print(("  ok    " if ok else "  FAIL  ") + name + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def main() -> int:
    import inference.agent.tool_agent as ta

    originals = {n: getattr(ta.ToolAgent, n) for n in
                 ("analyze", "_build_user_prompt", "_note_history_evicted", "_trim_messages_for_context")}
    ns: dict = {}
    exec(compile((ROOT / "src" / "arc3" / "franzen_solved_levels.py").read_text(encoding="utf-8"),
                 "franzen_solved_levels.py", "exec"), ns)

    print("1. THE PATCH TAKES ON THE REAL CLASS")
    for n in originals:
        check(getattr(ta.ToolAgent, n) is not originals[n], f"ToolAgent.{n} replaced")
    check(ns["_slm_original_analyze"] is originals["analyze"], "original analyze kept for chaining")

    print("2. PURE HELPERS")
    rule = ns["_slm_find_rule"]("thinking...\nRule for level 2: click the blue key, then walk the door; the wall toggles on each press\n", 2)
    check(rule is not None and rule.startswith("click the blue key"), "rule captured from text", repr(rule))
    check(ns["_slm_find_rule"]("Rule for level 2: short", 2) is None, "a too-short rule is rejected")
    check(ns["_slm_find_rule"]("Rule for level 3: " + "x" * 50, 2) is None, "another level's rule is not taken")
    frames = [SimpleNamespace(action="ACTION1", frame=SimpleNamespace(level=1)),
              SimpleNamespace(action="RESET", frame=SimpleNamespace(level=1)),
              SimpleNamespace(action="ACTION2", frame=SimpleNamespace(level=1)),
              SimpleNamespace(action="ACTION3", frame=SimpleNamespace(level=2)),   # this action cleared level 1
              SimpleNamespace(action="ACTION4", frame=SimpleNamespace(level=2))]
    check(ns["_slm_level_actions"](frames, 1) == ["ACTION2", "ACTION3"], "winning attempt's actions per level (RESET restarts)",
          str(ns["_slm_level_actions"](frames, 1)))
    block = ns["_slm_block"]([{"level": 1, "n_actions": 40, "actions": ["ACTION1"] * 30, "rule": "r" * 30, "reasoning": "why"}])
    check(block.startswith(ns["_SLM_HEADER"]) and block.endswith(ns["_SLM_FOOTER"]), "block has header and footer")
    check("(10 earlier omitted)" in block and "Your rule: " + "r" * 30 in block, "block lists omitted count and rule")
    check(ns["_slm_block"]([]) == "", "no levels -> empty block")

    print("3. THE HOOKS ON A FAKE AGENT")
    with tempfile.TemporaryDirectory() as tmp:
        state = Path(tmp) / "state.json"
        # minimal runtime state: level 1 cleared by ACTION3 (frame after it is level 2)
        state.write_text(json.dumps({"frame": None, "history": []}), encoding="utf-8")
        from inference.agent import runtime_state as rs
        fake_entries = frames
        ns["_slm_load_state"] = lambda p: (None, fake_entries)   # noqa: E731  (the module reads via this name)

        class Fake:
            pass
        ag = Fake()
        ag._system_prompt = "SYSTEM PROMPT\n"
        ag._history_messages = [{"role": "assistant", "content": "I pressed ACTION3", "reasoning_content": "door opened so level done"}]
        ag._last_step_summary = {"level": 2}
        ns["_slm_before_turn"](ag, state)
        levels = ag._slm_levels
        check(len(levels) == 1 and levels[0]["level"] == 1 and levels[0]["actions"] == ["ACTION2", "ACTION3"],
              "clearing level 1 is recorded from facts", str(levels))
        check(levels[0]["reasoning"].endswith("level done"), "tail of the clearing reasoning kept")
        # the user prompt asks for the rule, at most _SLM_ASK_TURNS times
        ns["_slm_original_user_prompt"] = lambda self, *a, **k: "USER"   # noqa: E731
        p1 = ns["_slm_user_prompt"](ag); p2 = ns["_slm_user_prompt"](ag); p3 = ns["_slm_user_prompt"](ag)
        check("Rule for level 1:" in p1 and "Rule for level 1:" in p2 and p3 == "USER", "asked exactly twice, then silent")
        # the model answers; the rule is captured on the next turn
        ag._history_messages.append({"role": "assistant", "content": "Rule for level 1: press the key then cross the open door quickly"})
        ns["_slm_before_turn"](ag, state)
        check(levels[0]["rule"] is not None and levels[0]["rule"].startswith("press the key"), "rule captured", repr(levels[0]["rule"]))
        # prefix-cache safety: nothing written to the system prompt until an eviction/trim
        check(ag._system_prompt == "SYSTEM PROMPT\n", "system prompt untouched before eviction")
        ns["_slm_original_evicted"] = lambda self: None   # noqa: E731
        ns["_slm_note_evicted"](ag)
        check(ag._system_prompt.startswith("SYSTEM PROMPT") and ag._system_prompt.rstrip().endswith(ns["_SLM_FOOTER"]),
              "block appended at the END of the system prompt on eviction")
        # trim: the refreshed prompt rides along only when messages were actually dropped
        ns["_slm_original_trim"] = lambda self, msgs, *a, **k: msgs[:1] + msgs[2:]   # noqa: E731
        msgs = [{"role": "system", "content": "SYSTEM PROMPT\n"}, {"role": "user", "content": "a"}, {"role": "user", "content": "b"}]
        out = ns["_slm_trim"](ag, msgs)
        check(out[0]["content"] == ag._system_prompt and len(out) == 2, "trimmed conversation carries the refreshed system prompt")
        ns["_slm_original_trim"] = lambda self, msgs, *a, **k: msgs   # noqa: E731
        out = ns["_slm_trim"](ag, msgs)
        check(out[0]["content"] == "SYSTEM PROMPT\n", "no drop -> system prompt left alone (cache kept)")
        # errors never break the turn
        ns["_slm_original_analyze"] = lambda self, sp, *a, **k: "ANALYZED"   # noqa: E731
        bad = Fake()   # no attributes at all
        check(ns["_slm_analyze"](bad, state) == "ANALYZED", "a failing hook leaves the turn unchanged")

    for n, f in originals.items():
        setattr(ta.ToolAgent, n, f)
    print("\n" + ("ALL PASS" if not failures else f"{len(failures)} FAILURE(S): " + ", ".join(failures)))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())

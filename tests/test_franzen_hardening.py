"""Tests for H1 (src/arc3/franzen_hardening.py) against Franzen's REAL patched tool_agent.py. CPU only.

Builds a `ToolAgent` without __init__ and exercises the two patched methods, comparing original and hardened code.
Needs the patched harness tree: `python scripts/prepare_harness_tree.py` (or set ARC3_HARNESS_TREE).

Usage:  python tests/test_franzen_hardening.py
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
TREE = Path(os.environ.get("ARC3_HARNESS_TREE", ROOT / "_tmp_pub" / "franzen" / "src" / "ARC3-Inference"))
sys.path.insert(0, str(TREE))

from arc3 import franzen_hardening as h1  # noqa: E402

failures: list[str] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    print(("  ok    " if ok else "  FAIL  ") + name + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def main() -> int:
    agent_dir = TREE / "inference" / "agent"
    src = (agent_dir / "tool_agent.py").read_text(encoding="utf-8")

    print("1. APPLICATION")
    hardened = h1.apply_to_source(src)
    check(hardened != src and "H1/R1" in hardened and "H1/R2" in hardened, "both anchors apply")
    check(h1.apply_to_source(hardened) == hardened, "idempotent")
    check("\r\n" not in h1.apply_to_source(src.replace("\r\n", "\n")), "LF input stays LF")
    try:
        h1.apply_to_source(src.replace("kept = {}", "kept = dict()", 1))
        check(False, "a missing anchor raises")
    except RuntimeError:
        check(True, "a missing anchor raises and touches nothing")
    compile(hardened, "tool_agent_h1", "exec")
    check(True, "the hardened source compiles")

    print("2. BEHAVIOUR ON THE REAL TOOLAGENT")
    os.environ.update({
        "ARC3_PERSISTENT_FUNCTIONS": "1", "ARC3_PERSISTENT_FUNCTIONS_SCOPE": "game",
        "ARC3_PERSISTENT_FUNCTIONS_IMPORTS": "1", "LOCAL_ANALYZER_BASE_URL": "http://127.0.0.1:1/v1",
        "LOCAL_ANALYZER_MODEL_ID": "x", "LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS": "3072",
    })
    results = {}
    for label, code in (("original", src), ("h1", hardened)):
        tmp = agent_dir / f"_tool_agent_{label}.py"
        tmp.write_text(code, encoding="utf-8")
        try:
            sys.modules.pop(f"inference.agent._tool_agent_{label}", None)
            mod = importlib.import_module(f"inference.agent._tool_agent_{label}")
            agent = mod.ToolAgent.__new__(mod.ToolAgent)
            agent._tool_output_tokens = 3072
            agent._tool_output_chars = 3072 * 4

            agent._kept_functions = {"bfs": "def bfs(a):\n    return a"}
            note = {}
            agent._record_retained_functions({"error": "Tool timed out after 30s", "stdout": ""}, note)   # timeout
            after_timeout = dict(agent._kept_functions)

            agent._record_retained_functions(
                {"keepable_functions": [{"name": "bfs", "source": "def bfs(a):\n    return a"}]}, {})       # success
            after_success = dict(agent._kept_functions)

            agent._record_retained_functions({"keepable_functions": [], "retention_rejected": [
                {"name": "all", "reason": "limit", "hint": ""}]}, {})                                     # deliberate clear
            after_clear = dict(agent._kept_functions)

            huge = agent._render_tool_payload({"tool": "python", "result": {"rows": ["x" * 80] * 3000}},
                                              truncate_fields=("stdout", "error", "result"))
            small = agent._render_tool_payload({"tool": "python", "result": {"a": 1, "b": [1, 2]}},
                                               truncate_fields=("stdout", "error", "result"))
            results[label] = dict(timeout=after_timeout, success=after_success, clear=after_clear,
                                  huge_len=len(huge), small=json.loads(small)["result"],
                                  timeout_note=note.get("function_retention"))
        finally:
            tmp.unlink(missing_ok=True)

    o, n = results["original"], results["h1"]
    check(o["timeout"] == {}, "ORIGINAL: a timeout wipes the library (the defect exists)", str(o["timeout"]))
    check("bfs" in n["timeout"], "H1: a timeout keeps the library")
    check(n["timeout_note"] is None, "H1: a timeout does not tell the model it lost functions")
    check(n["success"] == o["success"] and "bfs" in n["success"], "normal success: identical")
    check(n["clear"] == {} and o["clear"] == {}, "deliberate clear ('all'): still works")
    check(o["huge_len"] > 200000, "ORIGINAL: a huge result is dumped whole", str(o["huge_len"]))
    check(n["huge_len"] < 30000, "H1: a huge result is truncated", str(n["huge_len"]))
    check(n["small"] == o["small"] == {"a": 1, "b": [1, 2]}, "small result: identical")

    print(f"\n{'PASS' if not failures else 'FAIL — ' + '; '.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())

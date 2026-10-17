"""Tests for E1 (src/arc3/franzen_effort.py) against Franzen's REAL patched tool_agent.py. CPU only.

Needs the patched harness tree: `python scripts/kaggle/prepare_harness_tree.py` (or set ARC3_HARNESS_TREE).

Usage:  python tests/test_franzen_effort.py
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
TREE = Path(os.environ.get("ARC3_HARNESS_TREE", ROOT / ".work" / "pub" / "franzen" / "src" / "ARC3-Inference"))
sys.path.insert(0, str(TREE))

from arc3 import franzen_effort as e1  # noqa: E402

failures: list[str] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    print(("  ok    " if ok else "  FAIL  ") + name + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def kwargs_with(code: str, label: str, env: dict, rung: int = -1) -> dict:
    for k in ("ARC3_STATIC_REASONING_EFFORT", "ARC3_REASONING_EFFORT_LADDER"):
        os.environ.pop(k, None)
    os.environ.update(env)
    tmp = TREE / "inference" / "agent" / f"_ta_{label}.py"
    tmp.write_text(code, encoding="utf-8")
    try:
        sys.modules.pop(f"inference.agent._ta_{label}", None)
        mod = importlib.import_module(f"inference.agent._ta_{label}")
        agent = mod.ToolAgent.__new__(mod.ToolAgent)
        agent._reasoning_effort_rung = rung
        return agent._harness_template_kwargs()
    finally:
        tmp.unlink(missing_ok=True)


def main() -> int:
    os.environ.update({"LOCAL_ANALYZER_BASE_URL": "http://127.0.0.1:1/v1", "LOCAL_ANALYZER_MODEL_ID": "x"})
    src = (TREE / "inference" / "agent" / "tool_agent.py").read_text(encoding="utf-8")
    patched = e1.apply_to_source(src)

    print("1. APPLICATION")
    check(patched != src and e1.MARKER in patched, "the anchor applies")
    check(e1.apply_to_source(patched) == patched, "idempotent")
    check("\r\n" not in e1.apply_to_source(src.replace("\r\n", "\n")), "LF input stays LF")
    try:
        e1.apply_to_source(src.replace("return kwargs", "return dict(kwargs)", 1))
        check(False, "a missing anchor raises")
    except RuntimeError:
        check(True, "a missing anchor raises and touches nothing")
    compile(patched, "tool_agent_e1", "exec")

    print("2. BEHAVIOUR ON THE REAL TOOLAGENT")
    base = kwargs_with(src, "orig", {"ARC3_STATIC_REASONING_EFFORT": "medium"})
    check("reasoning_effort" not in base, "ORIGINAL ignores the variable", str(base))
    unset = kwargs_with(patched, "e1a", {})
    check("reasoning_effort" not in unset, "E1 with the variable empty changes nothing", str(unset))
    for level in e1.VALID_LEVELS:
        kw = kwargs_with(patched, "e1b", {"ARC3_STATIC_REASONING_EFFORT": level})
        check(kw.get("reasoning_effort") == level, f"E1 pins reasoning_effort={level}", str(kw))
    kw = kwargs_with(patched, "e1c", {"ARC3_STATIC_REASONING_EFFORT": "medium",
                                      "ARC3_REASONING_EFFORT_LADDER": "low"}, rung=0)
    check(kw.get("reasoning_effort") == "low", "Franzen's ladder still wins after a truncation", str(kw))
    kw = kwargs_with(patched, "e1d", {"ARC3_STATIC_REASONING_EFFORT": "low"})
    check(kw.get("preserve_thinking") is True, "other kwargs are preserved", str(kw))

    print(f"\n{'PASS' if not failures else 'FAIL — ' + '; '.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())

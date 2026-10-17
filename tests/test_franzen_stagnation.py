"""Tests for C1 (src/arc3/franzen_stagnation.py): fresh reasoning context after a very expensive unfinished level. CPU only.

Checks the restart policy on a stand-in agent, the patch application, and the wiring on the REAL patched ToolAgent.
Needs the patched harness tree: `python scripts/prepare_harness_tree.py` (or set ARC3_HARNESS_TREE).

Usage:  python tests/test_franzen_stagnation.py
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
TREE = Path(os.environ.get("ARC3_HARNESS_TREE", ROOT / "_tmp_pub" / "franzen" / "src" / "ARC3-Inference"))
sys.path.insert(0, str(TREE))

from arc3 import franzen_stagnation as c1  # noqa: E402

failures: list[str] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    print(("  ok    " if ok else "  FAIL  ") + name + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def fake_agent(generated: int, at_level_start: int = 0):
    calls = []
    return SimpleNamespace(
        _session_generated_tokens=generated, _tokens_at_level_start=at_level_start, _last_step_summary={},
        _history_messages=["m"] * 40, _kept_functions={"f": "def f():\n    return 1"}, _retained_clear_notice="x",
        _resume_after_yield=True, _resume_reason="y", _text_chars_per_token=3.9,
        _note_history_evicted=lambda: calls.append("evicted"), _calls=calls)


def main() -> int:
    agent_dir = TREE / "inference" / "agent"
    src = (agent_dir / "tool_agent.py").read_text(encoding="utf-8")

    print("1. THE POLICY ON A STAND-IN AGENT")
    ns: dict = {}
    exec(compile(c1.HELPER_SOURCE, "<stagnation_restart>", "exec"), ns)
    maybe_restart, limit = ns["maybe_restart"], ns["LEVEL_TOKEN_LIMIT"]
    frame = SimpleNamespace(level=3, step=120)
    check(limit == 96 * 1024, "the limit is 96 * 1024 generated tokens", str(limit))
    a = fake_agent(generated=limit - 1)
    check(maybe_restart(a, frame) == "" and len(a._history_messages) == 40, "just under the limit: nothing happens")
    check(maybe_restart(a, None) == "" and maybe_restart(a, SimpleNamespace(level=0, step=0)) == "", "no frame / level 0: nothing")
    a = fake_agent(generated=limit + 5)
    notice = maybe_restart(a, frame)
    check(notice.startswith("Fresh review of the current level"), "over the limit: a notice is returned")
    check(a._history_messages == [] and a._kept_functions == {} and a._calls == ["evicted"], "history and functions are cleared, cache told")
    check(a._resume_after_yield is False and a._text_chars_per_token is None, "resume state and token calibration reset")
    check(maybe_restart(a, frame) == notice, "a failed request (no new tokens) re-sends the same notice")
    a._session_generated_tokens += 1000
    check(maybe_restart(a, frame) == "", "once new tokens exist the notice is not repeated")
    a._session_generated_tokens += limit * 2
    check(maybe_restart(a, frame) == "", "only ONCE per level")
    a._tokens_at_level_start = a._session_generated_tokens - limit - 1
    check(maybe_restart(a, SimpleNamespace(level=4, step=300)) != "", "a new level can trigger it again")
    done = fake_agent(generated=limit + 5)
    done._last_step_summary = {"run_complete": True}
    check(maybe_restart(done, frame) == "", "a completed run is never restarted")

    print("2. PATCH APPLICATION")
    patched = c1.apply_to_source(src)
    check(patched != src and c1.MARKER in patched and patched.count("maybe_restart") == 2, "the three anchors apply (import + call)")
    check(c1.apply_to_source(patched) == patched, "idempotent")
    check("\r\n" not in c1.apply_to_source(src.replace("\r\n", "\n")), "LF input stays LF")
    try:
        c1.apply_to_source(src.replace("self._history_messages = []\n            self._session_total_tokens = 0",
                                       "self._history_messages = []\n            self._session_total_tokens = 1", 1))
        check(False, "a missing anchor raises")
    except RuntimeError:
        check(True, "a missing anchor raises and touches nothing")
    compile(patched, "tool_agent_c1", "exec")

    print("3. WIRING ON THE REAL TOOLAGENT")
    os.environ.update({"LOCAL_ANALYZER_BASE_URL": "http://127.0.0.1:1/v1", "LOCAL_ANALYZER_MODEL_ID": "x",
                       "ARC3_PERSISTENT_FUNCTIONS": "1", "ARC3_PERSISTENT_FUNCTIONS_SCOPE": "game"})
    helper = TREE / "inference" / "utils" / "stagnation_restart.py"
    helper.write_text(c1.HELPER_SOURCE, encoding="utf-8")
    tmp = agent_dir / "_ta_c1.py"
    tmp.write_text(patched, encoding="utf-8")
    try:
        sys.modules.pop("inference.agent._ta_c1", None)
        mod = importlib.import_module("inference.agent._ta_c1")
        agent = mod.ToolAgent.__new__(mod.ToolAgent)
        agent._session_runtime_dir = None
        agent._ensure_session(Path("game_a") / "state.json")
        check(agent._stagnation_restarted_levels == set() and agent._stagnation_pending_notice is None,
              "a new game starts with clean restart state")
        check(mod.maybe_restart is not None, "the harness module imports the policy")
    finally:
        tmp.unlink(missing_ok=True)
        helper.unlink(missing_ok=True)

    print(f"\n{'PASS' if not failures else 'FAIL — ' + '; '.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())

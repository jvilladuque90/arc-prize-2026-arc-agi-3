"""Tests for M1 (src/arc3/franzen_memory.py): pre-seeded helper functions. CPU only.

Checks the seed functions themselves, that the harness's own retention validation accepts them and that no seed collides with a
name the sandbox provides, and the behaviour of the REAL patched ToolAgent (a new game starts from the seeded library).
Needs the patched harness tree: `python scripts/prepare_harness_tree.py` (or set ARC3_HARNESS_TREE).

Usage:  python tests/test_franzen_memory.py
"""

from __future__ import annotations

import importlib
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
TREE = Path(os.environ.get("ARC3_HARNESS_TREE", ROOT / "_tmp_pub" / "franzen" / "src" / "ARC3-Inference"))
sys.path.insert(0, str(TREE))

from arc3 import franzen_memory as m1  # noqa: E402

failures: list[str] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    print(("  ok    " if ok else "  FAIL  ") + name + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def main() -> int:
    agent_dir = TREE / "inference" / "agent"
    src = (agent_dir / "tool_agent.py").read_text(encoding="utf-8")

    print("1. THE SEED FUNCTIONS")
    ns: dict = {}
    for source in m1.SEED_SOURCES.values():
        exec(compile(source, "<seed>", "exec"), ns)
    frame = SimpleNamespace(ascii="..A.\n.AA.\nB...\n")
    check(ns["ascii_rows"](frame) == ["..A.", ".AA.", "B..."], "ascii_rows drops the trailing blank line")
    check(ns["crop"](frame, 0, 2, 1, 3) == ".A\nAA", "crop is half-open [r0,r1) x [c0,c1)")
    check(ns["color_counts"](frame) == {".": 8, "A": 3, "B": 1}, "color_counts")
    check(ns["bbox_of_color"](frame, "A") == (0, 1, 1, 2), "bbox_of_color (r0, c0, r1, c1)")
    check(ns["bbox_of_color"](frame, "Z") is None, "bbox_of_color of an absent color is None")
    summary = ns["summarize_frame"](frame)
    check(summary.startswith(".:8") and "A:3" in summary and "B:1" in summary, "summarize_frame sorts by frequency", summary[:60])
    check(sorted(ns["neighbors4"](0, 0, 3, 4)) == [(0, 1), (1, 0)], "neighbors4 at a corner")
    grid = ["...#.", ".#.#.", ".#...", "...#."]
    passable = lambda r, c: grid[r][c] != "#"  # noqa: E731
    path = ns["bfs_path"]((0, 0), (3, 4), passable, 4, 5)
    check(path is not None and path[0] == (0, 0) and path[-1] == (3, 4) and len(path) == 8, "bfs_path finds a shortest path", str(path))
    blocked = ["..#", "###", "..."]
    check(ns["bfs_path"]((0, 0), (2, 2), lambda r, c: blocked[r][c] != "#", 3, 3) is None, "bfs_path returns None when blocked")

    print("2. HARNESS RETENTION VALIDATION")
    from inference.agent import python_tool_sandbox as sb
    from inference.utils.retained_functions import prepare_functions
    template = sb._SANDBOX_BOOTSTRAP
    block = re.search(r"SAFE_BUILTINS\s*=\s*\{(.*?)\}", template, re.S).group(1)
    safe = set(re.findall(r'"(\w+)"', block))
    provided = set(re.findall(r'runtime_globals\["(\w+)"\]', template))
    kept, rejected = prepare_functions(list(m1.SEED_SOURCES.values()), safe | provided)
    check(not rejected and set(kept) == set(m1.SEED_SOURCES), "all 7 seeds pass the harness retention validation", str(rejected))
    check(not (set(m1.SEED_SOURCES) & (safe | provided)), "no seed collides with a sandbox-provided name")

    print("3. PATCH APPLICATION")
    patched = m1.apply_to_source(src)
    check(patched != src and m1.MARKER in patched, "both anchors apply")
    check(m1.apply_to_source(patched) == patched, "idempotent")
    check("\r\n" not in m1.apply_to_source(src.replace("\r\n", "\n")), "LF input stays LF")
    try:
        m1.apply_to_source(src.replace("self._retained_clear_notice = \"\"", "self._retained_clear_notice = ''", 1))
        check(False, "a missing anchor raises")
    except RuntimeError:
        check(True, "a missing anchor raises and touches nothing")

    print("4. BEHAVIOUR ON THE REAL TOOLAGENT")
    os.environ.update({"LOCAL_ANALYZER_BASE_URL": "http://127.0.0.1:1/v1", "LOCAL_ANALYZER_MODEL_ID": "x",
                       "ARC3_PERSISTENT_FUNCTIONS": "1", "ARC3_PERSISTENT_FUNCTIONS_SCOPE": "game"})
    helper = TREE / "inference" / "utils" / "seed_functions.py"
    helper.write_text(m1.HELPER_SOURCE, encoding="utf-8")
    results = {}
    for label, code in (("original", src), ("m1", patched)):
        tmp = agent_dir / f"_ta_mem_{label}.py"
        tmp.write_text(code, encoding="utf-8")
        try:
            sys.modules.pop(f"inference.agent._ta_mem_{label}", None)
            mod = importlib.import_module(f"inference.agent._ta_mem_{label}")
            agent = mod.ToolAgent.__new__(mod.ToolAgent)
            agent._session_runtime_dir = None
            agent._ensure_session(Path("game_a") / "state.json")
            first = sorted(agent._kept_functions)
            agent._kept_functions = {"mine": "def mine():\n    return 1"}      # the model's own function
            agent._ensure_session(Path("game_a") / "state.json")                # same game: nothing is reset
            same_game = sorted(agent._kept_functions)
            agent._ensure_session(Path("game_b") / "state.json")                # a NEW game
            results[label] = (first, same_game, sorted(agent._kept_functions),
                              agent._retained_function_context())
        finally:
            tmp.unlink(missing_ok=True)
    helper.unlink(missing_ok=True)
    o, n = results["original"], results["m1"]
    check(o[0] == [] and o[2] == [], "ORIGINAL: every game starts with an empty library", str(o[0]))
    check(n[0] == sorted(m1.SEED_SOURCES), "M1: a new game starts with the 7 seeds", str(n[0]))
    check(n[1] == ["mine"], "M1: within the same game the library is untouched", str(n[1]))
    check(n[2] == sorted(m1.SEED_SOURCES), "M1: the NEXT game is re-seeded (the model's own function is gone)", str(n[2]))
    check("bfs_path(start, goal, passable, nrows, ncols)" in n[3], "the opener lists the signatures (~40 tokens)", n[3][:120].replace("\n", " "))

    print(f"\n{'PASS' if not failures else 'FAIL — ' + '; '.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())

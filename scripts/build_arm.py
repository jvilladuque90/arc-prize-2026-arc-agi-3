"""Build the experiment notebooks from Franzen's public notebook (one variable per arm).

  fork                      notebooks/franzen_m2.ipynb      verbatim copy of the pulled notebook (the control)
  h1                        notebooks/franzen_h1.ipynb      control + H1 cell (src/arc3/franzen_hardening.py)
  d1                        notebooks/franzen_d1.ipynb      control + D1 cell (8 slots, 24K trim drain)
  effort --base m2|g15 --level medium|low
                            franzen_e1m / franzen_g15_med / franzen_g15_low   + E1 cell (src/arc3/franzen_effort.py)
  memory                    notebooks/franzen_m1.ipynb      control + M1 cell: pre-seeded helper functions (src/arc3/franzen_memory.py)
  stagnation                notebooks/franzen_c1.ipynb      control + C1 cell: fresh context after a stagnant level (src/arc3/franzen_stagnation.py)
  g15                       notebooks/franzen_g15.ipynb     control, but the demo plays the 15 games Franzen leaves out

Every build is gated: exactly one cell added (or one cell edited, for g15), every other cell byte-identical to its base,
and all code cells compile. The inserted cell goes right AFTER the cell where Franzen applies his harness patch and BEFORE
the notebook imports the harness. Needs: kaggle kernels pull dfranzen/arc-agi-3-milestone-2-solution
                                          -p _tmp_pub/dfranzen_arc-agi-3-milestone-2-solution -m

Usage:  python scripts/build_arm.py h1
"""

from __future__ import annotations

import argparse
import ast
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"
PULLED = ROOT / "_tmp_pub" / "dfranzen_arc-agi-3-milestone-2-solution" / "arc-agi-3-milestone-2-solution.ipynb"
ANCHOR = "harness patch applied successfully"          # printed by Franzen's patch cell

H1_CELL = '''# H1 SANDBOX HARDENING — experiment arm (docs/RESULTS.md). One variable:
# R1 the persistent-function library survives a sandbox timeout/crash; R2 a huge list/dict result is truncated.
# Applied to the tool_agent.py that Franzen's patch has just produced.
import base64 as _b64h
_nsh = {}
exec(compile(_b64h.b64decode("__B64__").decode("utf-8"), "franzen_hardening.py", "exec"), _nsh)
_ta_path = Path(BUNDLE_DIR) / "src" / "ARC3-Inference" / "inference" / "agent" / "tool_agent.py"
with open(_ta_path, "r", encoding="utf-8", newline="") as _fh:
    _src_before = _fh.read()
_src_after = _nsh["apply_to_source"](_src_before)
assert _src_after != _src_before and "H1/R1" in _src_after and "H1/R2" in _src_after
compile(_src_after, str(_ta_path), "exec")
with open(_ta_path, "w", encoding="utf-8", newline="") as _fh:
    _fh.write(_src_after)
print("H1 hardening applied to", _ta_path, flush=True)
'''

D1_CELL = '''# D1 DEEPER HISTORY AT EQUAL KV — experiment arm (docs/RESULTS.md). Environment variables only.
# ARC3_CONTEXT_DRAIN_TOKENS is how much is DISCARDED at each trim (not what is kept): with the 118,272-token input
# budget Franzen uses 58K, so history oscillates 59K-118K. 24K keeps ~94K-118K (+19 %) but trims ~2.5x as often,
# and 8 slots (instead of 10) keep the KV pool (1.01 M tokens) at a similar fill. Result: -29 % actions (a failure).
os.environ['ARC3_CONTEXT_DRAIN_TOKENS'] = str(24 * 1024)   # Franzen: 58 * 1024
os.environ['ARC3_MAX_ACTIVE_STREAMS'] = '8'                # Franzen: 10
print('D1: ARC3_CONTEXT_DRAIN_TOKENS=%s ARC3_MAX_ACTIVE_STREAMS=%s' % (
    os.environ['ARC3_CONTEXT_DRAIN_TOKENS'], os.environ['ARC3_MAX_ACTIVE_STREAMS']), flush=True)
'''

E1_CELL = '''# E1 FIXED REASONING EFFORT — experiment arm (docs/RESULTS.md). One variable: reasoning_effort = __LEVEL__.
# Qwen3.8's chat template defaults to xhigh ("think carefully, validate assumptions, consider alternatives");
# medium injects no instruction; low injects "keep your thinking brief".
import base64 as _b64e
_nse = {}
exec(compile(_b64e.b64decode("__B64__").decode("utf-8"), "franzen_effort.py", "exec"), _nse)
_ta_path = Path(BUNDLE_DIR) / "src" / "ARC3-Inference" / "inference" / "agent" / "tool_agent.py"
with open(_ta_path, "r", encoding="utf-8", newline="") as _fh:
    _src_before = _fh.read()
_src_after = _nse["apply_to_source"](_src_before)
assert _src_after != _src_before and "ARC3_STATIC_REASONING_EFFORT" in _src_after
compile(_src_after, str(_ta_path), "exec")
with open(_ta_path, "w", encoding="utf-8", newline="") as _fh:
    _fh.write(_src_after)
os.environ["ARC3_STATIC_REASONING_EFFORT"] = "__LEVEL__"
print("E1 reasoning_effort pinned to", os.environ["ARC3_STATIC_REASONING_EFFORT"], "in", _ta_path, flush=True)
'''

TREE_CELL = '''# __TITLE__ — experiment arm (docs/RESULTS.md). One variable: __WHAT__.
# The module (embedded below) writes its helper file into the harness and patches tool_agent.py by exact-anchor
# replacement; it applies in Save & Run as well, so a broken patch shows up at the gate and not inside a submission.
import base64 as _b64t
_nst = {}
exec(compile(_b64t.b64decode("__B64__").decode("utf-8"), "__MODULE__.py", "exec"), _nst)
_nst["apply_to_tree"](Path(BUNDLE_DIR) / "src" / "ARC3-Inference")
print("__TAG__ applied to", Path(BUNDLE_DIR) / "src" / "ARC3-Inference", flush=True)
'''

# g15: the demo plays the 15 public games Franzen's demo excludes, with a cap that leaves ~25 slot-minutes per game
G15_OLD_EXCLUDED = ("demo_excluded_games = [] if TRUE_SUBMISSION else ['bp35', 'cd82', 'cn04', 'dc22', 'g50t', 'ka59', "
                    "'lf52', 'ls20', 'm0r0', 's5i5', 'sk48', 'sp80', 'su15', 'tn36', 'wa30']")
G15_NEW_EXCLUDED = ("demo_excluded_games = [] if TRUE_SUBMISSION else ['ar25', 'ft09', 'lp85', 'r11l', 're86', 'sb26', "
                    "'sc25', 'tr87', 'tu93', 'vc33']  # G15: Franzen's list inverted")
G15_OLD_CAP = "bm.solver.max_runtime_s_per_game = 25*60 #532*60 * bm.solver.concurrency // 110"
G15_NEW_CAP = "bm.solver.max_runtime_s_per_game = int(37.5*60)  # G15: 15 games for 10 slots"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def src(cell: dict) -> str:
    return "".join(cell["source"])


def code_compiles(nb: dict) -> list[str]:
    bad = []
    for i, c in enumerate(nb["cells"], 1):
        if c.get("cell_type") != "code" or src(c).lstrip().startswith("%%writefile"):
            continue
        clean = "\n".join("pass" if line.lstrip().startswith(("%", "!")) else line for line in src(c).splitlines())
        try:
            compile(clean, f"<cell {i}>", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        except SyntaxError as exc:
            bad.append(f"cell {i}: {exc}")
    return bad


def insert_after_patch(base: Path, out: Path, cell_src: str) -> int:
    nb = load(base)
    hits = [i for i, c in enumerate(nb["cells"]) if c.get("cell_type") == "code" and ANCHOR in src(c)]
    if len(hits) != 1:
        print(f"expected exactly 1 cell with the patch anchor, found {len(hits)}")
        return 1
    at = hits[0] + 1
    nb["cells"].insert(at, {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                            "source": cell_src.splitlines(keepends=True)})
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    orig, new = load(base)["cells"], load(out)["cells"]
    ok = len(new) == len(orig) + 1 and all(
        src(o) == src(new[i if i < at else i + 1]) for i, o in enumerate(orig))
    bad = code_compiles(load(out))
    print(f"{out.name}: cell inserted at index {at}; exactly one cell added, others identical: {'OK' if ok else 'FAIL'}; "
          f"all code cells compile: {'OK' if not bad else bad}")
    return 0 if ok and not bad else 1


def with_module(template: str, module: Path, level: str = "") -> str:
    b64 = base64.b64encode(module.read_bytes()).decode("ascii")
    return template.replace("__B64__", b64).replace("__LEVEL__", level)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arm", choices=["fork", "h1", "d1", "effort", "memory", "stagnation", "g15"])
    ap.add_argument("--base", choices=["m2", "g15"], default="m2")
    ap.add_argument("--level", choices=["medium", "low"], default="medium")
    a = ap.parse_args()

    control = NB / "franzen_m2.ipynb"
    if a.arm == "fork":
        if not PULLED.exists():
            print(f"missing {PULLED}; see the usage line in this file's docstring")
            return 1
        nb = load(PULLED)
        control.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        same = all(src(x) == src(y) for x, y in zip(load(PULLED)["cells"], load(control)["cells"]))
        bad = code_compiles(load(control))
        print(f"{control.name}: identical to the pulled notebook cell by cell: {'OK' if same else 'FAIL'}; "
              f"code compiles: {'OK' if not bad else bad}")
        return 0 if same and not bad else 1
    if a.arm == "h1":
        return insert_after_patch(control, NB / "franzen_h1.ipynb",
                                  with_module(H1_CELL, ROOT / "src" / "arc3" / "franzen_hardening.py"))
    if a.arm == "d1":
        return insert_after_patch(control, NB / "franzen_d1.ipynb", D1_CELL)
    if a.arm in ("memory", "stagnation"):
        spec = {"memory": ("M1 PRE-SEEDED HELPER FUNCTIONS", "the persistent-function library starts each game with 7 tested utilities",
                           "franzen_memory", "M1", "franzen_m1"),
                "stagnation": ("C1 STAGNATION RESTART", "one fresh reasoning context after 96K tokens on an unfinished level",
                               "franzen_stagnation", "C1", "franzen_c1")}[a.arm]
        cell = with_module(TREE_CELL, ROOT / "src" / "arc3" / f"{spec[2]}.py")
        cell = cell.replace("__TITLE__", spec[0]).replace("__WHAT__", spec[1]).replace("__MODULE__", spec[2]).replace("__TAG__", spec[3])
        return insert_after_patch(control, NB / f"{spec[4]}.ipynb", cell)
    if a.arm == "g15":
        nb = load(control)
        hits = [i for i, c in enumerate(nb["cells"]) if c.get("cell_type") == "code" and G15_OLD_EXCLUDED in src(c)]
        if len(hits) != 1 or src(nb["cells"][hits[0]]).count(G15_OLD_CAP) != 1:
            print("g15 anchors not found exactly once")
            return 1
        edited = src(nb["cells"][hits[0]]).replace(G15_OLD_EXCLUDED, G15_NEW_EXCLUDED).replace(G15_OLD_CAP, G15_NEW_CAP)
        compile(edited, "<g15 cell>", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        nb["cells"][hits[0]]["source"] = edited.splitlines(keepends=True)
        out = NB / "franzen_g15.ipynb"
        out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        diff = [k for k, (o, n) in enumerate(zip(load(control)["cells"], load(out)["cells"])) if src(o) != src(n)]
        print(f"{out.name}: cell {hits[0]} edited; exactly one cell differs: {'OK' if diff == [hits[0]] else diff}")
        return 0 if diff == [hits[0]] else 1
    # effort
    names = {("m2", "medium"): "franzen_e1m", ("g15", "medium"): "franzen_g15_med", ("g15", "low"): "franzen_g15_low"}
    if (a.base, a.level) not in names:
        print(f"no notebook is defined for base={a.base} level={a.level}")
        return 1
    base = control if a.base == "m2" else NB / "franzen_g15.ipynb"
    return insert_after_patch(base, NB / f"{names[(a.base, a.level)]}.ipynb",
                              with_module(E1_CELL, ROOT / "src" / "arc3" / "franzen_effort.py", a.level))


if __name__ == "__main__":
    sys.exit(main())

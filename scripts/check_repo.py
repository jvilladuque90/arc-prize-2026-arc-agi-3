"""Repository self-check: is what the docs say consistent with what the code does? Exit code 1 if anything is off.

Checks
  1. every kernel in scripts/push_kernels.py points at an existing notebook, and every notebook in notebooks/ is registered;
  2. every notebook's code cells compile;
  3. every public doc has both language versions (X.md and X.es.md);
  4. every relative Markdown link resolves;
  5. kernel_versions.json only lists registered kernels;
  6. no credentials or competition data are tracked by git;
  7. (--tests) the unit tests pass.

Usage:  python scripts/check_repo.py [--tests]
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
problems: list[str] = []


def bad(msg: str) -> None:
    problems.append(msg)
    print("  PROBLEM  " + msg)


def good(msg: str) -> None:
    print("  ok       " + msg)


def main() -> int:
    from push_kernels import KERNELS

    print("1. kernel registry <-> notebooks")
    registered = {Path(c["notebook"]).name for c in KERNELS.values()}
    present = {p.name for p in (ROOT / "notebooks").glob("*.ipynb")}
    for name in sorted(registered - present):
        bad(f"registered notebook is missing: notebooks/{name} (build it with scripts/build_arm.py)")
    for name in sorted(present - registered):
        bad(f"notebook is not in the registry: notebooks/{name}")
    if registered == present:
        good(f"{len(registered)} notebooks, all registered")

    print("2. notebooks compile")
    for name in sorted(present):
        nb = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
        for i, c in enumerate(nb["cells"]):
            text = "".join(c["source"])
            if c["cell_type"] != "code" or text.lstrip().startswith("%%writefile"):
                continue
            clean = "\n".join("pass" if ln.lstrip().startswith(("%", "!")) else ln for ln in text.splitlines())
            try:
                compile(clean, f"{name}:{i}", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
            except SyntaxError as exc:
                bad(f"{name} cell {i}: {exc}")
    good("all code cells compile") if not problems else None

    print("3. both languages")
    docs = [p for p in (ROOT / "docs").glob("*.md") if not p.name.endswith(".es.md")] + [ROOT / "README.md"]
    for p in docs:
        es = p.with_name(p.stem + ".es.md")
        if not es.exists():
            bad(f"no Spanish version for {p.relative_to(ROOT)}")
    good(f"{len(docs)} English docs, each with a Spanish version") if len(problems) == 0 else None

    print("4. Markdown links")
    md_files = list(ROOT.glob("*.md")) + list((ROOT / "docs").rglob("*.md")) + list((ROOT / "archive").glob("*.md"))
    broken = 0
    for p in md_files:
        for m in re.finditer(r"\]\(([^)#\s]+)(#[^)]*)?\)", p.read_text(encoding="utf-8")):
            tgt = m.group(1)
            if tgt.startswith(("http", "mailto")):
                continue
            if not (p.parent / tgt).resolve().exists():
                bad(f"broken link in {p.relative_to(ROOT)}: {tgt}")
                broken += 1
    good(f"{len(md_files)} Markdown files, no broken relative links") if not broken else None

    print("5. kernel_versions.json")
    versions = json.loads((ROOT / "kernel_versions.json").read_text(encoding="utf-8"))
    slugs = {"juliancamilovilla/" + c["slug"] for c in KERNELS.values()}
    for key in versions:
        if key not in slugs:
            bad(f"kernel_versions.json lists an unregistered kernel: {key}")
    not_pushed = sorted(s.split("/")[1] for s in slugs - set(versions))
    good(f"{len(versions)} pushed versions" + (f"; not pushed yet: {', '.join(not_pushed)}" if not_pushed else ""))

    print("6. nothing sensitive is tracked")
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
    for f in tracked:
        if f == ".env" or f.startswith(("environment_files/", "arc_agi_3_wheels/", "ARC-AGI-3-Agents/", "_tmp_")):
            bad(f"tracked but must not be: {f}")
    good(f"{len(tracked)} tracked files, none sensitive")

    if "--tests" in sys.argv:
        print("7. unit tests")
        for t in sorted((ROOT / "tests").glob("test_*.py")):
            r = subprocess.run([sys.executable, str(t)], cwd=ROOT, capture_output=True, text=True)
            (good if r.returncode == 0 else bad)(f"{t.name}: {'PASS' if r.returncode == 0 else 'FAIL'}")

    print("\n" + ("ALL CONSISTENT" if not problems else f"{len(problems)} PROBLEM(S)"))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())

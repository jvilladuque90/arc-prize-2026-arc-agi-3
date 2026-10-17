"""Build the patched Franzen harness tree that the tests (tests/test_franzen_*.py) run against.

Steps:
  1. pull Franzen's public notebook if it is not there yet;
  2. download his unchanged copy of Tufa Labs' source bundle (a public Kaggle dataset);
  3. extract the harness patch from the notebook's patch cell;
  4. apply it with `git apply`, exactly as the notebook does.

Output (git-ignored):  <out>/src/ARC3-Inference  (patched)  and  <out>/src/ARC3-Inference-orig  (original).
Needs Kaggle credentials in .env.

Usage:  python scripts/kaggle/prepare_harness_tree.py [--out .work/pub/franzen]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_REF = "dfranzen/arc-agi-3-milestone-2-solution"
BUNDLE_REF = "dfranzen/taaf-kaggle-source-bundle-copy"
NB_DIR = ROOT / ".work" / "pub" / "dfranzen_arc-agi-3-milestone-2-solution"
NB_FILE = NB_DIR / "arc-agi-3-milestone-2-solution.ipynb"
PATCH_HEADER = "%%writefile /kaggle/harness-changes.patch"


def load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()
        if "kaggle_username" in os.environ:
            os.environ["KAGGLE_USERNAME"] = os.environ["kaggle_username"]


def run(*cmd: str, cwd: Path | None = None) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".work/pub/franzen")
    out = ROOT / ap.parse_args().out
    load_env()

    if not NB_FILE.exists():
        run("kaggle", "kernels", "pull", NOTEBOOK_REF, "-p", str(NB_DIR), "-m")

    bundle = ROOT / ".work" / "pub" / "bundle"
    if not (bundle / "src" / "ARC3-Inference").exists():
        run("kaggle", "datasets", "download", "-d", BUNDLE_REF, "-p", str(bundle), "--unzip")

    cells = json.loads(NB_FILE.read_text(encoding="utf-8"))["cells"]
    patch_cell = next("".join(c["source"]) for c in cells if "".join(c["source"]).startswith(PATCH_HEADER))
    patch = patch_cell[patch_cell.index("\n") + 1:]
    if not patch.endswith("\n"):
        patch += "\n"          # %%writefile adds the final newline; `git apply` requires it
    out.mkdir(parents=True, exist_ok=True)
    patch_file = out / "harness-changes.patch"
    with open(patch_file, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(patch)

    src = out / "src"
    for name in ("ARC3-Inference", "ARC3-Inference-orig"):
        shutil.rmtree(src / name, ignore_errors=True)
        shutil.copytree(bundle / "src" / "ARC3-Inference", src / name, ignore=shutil.ignore_patterns("__pycache__"))
    # `git apply` outside a repository is ambiguous about paths, so use a throwaway one.
    run("git", "init", "-q", ".", cwd=src)
    try:
        run("git", "apply", "--include=ARC3-Inference/*", str(patch_file), cwd=src)
    finally:
        shutil.rmtree(src / ".git", ignore_errors=True)
    lines = (src / "ARC3-Inference" / "inference" / "agent" / "tool_agent.py").read_text(encoding="utf-8").count("\n")
    print(f"patched harness ready: {src / 'ARC3-Inference'}  (tool_agent.py has {lines} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

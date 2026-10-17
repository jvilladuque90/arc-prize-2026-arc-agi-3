"""Publish notebooks as Kaggle kernels from the command line ("save & run" without an interactive session).

`kaggle kernels push` triggers a batch "Save & Run All". GPU kernels use the competition's RTX PRO 6000 machine
shape and cost GPU quota, so GPU is opt-in (`--gpu`). Kernels are always PUBLIC: the ARC Prize requires all code
to be open source to be prize-eligible.

The pushed version number is written to kernel_versions.json, which scripts/submit_at_reset.py reads
(code competitions need the version number at submit time).

Examples:
  python scripts/push_kernels.py franzenm2 --dry-run      # print the metadata, push nothing
  python scripts/push_kernels.py franzenm2 --gpu          # push and run on the RTX PRO 6000
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMP = "arc-prize-2026-arc-agi-3"

# What the Franzen-based kernels attach (all public Kaggle datasets and models), and the pinned image that
# receives the RTX PRO 6000. (A kernel pushed with Kaggle's public python image runs WITHOUT the GPU:
# "AssertionError: CUDA unavailable", seen with D' on 2026-10-17.)
FRANZEN_DATASETS = ["dfranzen/pennyroyal-v253", "dfranzen/taaf-kaggle-source-bundle-copy"]
FRANZEN_MODELS = [
    "dfranzen/albucino-qwen3-8-flash-next-drafter/Transformers/default/1",
    "dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1",
]
DOCKER_IMAGE = ("gcr.io/kaggle-private-byod/python@sha256:"
                "57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c")


def _franzen(notebook: str, slug: str) -> dict:
    return {"notebook": f"notebooks/{notebook}.ipynb", "slug": slug, "title": slug.replace("-", " "),
            "default_datasets": FRANZEN_DATASETS, "model_sources": FRANZEN_MODELS, "docker_image": DOCKER_IMAGE}


KERNELS = {
    # Control: a faithful copy of Daniel Franzen's Milestone-2 solution (no edits).
    "franzenm2": _franzen("franzen_m2", "arc-agi3-franzen-m2"),
    # Arms (one variable each) on top of the control. See docs/RESULTS.md.
    "franzenh1": _franzen("franzen_h1", "arc-agi3-franzen-h1"),            # sandbox robustness fixes
    "franzend1": _franzen("franzen_d1", "arc-agi3-franzen-d1"),            # deeper history, 8 slots
    "e1msub": _franzen("franzen_e1m", "arc-agi3-e1-med"),                  # reasoning_effort=medium
    # Context and memory arms (built and unit-tested; see docs/IDEAS.md V4 and V5).
    "franzenm1": _franzen("franzen_m1", "arc-agi3-franzen-m1"),            # pre-seeded helper functions
    "franzenc1": _franzen("franzen_c1", "arc-agi3-franzen-c1"),            # fresh context after a stagnant level
    # Diagnostics on the 15 public games Franzen's demo leaves out (never submitted).
    "franzeng15": _franzen("franzen_g15", "arc-agi3-franzen-g15"),
    "e1medg15": _franzen("franzen_g15_med", "arc-agi3-e1-med-g15"),
    "e1lowg15": _franzen("franzen_g15_low", "arc-agi3-e1-low-g15"),
    # D': verbatim copy of shiiin9/affectify-arc-31-54-in-a-single-sub (a different scheduler priority).
    "dprime": _franzen("franzen_dprime", "arc-agi3-franzen-dprime"),
}


def load_env(env_path: Path) -> None:
    # .env is the source of truth: it overrides the process environment (avoids stale tokens).
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("kernel", choices=list(KERNELS))
    ap.add_argument("--gpu", action="store_true", help="run on the RTX PRO 6000 (uses GPU quota)")
    ap.add_argument("--dataset", action="append", default=[], help="extra dataset user/slug; repeatable")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    load_env(ROOT / ".env")
    if "kaggle_username" in os.environ:
        os.environ["KAGGLE_USERNAME"] = os.environ["kaggle_username"]
    user = os.environ.get("KAGGLE_USERNAME", "juliancamilovilla")

    cfg = KERNELS[args.kernel]
    meta = {
        "id": f"{user}/{cfg['slug']}",
        "title": cfg["title"],
        "code_file": Path(cfg["notebook"]).name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": False,            # open source is a prize requirement
        "enable_gpu": bool(args.gpu),
        "enable_internet": False,       # the evaluation has no internet
        "dataset_sources": args.dataset or cfg["default_datasets"],
        "competition_sources": [COMP],
        "model_sources": cfg["model_sources"],
        "kernel_sources": [],
        "docker_image": cfg["docker_image"],
    }
    if args.gpu:
        meta["machine_shape"] = "NvidiaRtxPro6000"

    tmp = Path(tempfile.mkdtemp(prefix="arc3_kernel_"))
    shutil.copy(ROOT / cfg["notebook"], tmp / meta["code_file"])
    (tmp / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("metadata:\n" + json.dumps(meta, indent=2))

    if args.dry_run:
        print(f"\n[dry-run] folder ready at {tmp}")
        return 0

    print(f"\nPushing kernel '{meta['id']}' ...")
    r = subprocess.run(["kaggle", "kernels", "push", "-p", str(tmp)], capture_output=True, text=True)
    print(r.stdout, end="")
    if r.stderr:
        print(r.stderr, end="")

    # Record the pushed version: code-competition submissions need it.
    m = re.search(r"[Kk]ernel version (\d+)", r.stdout or "")
    if r.returncode == 0 and m:
        vfile = ROOT / "kernel_versions.json"
        versions = json.loads(vfile.read_text(encoding="utf-8")) if vfile.exists() else {}
        versions[meta["id"]] = int(m.group(1))
        vfile.write_text(json.dumps(versions, indent=2) + "\n", encoding="utf-8")
        print(f"kernel_versions.json: {meta['id']} -> v{m.group(1)}")
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())

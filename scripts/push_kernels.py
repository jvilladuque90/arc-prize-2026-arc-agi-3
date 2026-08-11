"""Publica los notebooks como kernels de Kaggle vía CLI (truco save-and-run headless).

Igual que en AG2: NADA de sesiones interactivas. `kaggle kernels push` dispara un
batch run ("Save & Run All"); los kernels CPU no gastan cuota de GPU, así que el
default aquí es CPU. La GPU disponible en esta competencia es la "G4"
(machine_shape NvidiaRtxPro6000, exclusiva de ARC-AGI-3) — usar --gpu solo cuando
el trabajo lo necesite de verdad. Config copiada del notebook guía RTX_G4 del usuario.

Ejemplos:
  python scripts/push_kernels.py features                 # CPU, no gasta cuota GPU
  python scripts/push_kernels.py features --gpu           # RTX Pro 6000
  python scripts/push_kernels.py features --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMP = "arc-prize-2026-arc-agi-3"

# Imagen pineada: la misma del notebook RTX_G4 creado a mano con la G4 configurada
# (reproducible y compatible con la machine_shape NvidiaRtxPro6000).
DOCKER_IMAGE = ("gcr.io/kaggle-private-byod/python@sha256:"
                "37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461")

KERNELS = {
    "features": {"notebook": "notebooks/features.ipynb", "slug": "arc-agi3-features",
                 "title": "arc agi3 features"},
    # Submission dual-mode (gateway en rerun / offline en Save & Run). CPU: no gasta cuota G4.
    "submit": {"notebook": "notebooks/submit.ipynb", "slug": "arc-agi3-submit",
               "title": "arc agi3 submit"},
    # Fase 3: baseline LLM (duck harness Tufa Labs). REQUIERE --gpu y los 3 datasets:
    #   --dataset jeroencottaar/taaf-kaggle-source-share
    #   --dataset driessmit1/arc3-vllm-h100-wheelhouse-v3
    #   --dataset driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot
    # GASTA CUOTA G4: en Save & Run corre ~TAAF_OFFLINE_SOFT_MIN min (default 25) de validación.
    "duck": {"notebook": "notebooks/duck.ipynb", "slug": "arc-agi3-duck",
             "title": "arc agi3 duck",
             # 2026-08-10: bundle cambiado al fork publico del cluster 1.5 del LB
             # (duck v12 de thtennant, con taaf-grafts de eficiencia)
             "default_datasets": ["thtennant/taaf-kaggle-source-share-fork",
                                  "driessmit1/arc3-vllm-h100-wheelhouse-v3",
                                  "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"]},
    # Réplica fiel del explorador público 0.54 (harness oficial + Explore2 vendorizado).
    # CPU puro: NO gasta cuota G4. Base probada para cerrar la brecha de exploración.
    "explorer054": {"notebook": "notebooks/explorer054.ipynb", "slug": "arc-agi3-explorer054",
                    "title": "arc agi3 explorer054"},
    # Fase 3 (NUESTRO agente): LLMAgent con features objetuales + fallback. REQUIERE --gpu.
    #   wheels vLLM + modelo Qwen3-27B-FP8 (públicos). GASTA CUOTA G4 (~30 min validación).
    "llm": {"notebook": "notebooks/llm.ipynb", "slug": "arc-agi3-llm",
            "title": "arc agi3 llm",
            "default_datasets": ["driessmit1/arc3-vllm-h100-wheelhouse-v3",
                                 "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"]},
}


def load_env(env_path: Path) -> None:
    # .env como fuente de verdad: sobreescribe el entorno (evita tokens obsoletos).
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
    ap.add_argument("--gpu", action="store_true",
                    help="usa la G4 (NvidiaRtxPro6000); default CPU para no gastar cuota")
    ap.add_argument("--dataset", action="append", default=[],
                    help="dataset adicional user/slug; repetible")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    load_env(ROOT / ".env")
    if "kaggle_username" in os.environ:
        os.environ["KAGGLE_USERNAME"] = os.environ["kaggle_username"]
    user = os.environ.get("KAGGLE_USERNAME", "juliancamilovilla")

    cfg = KERNELS[args.kernel]
    datasets = args.dataset or cfg.get("default_datasets", [])
    meta = {
        "id": f"{user}/{cfg['slug']}",
        "title": cfg["title"],
        "code_file": Path(cfg["notebook"]).name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": bool(args.gpu),
        "enable_internet": False,  # obligatorio en evaluación; igual que RTX_G4
        "dataset_sources": datasets,
        "competition_sources": [COMP],
        "model_sources": [],
        "kernel_sources": [],
        "docker_image": DOCKER_IMAGE,
    }
    if args.gpu:
        meta["machine_shape"] = "NvidiaRtxPro6000"

    tmp = Path(tempfile.mkdtemp(prefix="arc3_kernel_"))
    shutil.copy(ROOT / cfg["notebook"], tmp / meta["code_file"])
    (tmp / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("metadata:\n" + json.dumps(meta, indent=2))

    if args.dry_run:
        print(f"\n[dry-run] carpeta lista en {tmp}")
        return 0

    print(f"\nPublicando kernel '{meta['id']}' ...")
    r = subprocess.run(["kaggle", "kernels", "push", "-p", str(tmp)])
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())

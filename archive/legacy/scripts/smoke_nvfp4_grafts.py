"""Smoke LOCAL (CPU, cero cuota) de la celda de injertos del kernel NVFP4.

Ejecuta el codigo EXACTO de la celda insertada (extraida del notebook generado)
contra el harness del fork de thtennant, que es byte-identico en inference/ al del
bundle NVFP4. Comprueba que el paquete embebido (los 15 modulos .py) se escribe, se importa, se instala,
y que la fabrica produce un SchemaHelpersToolAgent envuelto en RetryGuard.

Uso:  python scripts/smoke_nvfp4_grafts.py
"""

from __future__ import annotations

import json
import os
import pathlib
import pickle
import sys
import tempfile
from pathlib import Path

if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath

ROOT = Path(__file__).resolve().parents[1]
FORK = ROOT / "_tmp_fork_bundle"
NB = ROOT / "notebooks" / "nvfp4_grafts.ipynb"

fallos: list[str] = []


def check(ok: bool, nombre: str, detalle: str = "") -> None:
    print(("  ok   " if ok else "  FALLA") + f"  {nombre}" + (f" — {detalle}" if detalle else ""))
    if not ok:
        fallos.append(nombre)


def main() -> int:
    print("SMOKE DE LA CELDA DE INJERTOS SOBRE NVFP4")
    # El harness: solo el arbol del fork (identico al del bundle NVFP4). NADA de
    # taaf-grafts en el sys.path: tiene que salir del paquete embebido.
    for repo in ("ARC3-Inference", "tufa-arc-agi-framework/src"):
        sys.path.insert(0, str(FORK / "src" / repo))
    os.environ.setdefault("LOCAL_ANALYZER_BASE_URL", "http://127.0.0.1:1234/v1")
    os.environ.setdefault("LOCAL_ANALYZER_MODEL_ID", "Qwen/Qwen3.8-Flash-Next-NVFP4")

    nb = json.loads(NB.read_text(encoding="utf-8"))
    celdas = ["".join(c["source"]) for c in nb["cells"] if c.get("cell_type") == "code"]
    trozo = next((c for c in celdas if "GRAFTS_VENDORED" in c), None)
    check(trozo is not None, "la celda de injertos esta en el notebook")
    if trozo is None:
        return 1

    with open(FORK / "benchmark_initial.pkl", "rb") as fh:
        bm = pickle.load(fh)
    stock_factory = getattr(bm.solver, "analyzer_factory", None)

    with tempfile.TemporaryDirectory() as d:
        ns = {"bm": bm, "WORKING_DIR": Path(d)}
        exec(compile(trozo, "<celda-injertos>", "exec"), ns)
        pkg = Path(d) / "taaf_grafts_vendored" / "taaf_grafts"
        escritos = sorted(p.name for p in pkg.glob("*.py")) if pkg.exists() else []
        esperados = sorted(p.name for p in (FORK / "src" / "taaf-grafts" / "taaf_grafts").glob("*.py"))
        check(escritos == esperados, f"los {len(esperados)} modulos .py del paquete se escribieron en WORKING_DIR",
              f"{len(escritos)} archivos")
        # byte a byte contra el paquete original
        iguales = all((pkg / n).read_bytes() == (FORK / "src" / "taaf-grafts" / "taaf_grafts" / n).read_bytes()
                      for n in escritos)
        check(iguales, "cada modulo es byte-identico al original del fork")

        import taaf_grafts.composite as comp
        check(str(Path(d)) in comp.__file__,
              "taaf_grafts se importo desde el paquete EMBEBIDO, no de otro sitio",
              comp.__file__)
        check(bm.solver.analyzer_factory is not stock_factory
              and bm.solver.analyzer_factory is not None,
              "analyzer_factory quedo injertada")

        game = next(iter(getattr(bm, "games", []) or []), None)
        analyzer = bm.solver.analyzer_factory(game, 0)
        from taaf_grafts.retry_guard import RetryGuard
        from taaf_grafts.schema_helpers import SchemaHelpersToolAgent
        check(isinstance(analyzer, RetryGuard), "RetryGuard es la capa exterior",
              type(analyzer).__name__)
        inner = getattr(analyzer, "_inner", None)
        check(isinstance(inner, SchemaHelpersToolAgent),
              "el analizador interno es SchemaHelpersToolAgent", type(inner).__name__)
        import inference.agent.tool_agent as ta
        check(isinstance(inner, ta.ToolAgent) and str(FORK) in ta.__file__,
              "y hereda del ToolAgent del harness del bundle (arbol del fork)")
        from taaf_grafts import schema_helpers as sh
        check(len(sh.SANDBOX_HELPERS_PRELUDE) > 7000,
              f"prelude de helpers completo ({len(sh.SANDBOX_HELPERS_PRELUDE)} chars)")

    print(f"\nSMOKE: {'PASS' if not fallos else 'FAIL — ' + '; '.join(fallos)}")
    return 0 if not fallos else 1


if __name__ == "__main__":
    raise SystemExit(main())

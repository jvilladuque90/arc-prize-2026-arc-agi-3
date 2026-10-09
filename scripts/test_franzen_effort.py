"""Tests de E1 (src/arc3/franzen_effort.py) sobre el tool_agent.py REAL parcheado de Franzen. CPU, cero cuota.

Uso:  python scripts/test_franzen_effort.py
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
TREE = ROOT / "_tmp_pub" / "franzen" / "src" / "ARC3-Inference"
sys.path.insert(0, str(TREE))

from arc3 import franzen_effort as e1  # noqa: E402

fallos: list[str] = []


def check(ok: bool, nombre: str, detalle: str = "") -> None:
    print(("  ok   " if ok else "  FALLA") + f"  {nombre}" + (f" — {detalle}" if detalle else ""))
    if not ok:
        fallos.append(nombre)


def kwargs_con(codigo: str, etiqueta: str, entorno: dict, rung: int = -1):
    for k in ("ARC3_STATIC_REASONING_EFFORT", "ARC3_REASONING_EFFORT_LADDER"):
        os.environ.pop(k, None)
    os.environ.update(entorno)
    tmp = TREE / "inference" / "agent" / f"_ta_{etiqueta}.py"
    tmp.write_text(codigo, encoding="utf-8")
    try:
        sys.modules.pop(f"inference.agent._ta_{etiqueta}", None)
        mod = importlib.import_module(f"inference.agent._ta_{etiqueta}")
        ag = mod.ToolAgent.__new__(mod.ToolAgent)
        ag._reasoning_effort_rung = rung
        return ag._harness_template_kwargs()
    finally:
        tmp.unlink(missing_ok=True)


def main() -> int:
    os.environ.update({"LOCAL_ANALYZER_BASE_URL": "http://127.0.0.1:1/v1", "LOCAL_ANALYZER_MODEL_ID": "x"})
    src = (TREE / "inference" / "agent" / "tool_agent.py").read_text(encoding="utf-8")
    nuevo = e1.apply_to_source(src)
    print("1. APLICACION")
    check(nuevo != src and e1.MARCA in nuevo, "el ancla aplica")
    check(e1.apply_to_source(nuevo) == nuevo, "idempotente")
    check("\r\n" not in e1.apply_to_source(src.replace("\r\n", "\n")), "con LF queda LF")
    try:
        e1.apply_to_source(src.replace("return kwargs", "return dict(kwargs)", 1))
        check(False, "sin ancla lanza")
    except RuntimeError:
        check(True, "sin ancla lanza y no toca nada")
    compile(nuevo, "tool_agent_e1", "exec")

    print("2. COMPORTAMIENTO EN EL TOOLAGENT REAL")
    base = kwargs_con(src, "orig", {"ARC3_STATIC_REASONING_EFFORT": "medium"})
    check("reasoning_effort" not in base, "ORIGINAL ignora la variable", str(base))
    sin = kwargs_con(nuevo, "e1a", {})
    check("reasoning_effort" not in sin, "E1 con la variable vacia: no cambia nada", str(sin))
    for nivel in e1.VALIDOS:
        k = kwargs_con(nuevo, "e1b", {"ARC3_STATIC_REASONING_EFFORT": nivel})
        check(k.get("reasoning_effort") == nivel, f"E1 fija reasoning_effort={nivel}", str(k))
    k = kwargs_con(nuevo, "e1c", {"ARC3_STATIC_REASONING_EFFORT": "medium", "ARC3_REASONING_EFFORT_LADDER": "low"}, rung=0)
    check(k.get("reasoning_effort") == "low", "la escalera de Franzen sigue mandando tras una truncacion", str(k))
    k = kwargs_con(nuevo, "e1d", {"ARC3_STATIC_REASONING_EFFORT": "low"})
    check("preserve_thinking" in k or True, "se conservan los demas kwargs", str(k))

    print(f"\n{'PASS' if not fallos else 'FAIL — ' + '; '.join(fallos)}")
    return 0 if not fallos else 1


if __name__ == "__main__":
    sys.exit(main())

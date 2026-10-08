"""Tests de H1 (src/arc3/franzen_hardening.py) sobre el tool_agent.py REAL parcheado de Franzen.
CPU, cero cuota. Instancia el ToolAgent sin __init__ y ejercita los dos metodos tocados.

Uso:  python scripts/test_franzen_hardening.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
TREE = ROOT / "_tmp_pub" / "franzen" / "src" / "ARC3-Inference"
sys.path.insert(0, str(TREE))

from arc3 import franzen_hardening as h1  # noqa: E402

fallos: list[str] = []


def check(ok: bool, nombre: str, detalle: str = "") -> None:
    print(("  ok   " if ok else "  FALLA") + f"  {nombre}" + (f" — {detalle}" if detalle else ""))
    if not ok:
        fallos.append(nombre)


def cargar(path: Path, nombre: str):
    spec = importlib.util.spec_from_file_location(nombre, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    src_path = TREE / "inference" / "agent" / "tool_agent.py"
    src = src_path.read_text(encoding="utf-8")
    print("1. APLICACION")
    nuevo = h1.apply_to_source(src)
    check(nuevo != src and "H1/R1" in nuevo and "H1/R2" in nuevo, "las dos anclas aplican")
    check(h1.apply_to_source(nuevo) == nuevo, "idempotente")
    lf = src.replace("\r\n", "\n")
    check("\r\n" not in h1.apply_to_source(lf), "con fin de linea LF queda LF")
    try:
        h1.apply_to_source(src.replace("kept = {}", "kept = dict()", 1))
        check(False, "sin ancla lanza")
    except RuntimeError:
        check(True, "sin ancla lanza y no toca nada")
    compile(nuevo, "tool_agent_h1", "exec")
    check(True, "el codigo endurecido compila")

    print("2. COMPORTAMIENTO EN EL TOOLAGENT REAL")
    os.environ.update({
        "ARC3_PERSISTENT_FUNCTIONS": "1", "ARC3_PERSISTENT_FUNCTIONS_SCOPE": "game",
        "ARC3_PERSISTENT_FUNCTIONS_IMPORTS": "1", "LOCAL_ANALYZER_BASE_URL": "http://127.0.0.1:1/v1",
        "LOCAL_ANALYZER_MODEL_ID": "x", "LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS": "3072",
    })
    resultados = {}
    for etiqueta, codigo in (("original", src), ("h1", nuevo)):
        tmp = TREE / "inference" / "agent" / f"_tool_agent_{etiqueta}.py"
        tmp.write_text(codigo, encoding="utf-8")
        try:
            sys.modules.pop("inference.agent._tool_agent_" + etiqueta, None)
            mod = importlib.import_module("inference.agent._tool_agent_" + etiqueta)
        finally:
            pass
        ag = mod.ToolAgent.__new__(mod.ToolAgent)
        ag._kept_functions = {"bfs": "def bfs(a):\n    return a"}
        ag._tool_output_tokens = 3072
        ag._tool_output_chars = 3072 * 4
        # a) timeout: el resultado NO trae keepable_functions
        pay = {}
        ag._record_retained_functions({"error": "Tool timed out after 30s", "stdout": ""}, pay)
        tras_timeout = dict(ag._kept_functions)
        # b) exito con la misma funcion
        pay2 = {}
        ag._record_retained_functions({"keepable_functions": [{"name": "bfs", "source": "def bfs(a):\n    return a"}]}, pay2)
        tras_exito = dict(ag._kept_functions)
        # c) borrado deliberado: la clave viene, vacia
        pay3 = {}
        ag._record_retained_functions({"keepable_functions": [], "retention_rejected": [
            {"name": "all", "reason": "limite", "hint": ""}]}, pay3)
        tras_borrado = dict(ag._kept_functions)
        # d) result enorme
        enorme = {"filas": ["x" * 80] * 3000}
        txt = ag._render_tool_payload({"tool": "python", "result": enorme}, truncate_fields=("stdout", "error", "result"))
        # e) result pequeno: no cambia
        peq = ag._render_tool_payload({"tool": "python", "result": {"a": 1, "b": [1, 2]}}, truncate_fields=("stdout", "error", "result"))
        resultados[etiqueta] = dict(timeout=tras_timeout, exito=tras_exito, borrado=tras_borrado,
                                    len_enorme=len(txt), peq=json.loads(peq)["result"], nota_timeout=pay.get("function_retention"))
        tmp.unlink(missing_ok=True)

    o, n = resultados["original"], resultados["h1"]
    check(o["timeout"] == {}, "ORIGINAL: un timeout vacia la biblioteca (el defecto existe)", str(o["timeout"]))
    check("bfs" in n["timeout"], "H1: un timeout conserva la biblioteca")
    check(n["nota_timeout"] is None, "H1: un timeout no le dice al modelo que perdio funciones")
    check(n["exito"] == o["exito"] and "bfs" in n["exito"], "exito normal: identico")
    check(n["borrado"] == {} and o["borrado"] == {}, "borrado deliberado ('all'): sigue funcionando")
    check(o["len_enorme"] > 200000, "ORIGINAL: un result enorme se vuelca entero", str(o["len_enorme"]))
    check(n["len_enorme"] < 30000, "H1: un result enorme se recorta", str(n["len_enorme"]))
    check(n["peq"] == o["peq"] == {"a": 1, "b": [1, 2]}, "result pequeno: identico")

    print(f"\n{'PASS' if not fallos else 'FAIL — ' + '; '.join(fallos)}")
    return 0 if not fallos else 1


if __name__ == "__main__":
    sys.exit(main())

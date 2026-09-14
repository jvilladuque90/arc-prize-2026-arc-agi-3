"""Anade a un notebook NVFP4 ya generado UNA celda que sube el presupuesto de turno
(LOCAL_ANALYZER_YIELD_SECONDS 60 -> 180). Palanca MECANICA, cero tokens de prompt.

Por que: en las corridas de 60 min sobre la base NVFP4, el 45-49% de los turnos
terminan en "Yielded control to solver: turn_time_budget" -- el modelo agota los
60 s pensando y el harness lo corta ANTES de que actue (control 220/546, consolidacion
254/597, manual 251/568). No es dispersion con herramientas (1.1 llamadas/turno): es el
pensamiento de Flash-Next contra el limite. Cada turno cortado es generacion y prefill
tirados sin accion.

Mecanica: tool_agent.py define _LOCAL_ANALYZER_YIELD_SECONDS como global de modulo y lo
lee en ToolAgent.__init__ (self._yield_seconds). Los agentes se construyen al jugar,
DESPUES del hook, asi que reasignar el global en la celda del hook llega a todos.
Verificacion en vivo: "yield_seconds: 180.0" en el bloque ANALYZER STATUS.

Uso:
  python scripts/build_nvfp4_yield.py nvfp4_carry_long   -> notebooks/nvfp4_carry_yield_long.ipynb
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOOK_ANCHOR = "PUBLIC25_SETTINGS budget_s="
YIELD = 180.0

CELL = f'''# PRESUPUESTO DE TURNO (nuestro): YIELD_SECONDS 60 -> {YIELD:.0f}. Palanca MECANICA, cero
# tokens de prompt. En las corridas de 60 min el 45-49% de los turnos se cortaban por
# turn_time_budget antes de actuar (el pensamiento de Flash-Next contra los 60 s).
# El global se lee en ToolAgent.__init__, y los agentes se construyen al jugar.
try:
    import os as _osy
    import inference.agent.tool_agent as _tay
    _osy.environ["LOCAL_ANALYZER_YIELD_SECONDS"] = "{YIELD:.0f}"
    _tay._LOCAL_ANALYZER_YIELD_SECONDS = {YIELD}
    print("YIELD_PATCH _LOCAL_ANALYZER_YIELD_SECONDS =", _tay._LOCAL_ANALYZER_YIELD_SECONDS, flush=True)
except Exception as exc:
    print("[yield_patch] failed, running stock: %s: %s" % (type(exc).__name__, exc), flush=True)
'''


def main() -> int:
    nombre = sys.argv[1] if len(sys.argv) > 1 else "nvfp4_carry_long"
    src = ROOT / "notebooks" / f"{nombre}.ipynb"
    out = ROOT / "notebooks" / f"{nombre.replace('_long', '')}_yield_long.ipynb"
    compile(CELL, "<celda-yield>", "exec")

    nb = json.loads(src.read_text(encoding="utf-8"))
    idx = [i for i, c in enumerate(nb["cells"])
           if c.get("cell_type") == "code" and HOOK_ANCHOR in "".join(c["source"])]
    if len(idx) != 1:
        print(f"esperaba 1 celda con el ancla del hook, hay {len(idx)}")
        return 1
    at = idx[0] + 1
    nb["cells"].insert(at, {"cell_type": "code", "execution_count": None, "metadata": {},
                            "outputs": [], "source": CELL.splitlines(keepends=True)})
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")

    orig = json.loads(src.read_text(encoding="utf-8"))["cells"]
    new = json.loads(out.read_text(encoding="utf-8"))["cells"]
    ok = len(new) == len(orig) + 1
    for i, c in enumerate(orig):
        j = i if i < at else i + 1
        if "".join(c["source"]) != "".join(new[j]["source"]):
            print(f"celda original {i} cambio")
            ok = False
    malas = []
    for i, c in enumerate(new, 1):
        if c.get("cell_type") != "code":
            continue
        try:
            compile("".join(c["source"]), f"<c{i}>", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        except SyntaxError as e:
            malas.append(f"celda {i}: {e}")
    print(f"{out.name}: celda yield en indice {at} | una sola celda anadida y las demas intactas: "
          f"{'OK' if ok else 'FALLA'} | compilan: {'OK' if not malas else malas}")
    return 0 if ok and not malas else 1


if __name__ == "__main__":
    sys.exit(main())

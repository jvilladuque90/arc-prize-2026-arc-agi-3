"""Anade a un notebook NVFP4 ya generado UNA celda que APAGA el pensamiento del modelo
(LOCAL_ANALYZER_ENABLE_THINKING true -> false). Palanca MECANICA, una variable, cero
tokens de prompt.

Por que (DESIGN 8.51): con dos puntos en la curva del presupuesto de pensamiento
por turno -- 60 s -> 26 niveles, 180 s -> 17 niveles -- la pendiente apunta a MENOS
pensamiento por turno, no mas. El corte a 60 s actua como limite forzoso util. El
siguiente punto natural es apagarlo: turnos cortos que siempre actuan, mas turnos y
acciones por hora; el riesgo es calidad por accion (la metrica castiga acciones por
nivel al cuadrado). Nunca medido sobre Flash-Next.

Mecanica: tool_agent.py define _LOCAL_ANALYZER_ENABLE_THINKING como global de modulo
y lo lee EN CADA PETICION (thinking=bool(_LOCAL_ANALYZER_ENABLE_THINKING)), asi que
reasignarlo en la celda del hook llega a todas las llamadas.

Uso:
  python scripts/build_nvfp4_nothink.py nvfp4_carry_long  -> notebooks/nvfp4_carry_nothink_long.ipynb
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOOK_ANCHOR = "PUBLIC25_SETTINGS budget_s="

CELL = '''# PENSAMIENTO APAGADO (nuestro): ENABLE_THINKING true -> false. Palanca MECANICA, una
# variable, cero tokens de prompt. Con 60 s de presupuesto salen 26 niveles y con 180 s
# solo 17: menos pensamiento por turno, no mas. El global se lee en cada peticion.
try:
    import os as _osn
    import inference.agent.tool_agent as _tan
    _osn.environ["LOCAL_ANALYZER_ENABLE_THINKING"] = "false"
    _tan._LOCAL_ANALYZER_ENABLE_THINKING = False
    print("NOTHINK_PATCH _LOCAL_ANALYZER_ENABLE_THINKING =", _tan._LOCAL_ANALYZER_ENABLE_THINKING, flush=True)
except Exception as exc:
    print("[nothink_patch] failed, running stock: %s: %s" % (type(exc).__name__, exc), flush=True)
'''


def main() -> int:
    nombre = sys.argv[1] if len(sys.argv) > 1 else "nvfp4_carry_long"
    src = ROOT / "notebooks" / f"{nombre}.ipynb"
    out = ROOT / "notebooks" / f"{nombre.replace('_long', '')}_nothink_long.ipynb"
    compile(CELL, "<celda-nothink>", "exec")

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
    print(f"{out.name}: celda nothink en indice {at} | una sola celda anadida y las demas intactas: "
          f"{'OK' if ok else 'FALLA'} | compilan: {'OK' if not malas else malas}")
    return 0 if ok and not malas else 1


if __name__ == "__main__":
    sys.exit(main())
